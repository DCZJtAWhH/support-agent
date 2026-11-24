"""在独立临时数据库演示只读重试与写超时结果不确定，不连接真实业务库。"""

import asyncio
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.database import Database  # noqa: E402
from support_agent.repositories import ChatRepository  # noqa: E402
from support_agent.tools import ToolRegistry  # noqa: E402


async def main():
    """故意制造网络等待，输出引擎兜底、审计次数和已写入的临时工单凭据。"""
    with TemporaryDirectory(prefix="support-agent-ch08-") as directory:
        db = Database(f"sqlite+aiosqlite:///{Path(directory) / 'demo.db'}")
        await db.initialize()
        repo = ChatRepository(db)
        cid, rid = str(uuid4()), str(uuid4())
        await repo.begin_request(cid, rid, "演示写超时")

        @tool
        async def slow_read() -> dict:
            """模拟只读网络持续无响应。"""
            await asyncio.sleep(5)
            return {"value": 1}

        @tool
        async def create_ticket(description: str, ticket_type: str) -> dict:
            """先在临时库写入，再模拟响应丢失，证明超时并不代表写入失败。"""
            result = await repo.create_ticket(cid, rid, description, ticket_type)
            await asyncio.sleep(5)
            return result

        registry = ToolRegistry(
            [slow_read, create_ticket],
            repo=repo,
            conversation_id=cid,
            timeout=0.05,
            retries=1,
            retry_delay=0.01,
        )
        try:
            read = await registry.execute("slow_read", {}, tool_call_id="read-timeout")
            write = await registry.execute(
                "create_ticket",
                {"description": "演示水壶不加热", "ticket_type": "维修"},
                tool_call_id="write-timeout",
                confirmed=True,
            )
            audit = await repo.read_conversation(cid)
            rows = audit["tool_audit_logs"]
            assert [r["status"] for r in rows] == ["超时", "超时"]
            assert [r["retry_count"] for r in rows] == [1, 0]
            assert len(audit["tickets"]) == 1
            report = {
                "read_result": read,
                "write_result": write,
                "audit_logs": rows,
                "ticket_count_after_write_timeout": len(audit["tickets"]),
                "passed": True,
                "note": "全部数据来自独立临时库，演示结束后清理；超时写操作未自动重试。",
            }
            (ROOT / "dev-notes/ch08-timeouts.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            await db.close()


if __name__ == "__main__":
    asyncio.run(main())
