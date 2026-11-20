"""对真实聊天服务核验拒答、来源引用与低置信度问题池，生成可复核证据。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_tools import parse_events  # noqa: E402

from support_agent.config import Settings  # noqa: E402
from support_agent.database import Database  # noqa: E402
from support_agent.repositories import ChatRepository  # noqa: E402


async def main(args):
    """使用当前 .env 数据库核对同源本地服务；请求重放不重复入池。"""
    config = Settings()
    database = Database(config.database_url.get_secret_value())
    report = []
    try:
        repo = ChatRepository(database)
        async with httpx.AsyncClient(
            base_url=args.base_url, timeout=180, trust_env=False
        ) as client:
            for question in ("火星殖民飞船燃料补贴是多少", "BX-500容量是多少，能保温几个小时"):
                rid = str(uuid4())
                body = {"message": question, "request_id": rid}
                response = await client.post("/chat", json=body)
                response.raise_for_status()
                events = parse_events(response.text)
                assert events[-1][0] == "done", events
                answer = "".join(d["text"] for e, d in events if e == "token")
                assert "无法确认" in answer, answer
                for _ in range(20):
                    replay = await client.post("/chat", json=body)
                    if replay.status_code != 409:
                        break
                    # done 已送达后数据库收尾仍可能在进行，遵守服务的同轮占用约束。
                    await asyncio.sleep(0.1)
                replay.raise_for_status()
                assert parse_events(replay.text)[-1][0] == "done"
                pool = [r for r in await repo.low_confidence_questions() if r["request_id"] == rid]
                assert len(pool) == 1, pool
                assert pool[0]["question"] == question and pool[0]["reason"]
                assert pool[0]["conversation_id"] == events[0][1]["conversation_id"]
                if question.startswith("BX-500"):
                    assert pool[0]["entry_point"] == "generation", pool
                report.append(
                    {
                        "question": question,
                        "answer": answer,
                        "pool": pool[0],
                        "idempotent_replay": True,
                    }
                )
        output = Path(args.output)
        await asyncio.to_thread(
            output.write_text,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("真实拒答及入池验收通过：未知问题、局部相关但证据不足、会话关联、重试幂等。")
    finally:
        await database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument("--output", default="dev-notes/ch04-acceptance-results.json")
    asyncio.run(main(parser.parse_args()))
