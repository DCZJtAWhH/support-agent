import asyncio
from uuid import uuid4

from langchain_core.tools import tool

from support_agent.database import Database
from support_agent.repositories import ChatRepository
from support_agent.tools import ToolRegistry, build_tools


class ReturnPolicyRetriever:
    async def search(self, keyword):
        return [{"question": "退货政策是什么", "answer": "签收后7天内申请。", "category": "售后"}]


def test_business_tools_validate_literal_keyword_and_deduplicate_ticket(tmp_path):
    async def scenario():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'tools.db'}")
        try:
            await db.initialize(seed=True)
            repo = ChatRepository(db, ReturnPolicyRetriever())
            cid, rid = str(uuid4()), str(uuid4())
            await repo.begin_request(cid, rid, "退货政策是什么，请创建人工工单")
            registry = ToolRegistry(build_tools(repo, cid, rid, "退货政策是什么，请创建人工工单"))
            faq = await registry.execute("query_faq", {"keyword": "退货"})
            assert faq["ok"] and "7天" in faq["data"]["matches"][0]["answer"]
            invalid = await registry.execute("query_faq", {"keyword": "运费"})
            assert invalid["error"]["code"] == "invalid_arguments"
            assert (await registry.execute("query_order", {"order_id": 1001}))["ok"] is False
            assert (await registry.execute("missing", {}))["error"]["code"] == "unknown_tool"
            first = await registry.execute(
                "create_ticket", {"description": "退货咨询", "ticket_type": "其他"}, confirmed=True
            )
            second = await registry.execute(
                "create_ticket", {"description": "退货咨询", "ticket_type": "其他"}, confirmed=True
            )
            assert first["data"]["ticket_no"] == second["data"]["ticket_no"]
            assert "query_logistics" not in registry.tools  # 物流只由外部 MCP Server 提供。
        finally:
            await db.close()

    asyncio.run(scenario())


def test_executor_retries_timeout_but_not_validation_or_permanent_error():
    async def scenario():
        attempts = 0

        @tool
        async def transient(value: str) -> dict:
            """模拟第一次外部访问超时，第二次恢复。"""
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                await asyncio.sleep(1)
            return {"value": value}

        @tool
        async def broken() -> dict:
            """模拟不可重试异常，错误详情不得进入工具结果。"""
            raise ValueError("private-secret")

        registry = ToolRegistry([transient, broken], timeout=0.01, retries=1, retry_delay=0)
        result = await registry.execute("transient", {"value": "成功"})
        assert result["ok"] and result["attempts"] == 2
        assert attempts == 2
        result = await registry.execute("broken", {})
        assert result["error"]["code"] == "execution_failed"
        assert "private-secret" not in str(result)

    asyncio.run(scenario())


def test_executor_stops_after_timeout_retry_limit():
    async def scenario():
        @tool
        async def unavailable() -> dict:
            """模拟持续超时的外部查询。"""
            await asyncio.sleep(1)
            return {}

        result = await ToolRegistry([unavailable], timeout=0.01, retries=1, retry_delay=0).execute(
            "unavailable", {}
        )
        assert result["error"]["code"] == "timeout"
        assert result["attempts"] == 2

    asyncio.run(scenario())
