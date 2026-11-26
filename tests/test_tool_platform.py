import asyncio

import pytest
from langchain_core.tools import tool
from module_support import tool_registry as ToolRegistry


def test_registered_pydantic_v1_tool_executes_and_records_audit():
    from langchain_core.tools import StructuredTool
    from pydantic.v1 import BaseModel

    from support_agent.platform.database import Database
    from support_agent.tooling import ToolAuditStore
    from support_agent.tooling import ToolRegistry as PublicRegistry

    class LegacyInput(BaseModel):
        value: str

    async def lookup(value: str):
        """模拟使用 Pydantic v1 参数结构的外部只读工具。"""
        return {"value": value}

    async def scenario():
        database = Database("sqlite+aiosqlite:///:memory:")
        await database.initialize()
        audit = ToolAuditStore(database)
        try:
            item = StructuredTool.from_function(coroutine=lookup, args_schema=LegacyInput)
            registry = PublicRegistry([item], repo=audit, conversation_id="legacy-schema")
            result = await registry.execute("lookup", {"value": "已查询"}, tool_call_id="v1")
            assert result["ok"] and result["data"] == {"value": "已查询"}
            assert (await audit.tool_audit_logs("legacy-schema"))[0]["status"] == "成功"
        finally:
            await database.close()

    asyncio.run(scenario())


@tool(
    args_schema={
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 3}},
        "required": ["count"],
        "additionalProperties": False,
    }
)
async def registered_later(count: int) -> dict:
    """查询指定数量的演示商品。"""
    return {"count": count, "说明": "注册后立即可用"}


@pytest.mark.parametrize("arguments", [{}, {"count": "2"}, {"count": 4}])
def test_registered_json_schema_tool_rejects_missing_type_and_range(arguments):
    async def scenario():
        registry = ToolRegistry([])
        registry.register(registered_later)
        result = await registry.execute("registered_later", arguments, tool_call_id="call-1")
        assert not result["ok"]
        assert result["error"]["code"] == "invalid_arguments"
        assert "count" in result["error"]["message"]
        valid = await registry.execute("registered_later", {"count": 2})
        assert valid["data"]["说明"] == "注册后立即可用"

    asyncio.run(scenario())


def test_cancelled_call_is_audited_and_cancellation_still_propagates(tmp_path):
    async def scenario():
        from module_support import TestServices as ChatRepository

        from support_agent.platform.database import Database

        started = asyncio.Event()

        @tool
        async def cancelled_read() -> dict:
            """模拟查询尚未返回时顾客断开连接。"""
            started.set()
            await asyncio.sleep(60)
            return {"value": 1}

        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'cancelled.db'}")
        await db.initialize()
        repo = ChatRepository(db)
        try:
            registry = ToolRegistry([cancelled_read], repo=repo, conversation_id="cancelled")
            task = asyncio.create_task(
                registry.execute("cancelled_read", {}, tool_call_id="cancel")
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            rows = await repo.tool_audit_logs("cancelled")
            assert len(rows) == 1 and rows[0]["status"] == "失败"
            assert rows[0]["retry_count"] == 0 and "取消" in rows[0]["error"]
        finally:
            await db.close()

    asyncio.run(scenario())


def test_slow_audit_storage_cannot_hold_a_successful_tool_result():
    class SlowAuditStorage:
        async def record_tool_audit(self, **values):
            await asyncio.sleep(60)

    async def scenario():
        registry = ToolRegistry([registered_later], repo=SlowAuditStorage(), audit_timeout=0.01)
        result = await asyncio.wait_for(registry.execute("registered_later", {"count": 1}), 0.2)
        assert result["ok"]

    asyncio.run(scenario())


def test_new_tool_is_usable_in_chat_without_adding_a_route(harness):
    import json

    from conftest import sse_events, stream_response, tool_response

    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"商品咨询","confidence":0.99,"tool_query":true}',))
        if payload["messages"][-1]["role"] == "tool":
            result = json.loads(payload["messages"][-1]["content"])
            assert result["data"]["count"] == 2
            return stream_response(("已查询2件演示商品。",))
        return tool_response([{"name": "registered_later", "args": {"count": 2}, "id": "hot"}])

    async def scenario():
        async with harness(handler) as (client, _, app):
            app.state.chat.workflow.catalog.register(registered_later)
            response = await client.post("/chat", json={"message": "查询2件商品"})
            assert sse_events(response)[-1][0] == "done", response.text
            assert "已查询2件" in response.text
            assert '"node": "retrieve"' not in response.text

    asyncio.run(scenario())


def test_empty_result_is_not_retried_and_output_uses_local_projection():
    async def scenario():
        calls = []

        @tool
        async def lookup() -> dict:
            """查询不到数据是正常业务结果。"""
            calls.append(1)
            return {"found": False}

        registry = ToolRegistry([lookup], retries=2)
        empty = await registry.execute("lookup", {})
        assert empty["error"]["code"] == "not_found"
        assert empty["error"]["category"] == "查询落空"
        assert len(calls) == 1

        @tool
        async def receipt() -> dict:
            """模拟带内部信息的工单查询回执。"""
            return {
                "ticket_no": "T123",
                "status": "pending",
                "description": "pending",
                "internal_note": "隐藏",
            }

        registry.register(receipt, result_fields=["ticket_no", "status"])
        result = await registry.execute("receipt", {})
        assert result["data"] == {"ticket_no": "T123", "status": "待处理"}
        full = await ToolRegistry([receipt]).execute("receipt", {})
        assert full["data"]["description"] == "pending"  # 用户原文不能被当作枚举翻译。

    asyncio.run(scenario())


def test_timeout_audit_has_retries_elapsed_and_write_denial(tmp_path):
    async def scenario():
        from module_support import TestServices as ChatRepository

        from support_agent.platform.database import Database

        @tool
        async def slow_read() -> dict:
            """模拟只读网络查询超时。"""
            await asyncio.sleep(1)
            return {"value": 1}

        @tool
        async def create_ticket(description: str) -> dict:
            """模拟写操作超时。"""
            await asyncio.sleep(1)
            return {}

        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'timeouts.db'}")
        await db.initialize()
        repo = ChatRepository(db)
        try:
            registry = ToolRegistry(
                [slow_read, create_ticket],
                timeout=0.01,
                retry_delay=0,
                retries=1,
                repo=repo,
                conversation_id="timeouts",
            )
            await registry.execute("slow_read", {}, tool_call_id="read")
            await registry.execute("create_ticket", {"description": "损坏"}, tool_call_id="denied")
            await registry.execute(
                "create_ticket",
                {"description": "损坏"},
                tool_call_id="write",
                confirmed=True,
            )
            rows = await repo.tool_audit_logs("timeouts")
            assert [r["status"] for r in rows] == ["超时", "权限拒绝", "超时"]
            assert [r["retry_count"] for r in rows] == [1, 0, 0]
            assert rows[0]["duration_ms"] >= 20 and rows[2]["duration_ms"] >= 10
        finally:
            await db.close()

    asyncio.run(scenario())


def test_catalog_registration_is_visible_to_next_agent_without_rebuild():
    async def scenario():
        from support_agent.dialogue.agent import agent_tools
        from support_agent.tooling.catalog import ToolCatalog

        catalog = ToolCatalog()
        first = catalog.bind("c", "r", "查询商品")
        assert "registered_later" not in first.tools
        catalog.register(registered_later)
        second = catalog.bind("c", "r2", "查询商品")
        assert "registered_later" in {t.name for t in agent_tools(second)}
        assert (await second.execute("registered_later", {"count": 1}))["ok"]

    asyncio.run(scenario())


def test_mcp_permissions_come_from_local_policy_and_cannot_impersonate_ticket():
    async def scenario():
        registry = ToolRegistry([])
        registry.register(registered_later, source="mcp:untrusted", permission="deny")
        result = await registry.execute("registered_later", {"count": 1}, confirmed=True)
        assert result["error"]["code"] == "permission_denied"

        @tool
        async def create_ticket(description: str) -> dict:
            """声称只读但实际写入的外部工具。"""
            pytest.fail("不应执行外部写操作")

        registry.register(create_ticket, source="mcp:untrusted", permission="read")
        denied = await registry.execute("create_ticket", {"description": "测试"}, confirmed=True)
        assert denied["error"]["code"] == "permission_denied"

    asyncio.run(scenario())


def test_every_terminal_outcome_is_audited_even_when_audit_storage_fails(tmp_path):
    async def scenario():
        from module_support import TestServices as ChatRepository

        from support_agent.platform.database import Database

        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
        await db.initialize()
        repo = ChatRepository(db)
        try:
            registry = ToolRegistry([], repo=repo, conversation_id="no-foreign-key")
            registry.register(registered_later)
            await registry.execute("registered_later", {}, tool_call_id="invalid")
            await registry.execute("registered_later", {"count": 1}, tool_call_id="success")
            rows = await repo.tool_audit_logs("no-foreign-key")
            assert [r["status"] for r in rows] == ["校验拦下", "成功"]
            assert rows[0]["tool_call_id"] == "invalid"
            assert rows[1]["source"] == "builtin"
            assert rows[1]["arguments"] == {"count": 1}
            assert rows[1]["retry_count"] == 0
            assert rows[1]["duration_ms"] >= 0 and rows[1]["created_at"]
            assert "注册后立即可用" in rows[1]["result_summary"]
            from support_agent.tooling.models import ToolAuditLog

            assert not ToolAuditLog.__table__.foreign_keys
            async with db.engine.begin() as connection:
                await connection.run_sync(ToolAuditLog.__table__.drop)
            result = await registry.execute("registered_later", {"count": 2})
            assert result["ok"]  # 审计库不可写不能逆转工具成功。
        finally:
            await db.close()

    asyncio.run(scenario())


def test_write_requires_engine_confirmation_and_never_retries_timeout():
    async def scenario():
        calls = []

        @tool
        async def create_ticket(description: str, ticket_type: str) -> dict:
            """模拟已发送写请求后连接超时。"""
            calls.append(description)
            await asyncio.sleep(1)
            return {}

        registry = ToolRegistry([create_ticket], timeout=0.01, retries=2, retry_delay=0)
        arguments = {"description": "商品损坏，请维修", "ticket_type": "维修"}
        denied = await registry.execute("create_ticket", arguments)
        assert denied["error"]["code"] == "permission_denied"
        assert not calls
        timed_out = await registry.execute("create_ticket", arguments, confirmed=True)
        assert timed_out["error"]["code"] == "timeout"
        assert timed_out["attempts"] == 1
        assert len(calls) == 1

    asyncio.run(scenario())
