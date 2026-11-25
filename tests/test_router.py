import asyncio
import json
from uuid import uuid4

import pytest
from conftest import sse_events, stream_response
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from support_agent.config import Settings
from support_agent.database import Database
from support_agent.errors import ModelResponseError
from support_agent.repositories import ChatRepository
from support_agent.workflow import SupportWorkflow


class RouterModel:
    """模拟模型上游，约束 HTTP 以外的模型消息边界。"""

    def __init__(self, resolved, intent="其他", confidence=0.93):
        self.resolved, self.intent, self.confidence = resolved, intent, confidence
        self.inputs = []

    def bind(self, **options):
        self.options = options
        return self

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.inputs.append(messages)
        prompt = messages[0].content
        if "问题理解器" in prompt:
            result = {"question": self.resolved}
        elif "将顾客" in prompt:
            result = {"intent": self.intent, "confidence": self.confidence}
        elif "查询扩写器" in prompt:
            result = {"queries": ["退货条件", "退货期限", "退款例外"]}
        elif "生成证据自评" in prompt:
            result = {"sufficient": True, "reason": "测试证据支持回答"}
        else:
            yield AIMessageChunk(
                content="这是演示订单，请结合签收时间核对退货条件。[1]",
                response_metadata={"finish_reason": "stop"},
            )
            return
        yield AIMessageChunk(
            content=json.dumps(result, ensure_ascii=False),
            response_metadata={"finish_reason": "stop"},
        )


def settings():
    return Settings(
        _env_file=None,
        openai_base_url="https://unused.invalid/v1",
        openai_api_key="test",
        openai_model="test",
        database_url="sqlite+aiosqlite:///:memory:",
        checkpoint_path=":memory:",
    )


def test_workflow_exposes_resolved_question_and_other_without_forcing_business_intent():
    async def scenario():
        database = Database(settings().database_url.get_secret_value())
        await database.initialize()
        repo = ChatRepository(database)
        cid, rid = str(uuid4()), str(uuid4())
        await repo.begin_request(cid, rid, "月亮能申请保价吗")
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(RouterModel("月亮能申请保价吗"), repo, settings(), saver)
            events = [e async for e in workflow.stream(cid, rid, "月亮能申请保价吗")]
            understanding = next(e for e in events if e["event"] == "understanding")
            assert understanding == {
                "event": "understanding",
                "question": "月亮能申请保价吗",
                "intent": "其他",
                "confidence": 0.93,
            }
            assert "agent" not in [e.get("node") for e in events]
            assert any(e.get("text") for e in events)
        await database.close()

    asyncio.run(scenario())


def test_refund_pauses_without_guessing_order_and_resumes_after_restart(tmp_path):
    class PolicyAPI:
        async def search_many(self, question, queries, **kwargs):
            assert "BX-500" in question
            assert len(queries) >= 2
            return [
                {
                    "chunk_id": 1,
                    "content_hash": "v1",
                    "score": 0.95,
                    "content_type": "policy",
                    "text": "签收7天内商品完好可退货",
                    "answer": "签收7天内商品完好可退货",
                    "section_path": "退货条件",
                }
            ]

    async def scenario():
        db = Database(settings().database_url.get_secret_value())
        await db.initialize()
        repo = ChatRepository(db, PolicyAPI())
        cid, rid = str(uuid4()), str(uuid4())
        await repo.begin_request(cid, rid, "这个能退吗")
        from langchain_core.messages import AIMessage, HumanMessage

        history = [HumanMessage("BX-500容量多少"), AIMessage("BX-500为500毫升")]
        path = str(tmp_path / "graph.sqlite")
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            workflow = SupportWorkflow(
                RouterModel("BX-500能退吗", "退款退货"), repo, settings(), saver
            )
            events = [
                e async for e in workflow.stream(cid, rid, "这个能退吗", initial_history=history)
            ]
            picker = next(e for e in events if e["event"] == "order_required")
            assert picker["orders"] and all(o["simulated"] for o in picker["orders"])
            assert not any(e.get("node") in {"retrieve", "agent"} for e in events)
            await workflow.commit(cid, rid)  # 暂停不能意外自动恢复或提交本轮历史。
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            workflow = SupportWorkflow(RouterModel("不应重新理解", "其他"), repo, settings(), saver)
            events = [
                e
                async for e in workflow.stream(
                    cid,
                    rid,
                    "这个能退吗",
                    selected_order_id="1001",
                )
            ]
            nodes = [e["node"] for e in events if e["event"] == "workflow"]
            assert "resolve" not in nodes and "classify" not in nodes
            assert nodes.index("load_order") < nodes.index("retrieve") < nodes.index("agent")
            assert any(e["event"] == "refund_form" and e["order_id"] == "1001" for e in events)
            assert not any(e["event"] == "order_required" for e in events)
        await db.close()

    asyncio.run(scenario())


def test_chat_selection_is_bound_to_original_turn_and_replay_does_not_rerun(harness):
    async def handler(request, payload):
        prompt = payload["messages"][0]["content"]
        if "问题理解器" in prompt:
            result = {"question": "这个能退吗"}
        elif "将顾客" in prompt:
            result = {"intent": "退款退货", "confidence": 0.98}
        elif "查询扩写器" in prompt:
            result = {"queries": ["退货期限", "退款条件"]}
        elif "生成证据自评" in prompt:
            result = {"sufficient": True, "reason": "测试证据支持回答"}
        else:
            return stream_response(("演示订单签收3天，商品完好时可以申请退货。[1]",))
        return stream_response((json.dumps(result, ensure_ascii=False),))

    class PolicyAPI:
        async def search_many(self, question, queries, **kwargs):
            return [
                {
                    "chunk_id": 1,
                    "score": 0.95,
                    "content_type": "policy",
                    "text": "签收7天内商品完好可退货",
                    "section_path": "退货条件",
                }
            ]

    async def scenario():
        async with harness(handler, structured_output_method="json_mode") as (client, calls, app):
            app.state.repository.retriever = PolicyAPI()
            rid = str(uuid4())
            body = {"message": "这个能退吗", "request_id": rid}
            first = sse_events(await client.post("/chat", json=body))
            cid = first[0][1]["conversation_id"]
            body["conversation_id"] = cid
            assert any(e == "order_required" for e, _ in first), first
            before = len(calls)
            invalid = await client.post("/chat", json={**body, "selected_order_id": "9999"})
            assert invalid.status_code == 409 and len(calls) == before
            picked = {**body, "selected_order_id": "1001"}
            resumed = sse_events(await client.post("/chat", json=picked))
            assert resumed[-1][0] == "done"
            assert any(e == "refund_form" for e, _ in resumed)
            assert not any(e == "order_required" for e, _ in resumed)
            before = len(calls)
            replay = sse_events(await client.post("/chat", json=picked))
            assert replay[-1][0] == "done" and len(calls) == before
            changed = await client.post("/chat", json={**body, "selected_order_id": "1002"})
            assert changed.status_code == 409

    asyncio.run(scenario())


def test_refund_submission_requires_completed_policy_turn_and_fixed_reason(harness):
    async def handler(request, payload):
        prompt = payload["messages"][0]["content"]
        if "问题理解器" in prompt:
            result = {"question": "订单1001能退吗"}
        elif "将顾客" in prompt:
            result = {"intent": "退款退货", "confidence": 0.99}
        elif "查询扩写器" in prompt:
            result = {"queries": ["退货条件", "退货期限"]}
        elif "生成证据自评" in prompt:
            result = {"sufficient": True, "reason": "测试证据支持回答"}
        else:
            return stream_response(("演示订单签收3天，商品完好时可申请退货。[1]",))
        return stream_response((json.dumps(result, ensure_ascii=False),))

    class PolicyAPI:
        async def search_many(self, question, queries, **kwargs):
            return [
                {
                    "chunk_id": 1,
                    "score": 0.95,
                    "content_type": "policy",
                    "text": "签收7天内商品完好可退货",
                    "section_path": "退货条件",
                }
            ]

    async def scenario():
        async with harness(handler, structured_output_method="json_mode") as (client, _, app):
            app.state.repository.retriever = PolicyAPI()
            rid = str(uuid4())
            events = sse_events(
                await client.post(
                    "/chat",
                    json={
                        "message": "订单1001能退吗",
                        "request_id": rid,
                    },
                )
            )
            cid = events[0][1]["conversation_id"]
            body = {"conversation_id": cid, "request_id": rid, "reason": "不想要了"}
            assert (
                await client.post("/actions/refund", json={**body, "reason": "自由编造"})
            ).status_code == 422
            assert (
                await client.post("/actions/refund", json={**body, "conversation_id": str(uuid4())})
            ).status_code == 409
            first = await client.post("/actions/refund", json=body)
            assert first.status_code == 200, first.text
            second = await client.post("/actions/refund", json=body)
            assert first.json() == second.json()
            assert first.json()["order_id"] == "1001" and first.json()["status"] == "pending"
            assert (
                await client.post("/actions/refund", json={**body, "reason": "商品质量问题"})
            ).status_code == 409
            audit = (await client.get(f"/conversations/{cid}")).json()
            assert len(audit["refunds"]) == 1

    asyncio.run(scenario())


def test_multiple_explicit_orders_cannot_be_silently_reduced_to_first_order():
    async def scenario():
        db = Database(settings().database_url.get_secret_value())
        await db.initialize()
        repo = ChatRepository(db)
        cid, rid = str(uuid4()), str(uuid4())
        question = "订单1001和1002能退吗"
        await repo.begin_request(cid, rid, question)
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(RouterModel(question, "退款退货"), repo, settings(), saver)
            events = [e async for e in workflow.stream(cid, rid, question)]
            assert any(e["event"] == "order_required" for e in events)
            assert not any(e.get("node") == "load_order" for e in events)
        await db.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "output",
    [
        '{"intent":"物流"}',
        '{"intent":"物流","confidence":"0.9"}',
        '{"intent":"物流","confidence":true}',
        '{"intent":"物流","confidence":1.1}',
        '{"intent":"物流","confidence":NaN}',
        '{"intent":"不存在","confidence":0.9}',
        '{"intent":"物流","confidence":0.9,"order_id":"1001"}',
        '```json\n{"intent":"物流","confidence":0.9}\n```',
    ],
)
def test_workflow_rejects_invalid_intent_json_before_business_execution(output):
    class InvalidModel(RouterModel):
        async def astream(self, messages):
            if "将顾客" in messages[0].content:
                yield AIMessageChunk(content=output, response_metadata={"finish_reason": "stop"})
            else:
                async for chunk in super().astream(messages):
                    yield chunk

    async def scenario():
        db = Database(settings().database_url.get_secret_value())
        await db.initialize()
        repo = ChatRepository(db)
        cid, rid = str(uuid4()), str(uuid4())
        await repo.begin_request(cid, rid, "订单1001物流在哪")
        events = []
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(InvalidModel("订单1001物流在哪"), repo, settings(), saver)
            with pytest.raises(ModelResponseError):
                async for event in workflow.stream(cid, rid, "订单1001物流在哪"):
                    events.append(event)
            assert not any(e["event"] in {"token", "tool_status"} for e in events)
        await db.close()

    asyncio.run(scenario())


def test_core_agent_cannot_switch_to_a_different_order_even_if_it_requests_a_tool():
    class MisbehavingModel(RouterModel):
        async def astream(self, messages):
            if not messages[0].content.startswith("你是本店"):
                async for chunk in super().astream(messages):
                    yield chunk
            elif messages[-1].type != "tool":
                yield AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "query_order",
                            "args": '{"order_id":"1002"}',
                            "id": "wrong-order",
                            "index": 0,
                        }
                    ],
                    response_metadata={"finish_reason": "tool_calls"},
                )
            else:
                assert json.loads(messages[-1].content)["error"]["code"] == "permission_denied"
                yield AIMessageChunk(
                    content="仅核对演示订单1001及政策。[1]",
                    response_metadata={"finish_reason": "stop"},
                )

    class PolicyAPI:
        async def search_many(self, question, queries, **kwargs):
            return [
                {
                    "chunk_id": 1,
                    "score": 0.95,
                    "content_type": "policy",
                    "text": "签收7天内商品完好可退货",
                    "section_path": "退货条件",
                }
            ]

    async def scenario():
        db = Database(settings().database_url.get_secret_value())
        await db.initialize()
        repo = ChatRepository(db, PolicyAPI())
        cid, rid = str(uuid4()), str(uuid4())
        question = "订单1001能退吗"
        await repo.begin_request(cid, rid, question)
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(
                MisbehavingModel(question, "退款退货"), repo, settings(), saver
            )
            events = [e async for e in workflow.stream(cid, rid, question)]
            assert any(
                e.get("tool_call_id") == "wrong-order" and e.get("state") == "failed"
                for e in events
            )
        await db.close()

    asyncio.run(scenario())
