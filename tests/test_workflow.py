import asyncio
import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from test_agent_loop import ScriptedModel

from support_agent.config import Settings
from support_agent.database import Database
from support_agent.repositories import ChatRepository
from support_agent.workflow import SupportWorkflow


class IntentModel:
    def __init__(self, intent):
        self.intent = intent
        self.calls = []

    def bind(self, **kwargs):
        return self

    async def astream(self, messages):
        self.calls.append(messages)
        if "问题理解器" in messages[0].content:
            result = {"question": messages[-1].content}
        elif "查询扩写器" in messages[0].content:
            result = {"queries": ["退货条件", "退货期限"]}
        else:
            result = {"intent": self.intent, "confidence": 0.99}
        yield AIMessageChunk(
            content=json.dumps(result, ensure_ascii=False),
            response_metadata={"finish_reason": "stop"},
        )


def settings(**overrides):
    return Settings(
        _env_file=None,
        openai_base_url="https://unused.invalid/v1",
        openai_api_key="test",
        openai_model="test",
        database_url="sqlite+aiosqlite:///:memory:",
        **overrides,
    )


def test_complaint_offers_two_independent_actions_without_creating_ticket():
    async def scenario():
        database = Database("sqlite+aiosqlite:///:memory:")
        await database.initialize()
        cid, rid = str(uuid4()), str(uuid4())
        repo = ChatRepository(database)
        await repo.begin_request(cid, rid, "我要投诉")
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(IntentModel("投诉"), repo, settings(), saver)
            events = [e async for e in workflow.stream(cid, rid, "我要投诉")]
            actions = next(e for e in events if e["event"] == "actions")
            assert actions["actions"] == ["handoff", "create_ticket"]
            assert "agent" not in [e.get("node") for e in events]
            assert not (await repo.read_conversation(cid))["tickets"]
        await database.close()

    asyncio.run(scenario())


def test_core_workflow_loads_order_before_policy_and_agent_only_judges():
    class SearchAPI:
        async def search_many(self, question, queries, **kwargs):
            return [
                {
                    "chunk_id": 1,
                    "score": 0.95,
                    "answer": "签收后7天可申请退货。",
                    "text": "签收后7天可申请退货。",
                    "section_path": "退货政策",
                }
            ]

    class RefundModel(IntentModel):
        def bind_tools(self, tools):
            assert "create_ticket" not in {t.name for t in tools}
            return self

        async def astream(self, messages):
            if not messages[0].content.startswith("你是本店"):
                async for chunk in super().astream(messages):
                    yield chunk
            else:
                assert "签收后7天" in messages[0].content
                assert '"order_id": "1001"' in messages[0].content
                yield AIMessageChunk(
                    content="演示订单已查询；签收后7天可申请退货。[1]",
                    response_metadata={"finish_reason": "stop"},
                )

    async def scenario():
        database = Database("sqlite+aiosqlite:///:memory:")
        await database.initialize()
        cid, rid = str(uuid4()), str(uuid4())
        repo = ChatRepository(database, SearchAPI())
        await repo.begin_request(cid, rid, "订单1001可退货吗")
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(RefundModel("退款退货"), repo, settings(), saver)
            events = [e async for e in workflow.stream(cid, rid, "订单1001可退货吗")]
            nodes = [e["node"] for e in events if e["event"] == "workflow"]
            assert nodes.index("load_order") < nodes.index("retrieve") < nodes.index("agent")
            assert "tools" not in nodes
            assert any(e.get("name") == "query_order" for e in events)
            assert not await repo.low_confidence_questions()
        await database.close()

    asyncio.run(scenario())


def test_file_checkpointer_restores_history_after_workflow_is_reopened(tmp_path):
    async def scenario():
        database = Database("sqlite+aiosqlite:///:memory:")
        await database.initialize()
        repo = ChatRepository(database)
        cid, rid = str(uuid4()), str(uuid4())
        await repo.begin_request(cid, rid, "你好")
        path = str(tmp_path / "checkpoints.sqlite")
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            workflow = SupportWorkflow(IntentModel("闲聊"), repo, settings(), saver)
            _ = [e async for e in workflow.stream(cid, rid, "你好")]
            await workflow.commit(cid, rid)
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            reopened = SupportWorkflow(IntentModel("闲聊"), repo, settings(), saver)
            snapshot = await reopened.graph.aget_state({"configurable": {"thread_id": cid}})
            assert snapshot.values["history"][0].content == "你好"
            assert len(snapshot.values["history"]) == 2
        await database.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("intent", ["物流", "订单"])
def test_business_intents_skip_retrieval_and_agent_can_take_multiple_steps(intent):
    class BusinessModel(IntentModel, ScriptedModel):
        def bind_tools(self, tools):
            return ScriptedModel.bind_tools(self, tools)

        async def astream(self, messages):
            if not messages[0].content.startswith("你是本店"):
                async for chunk in IntentModel.astream(self, messages):
                    yield chunk
            else:
                async for chunk in ScriptedModel.astream(self, messages):
                    yield chunk

    async def scenario():
        database = Database("sqlite+aiosqlite:///:memory:")
        await database.initialize()
        cid, rid = str(uuid4()), str(uuid4())
        repo = ChatRepository(database)
        await repo.begin_request(cid, rid, "先查订单1001再查物流")
        model = BusinessModel(intent)
        model.inputs = []
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(model, repo, settings(), saver)
            events = [e async for e in workflow.stream(cid, rid, "先查订单1001再查物流")]
            nodes = [e["node"] for e in events if e["event"] == "workflow"]
            assert "retrieve" not in nodes and "confidence" not in nodes
            assert nodes.count("agent") == 3 and nodes.count("tools") == 2
            assert "运输中" in "".join(e["text"] for e in events if e["event"] == "token")
        await database.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("intent", ["商品咨询", "退款退货", "售后"])
def test_knowledge_always_retrieves_and_weak_evidence_refuses_before_agent(intent):
    class WeakRetriever:
        async def search(self, question, **kwargs):
            return [{"chunk_id": 1, "answer": "不相关", "score": 0.01}]

        async def search_many(self, question, queries, **kwargs):
            return await self.search(question)

    async def scenario():
        database = Database("sqlite+aiosqlite:///:memory:")
        await database.initialize()
        cid, rid = str(uuid4()), str(uuid4())
        repo = ChatRepository(database, WeakRetriever())
        await repo.begin_request(cid, rid, "退货政策是什么")
        model = IntentModel(intent)
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(model, repo, settings(), saver)
            question = "退货政策是什么" if intent == "商品咨询" else "订单1001能退吗"
            events = [e async for e in workflow.stream(cid, rid, question)]
            nodes = [e["node"] for e in events if e["event"] == "workflow"]
            expected = [
                "resolve",
                "classify",
                "route",
                "retrieve",
                "confidence",
                "fallback",
                "log",
            ]
            if intent != "商品咨询":
                expected[3:3] = ["select_order", "load_order"]
            assert nodes == expected
            assert len(model.calls) == (2 if intent == "商品咨询" else 3)
            assert "可靠依据" in "".join(e["text"] for e in events if e["event"] == "token")
            assert len(await repo.low_confidence_questions()) == 1
        await database.close()

    asyncio.run(scenario())
