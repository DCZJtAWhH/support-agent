import asyncio
import json
import logging
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from test_session_context import config

from support_agent.database import Database
from support_agent.repositories import ChatRepository
from support_agent.workflow import SupportWorkflow


class ConversationModel:
    def __init__(self):
        self.contexts = []

    def bind(self, **kwargs):
        return self

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        prompt = messages[0].content
        if "事实摘要员" in prompt:
            result = "订单1001要求换货，尚未解决。"
        elif "问题理解器" in prompt:
            # 从当前用户原话读问题，末尾背景消息不充当当前问题。
            current = next(
                m
                for m in reversed(messages)
                if m.type == "human" and not m.content.startswith("以下仅为")
            )
            result = json.dumps({"question": current.content}, ensure_ascii=False)
        elif prompt.startswith("将顾客"):
            result = '{"intent":"订单","confidence":0.99}'
        else:
            self.contexts.append(messages)
            result = "演示订单1001换货诉求尚未解决。" + "请等待人工核实。" * 90
        yield AIMessageChunk(content=result, response_metadata={"finish_reason": "stop"})


def test_turn_finishes_while_background_summary_is_still_waiting():
    class HeldSummaryModel(ConversationModel):
        async def astream(self, messages):
            if "事实摘要员" in messages[0].content:
                started.set()
                await release.wait()
            async for chunk in super().astream(messages):
                yield chunk

    async def scenario():
        nonlocal started, release
        started, release = asyncio.Event(), asyncio.Event()
        settings = config(model_context_window=18000, max_output_tokens=2000, max_agent_steps=3)
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        repo, cid = ChatRepository(db), str(uuid4())
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(HeldSummaryModel(), repo, settings, saver)
            try:
                for _ in range(22):
                    rid = str(uuid4())
                    await repo.begin_request(cid, rid, "订单1001要求换货")

                    async def reply(rid=rid):
                        return [e async for e in workflow.stream(cid, rid, "订单1001要求换货")]

                    events = await asyncio.wait_for(reply(), 2)
                    await workflow.commit(cid, rid)
                    await repo.set_request_state(rid, "completed")
                    if started.is_set():
                        assert any(e["event"] == "actions" for e in events)
                        assert not release.is_set()
                        assert not workflow.summaries.tasks[cid].done()
                        assert (await repo.read_conversation(cid))["summary_upto_msg_id"] is None
                        break
                else:
                    raise AssertionError("预期触发后台摘要")
            finally:
                await workflow.summaries.close()
        await db.close()

    started = release = None
    asyncio.run(scenario())


@pytest.mark.parametrize("demo", [False, True])
def test_22_turns_preserve_full_checkpoint_and_only_demo_compresses(demo, caplog):
    caplog.set_level(logging.INFO, logger="support_agent")

    async def scenario():
        settings = config(
            **(
                {"model_context_window": 18000, "max_output_tokens": 2000, "max_agent_steps": 3}
                if demo
                else {}
            )
        )
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        repo, model = ChatRepository(db), ConversationModel()
        cid = str(uuid4())
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            workflow = SupportWorkflow(model, repo, settings, saver)
            for index in range(22):
                rid = str(uuid4())
                question = (
                    "订单1001要求换货，手机号13800138000。"
                    if index == 0
                    else f"请查询订单1001，补充说明第{index}轮。"
                )
                await repo.begin_request(cid, rid, question)
                events = [e async for e in workflow.stream(cid, rid, question)]
                assert not any(e.get("text", "").startswith("本轮查询已达到") for e in events)
                await workflow.commit(cid, rid)
                await repo.set_request_state(rid, "completed")
                await workflow.summaries.drain()
            state = await workflow.graph.aget_state({"configurable": {"thread_id": cid}})
            assert len(state.values["messages"]) == 44
            assert state.values["messages"][0].content.startswith("订单1001要求换货")
            audit = await repo.read_conversation(cid)
            assert bool(audit["summaries"]) == demo
            assert ("层1 降级" in caplog.text) == demo
            assert ("summary trigger" in caplog.text) == demo
            assert sum("history_ctx" in r.message for r in caplog.records) == 22
            assert sum("model_ctx" in r.message for r in caplog.records) == 22
            assert all(c[0].content == model.contexts[0][0].content for c in model.contexts)
            if demo:
                assert "订单1001" in model.contexts[-1][-1].content
            await workflow.summaries.close()
        await db.close()

    asyncio.run(scenario())
