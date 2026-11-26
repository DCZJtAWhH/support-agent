import asyncio
from uuid import uuid4

from module_support import TestServices as ChatRepository

from support_agent.platform.database import Database


def test_background_summary_does_not_block_and_failure_keeps_the_boundary(caplog):
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
    from test_session_context import config

    from support_agent.conversation.summaries import SummaryManager

    class HeldModel:
        def bind(self, **kwargs):
            return self

        async def astream(self, messages):
            started.set()
            await release.wait()
            if fail[0]:
                raise RuntimeError("暂时失败")
            yield AIMessageChunk(
                content="订单1001要求换货，尚待核实。", response_metadata={"finish_reason": "stop"}
            )

    async def scenario():
        nonlocal started, release
        started, release = asyncio.Event(), asyncio.Event()
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        repo = ChatRepository(db)
        cid, rid = str(uuid4()), str(uuid4())
        first = await repo.begin_request(cid, rid, "订单1001要求换货")
        last = await repo.save_message(cid, rid, 1000000, "assistant", "尚待核实")
        await repo.move_layer1(cid, last + 1)
        batch = [
            HumanMessage("订单1001要求换货", additional_kwargs={"turn_id": first, "msg_id": first}),
            AIMessage("尚待核实", additional_kwargs={"turn_id": first, "msg_id": last}),
        ]
        manager = SummaryManager(HeldModel(), repo, config())
        assert manager.schedule(cid, batch, [], None)
        await asyncio.wait_for(started.wait(), 1)
        assert not manager.schedule(cid, batch, [], None)
        assert (await repo.read_conversation(cid))["summary_upto_msg_id"] is None
        release.set()
        await manager.drain()
        assert (await repo.read_conversation(cid))["summary_upto_msg_id"] is None
        fail[0] = False
        assert manager.schedule(cid, batch, [], None)
        await manager.drain()
        assert (await repo.read_conversation(cid))["summaries"][0]["seq"] == 1
        await manager.close()
        await db.close()

    started = release = None
    fail = [True]
    asyncio.run(scenario())


def test_summary_append_is_atomic_idempotent_and_keeps_original_messages():
    async def scenario():
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        repo = ChatRepository(db)
        cid, rid = str(uuid4()), str(uuid4())
        first = await repo.begin_request(cid, rid, "订单1001要求换货")
        last = await repo.save_message(cid, rid, 1000000, "assistant", "尚待核实")
        await repo.set_request_state(rid, "completed")
        await repo.move_layer1(cid, last + 1)
        summary = await repo.append_summary(
            cid, first, last, "订单1001要求换货，尚待核实。", expected_upto=None
        )
        assert summary["seq"] == 1
        duplicate = await repo.append_summary(cid, first, last, "重复", expected_upto=None)
        assert duplicate is None
        audit = await repo.read_conversation(cid)
        assert audit["summary_upto_msg_id"] == last
        assert audit["summaries"][0]["content"] == "订单1001要求换货，尚待核实。"
        assert [m["content"] for m in audit["messages"]] == ["订单1001要求换货", "尚待核实"]
        await repo.move_layer1(cid, first)
        assert (await repo.read_conversation(cid))["layer1_from_msg_id"] == last + 1
        await db.close()

    asyncio.run(scenario())
