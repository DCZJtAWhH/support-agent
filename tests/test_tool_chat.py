import asyncio
import json
from uuid import uuid4

from conftest import mock_logistics, sse_events, stream_response, tool_response


def test_simple_logistics_converges_after_one_tool_and_audits_observation(harness, caplog):
    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"物流","confidence":0.99}',))
        if payload["messages"][-1]["role"] != "tool":
            return tool_response(
                [{"id": "logistics", "name": "query_logistics", "args": {"order_id": "1001"}}]
            )
        result = json.loads(payload["messages"][-1]["content"])
        return stream_response(("演示物流：", result["data"]["location"]))

    async def scenario():
        async with harness(handler) as (client, calls, app):
            app.state.chat.workflow.catalog.entries.register(mock_logistics, source="mcp:logistics")
            response = await client.post("/chat", json={"message": "订单1001的物流到哪了"})
            events = sse_events(response)
            assert events[-1][0] == "done"
            assert [(d["name"], d["state"]) for e, d in events if e == "tool_status"] == [
                ("query_logistics", "running"),
                ("query_logistics", "succeeded"),
            ]
            assert len([c for c in calls if "tools" in c]) == 2
            logged = [
                json.loads(r.message.split("model_ctx ", 1)[1])
                for r in caplog.records
                if "model_ctx " in r.message
            ]
            assert logged[-1]["messages"] == calls[-1]["messages"]
            audit = (await client.get("/conversations/" + events[0][1]["conversation_id"])).json()
            assert not [m for m in audit["messages"] if m["role"] == "tool"]
            snapshot = await app.state.chat.workflow.graph.aget_state(
                {"configurable": {"thread_id": audit["id"]}}
            )
            observations = [m for m in snapshot.values["messages"] if m.type == "tool"]
            assert observations[0].tool_call_id == "logistics"
            assert audit["messages"][0]["state"] == "completed"

    asyncio.run(scenario())


def test_agent_cannot_execute_ticket_even_if_model_requests_it(harness):
    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"订单","confidence":0.99}',))
        assert "create_ticket" in {t["function"]["name"] for t in payload["tools"]}
        if payload["messages"][-1]["role"] != "tool":
            return tool_response(
                [
                    {
                        "id": "forged",
                        "name": "create_ticket",
                        "args": {"description": "损坏", "ticket_type": "维修"},
                    }
                ]
            )
        result = json.loads(payload["messages"][-1]["content"])
        assert result["error"]["code"] == "permission_denied"
        return stream_response(("您可以选择建工单，由人工跟进。",))

    async def scenario():
        async with harness(handler) as (client, _, _):
            response = await client.post("/chat", json={"message": "查询订单1001"})
            events = sse_events(response)
            assert events[-1][0] == "done"
            audit = (await client.get("/conversations/" + events[0][1]["conversation_id"])).json()
            assert audit["tickets"] == []
            assert next(d["actions"] for e, d in events if e == "actions") == ["create_ticket"]

    asyncio.run(scenario())


def test_completed_request_replays_without_model_calls_and_conflicting_reuse_is_rejected(harness):
    async def scenario():
        async with harness() as (client, calls, _):
            body = {"message": "查订单1001", "request_id": str(uuid4())}
            first = await client.post("/chat", json=body)
            count = len(calls)
            replay = await client.post("/chat", json=body)
            assert sse_events(replay)[-1][0] == "done" and len(calls) == count
            assert sse_events(first)[0][1] == sse_events(replay)[0][1]
            conflict = await client.post("/chat", json={**body, "message": "另一个问题"})
            assert conflict.status_code == 409

    asyncio.run(scenario())


def test_runaway_agent_has_finite_model_steps_and_offers_followup(harness):
    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"订单","confidence":0.99}',))
        return tool_response([{"id": "again", "name": "query_order", "args": {"order_id": "1001"}}])

    async def scenario():
        async with harness(handler, agent_max_steps=2) as (client, calls, _):
            response = await client.post("/chat", json={"message": "循环查订单1001"})
            assert sse_events(response)[-1][0] == "done"
            assert "预算上限" in response.text
            assert len([c for c in calls if "tools" in c]) == 2

    asyncio.run(scenario())


def test_failed_model_step_resumes_from_native_checkpoint_without_repeating_order_query(harness):
    import httpx

    attempts = 0

    async def handler(request, payload):
        nonlocal attempts
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"订单","confidence":0.99}',))
        if payload["messages"][-1]["role"] != "tool":
            return tool_response(
                [
                    {
                        "id": "order",
                        "name": "query_order",
                        "args": {"order_id": "1001"},
                    }
                ]
            )
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": {"message": "private"}})
        return stream_response(("已查询演示订单。",))

    async def scenario():
        async with harness(handler) as (client, calls, _):
            body = {"message": "查订单1001", "request_id": str(uuid4())}
            assert sse_events(await client.post("/chat", json=body))[-1][0] == "error"
            replay = await client.post("/chat", json=body)
            assert sse_events(replay)[-1][0] == "done"
            # 理解、分类、首个决策、失败回答、恢复回答；不能重跑理解、分类或查单决策。
            assert len(calls) == 5

    asyncio.run(scenario())
