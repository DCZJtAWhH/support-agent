import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from conftest import sse_events, stream_response, tool_response


def test_chat_selects_two_tools_reports_status_and_streams_grounded_final_answer(harness):
    async def handler(request, payload):
        if "tools" in payload:
            return tool_response(
                [
                    {"id": "call_order", "name": "query_order", "args": {"order_id": "1001"}},
                    {
                        "id": "call_logistics",
                        "name": "query_logistics",
                        "args": {"order_id": "1001"},
                    },
                ]
            )
        assert [m["role"] for m in payload["messages"]][-4:] == [
            "user",
            "assistant",
            "tool",
            "tool",
        ]
        result = json.loads(payload["messages"][-1]["content"])
        assert result["ok"] and result["data"]["order_id"] == "1001"
        return stream_response(("演示物流：", result["data"]["location"]))

    async def scenario():
        async with harness(handler) as (client, calls, _):
            response = await client.post("/chat", json={"message": "订单1001的订单情况和物流"})
            events = sse_events(response)
            assert events[-1][0] == "done"
            statuses = [d for e, d in events if e == "tool_status"]
            assert [(d["name"], d["state"]) for d in statuses] == [
                ("query_order", "running"),
                ("query_order", "succeeded"),
                ("query_logistics", "running"),
                ("query_logistics", "succeeded"),
            ]
            assert len(calls) == 2 and "tools" not in calls[1]
            audit = (await client.get("/conversations/" + events[0][1]["conversation_id"])).json()
            assert [m["role"] for m in audit["messages"]] == [
                "user",
                "assistant",
                "tool",
                "tool",
                "assistant",
            ]
            assert audit["messages"][1]["tool_calls"][1]["id"] == "call_logistics"
            assert audit["messages"][3]["tool_call_id"] == "call_logistics"
            assert audit["messages"][0]["state"] == "completed"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["third_tool", "second_round", "invalid_arguments", "unknown_tool"]
)
def test_tool_boundaries_never_execute_an_extra_round_and_return_safe_errors(harness, failure):
    async def scenario():
        async def handler(request, payload):
            if "tools" in payload:
                call = {"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}}
                if failure == "third_tool":
                    return tool_response([{**call, "id": f"call_{i}"} for i in range(3)])
                if failure == "invalid_arguments":
                    call["args"] = {"order_id": 1001}
                if failure == "unknown_tool":
                    call["name"] = "delete_database"
                return tool_response([call])
            if failure == "second_round":
                return tool_response(
                    [{"id": "call_extra", "name": "query_order", "args": {"order_id": "1002"}}]
                )
            result = json.loads(payload["messages"][-1]["content"])
            assert not result["ok"]
            return stream_response(("查询失败，请补充信息或稍后重试。",))

        async with harness(handler) as (client, calls, _):
            response = await client.post("/chat", json={"message": "查订单1001"})
            events = sse_events(response)
            assert len(calls) <= 2
            if failure in {"third_tool", "second_round"}:
                assert events[-1][0] == "error"
            else:
                assert events[-1][0] == "done"
                assert any(e == "tool_status" and d["state"] == "failed" for e, d in events)
            if len(calls) == 2:
                assert "tools" not in calls[1]

    asyncio.run(scenario())


def test_context_trimming_drops_entire_tool_turn_without_orphan_results(harness):
    async def scenario():
        async def handler(request, payload):
            if "tools" in payload and payload["messages"][-1]["content"] == "查订单1001":
                return tool_response(
                    [{"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}}]
                )
            return stream_response(("您好，请问有什么可以帮您？",))

        async with harness(
            handler, context_window_tokens=3250, max_output_tokens=100, token_safety_margin=100
        ) as (client, calls, _):
            first = await client.post("/chat", json={"message": "查订单1001"})
            cid = sse_events(first)[0][1]["conversation_id"]
            second = await client.post(
                "/chat", json={"message": "新问题" + "a" * 700, "conversation_id": cid}
            )
            assert sse_events(second)[-1][0] == "done"
            assert [m["role"] for m in calls[-1]["messages"]] == ["system", "user"]

    asyncio.run(scenario())


def test_failed_final_answer_keeps_ticket_and_retry_reuses_tools_even_without_meta(harness):
    async def scenario():
        final_attempts = 0

        async def handler(request, payload):
            nonlocal final_attempts
            if "tools" in payload:
                return tool_response(
                    [
                        {
                            "id": "call_ticket",
                            "name": "create_ticket",
                            "args": {"description": "商品损坏", "ticket_type": "维修"},
                        }
                    ]
                )
            final_attempts += 1
            if final_attempts == 1:
                return httpx.Response(503, json={"error": {"message": "private"}})
            result = json.loads(payload["messages"][-1]["content"])
            return stream_response(("已登记工单：", result["data"]["ticket_no"]))

        async with harness(handler) as (client, calls, _):
            body = {"message": "请人工处理商品损坏", "request_id": str(uuid4())}
            first = await client.post("/chat", json=body)
            events = sse_events(first)
            assert events[-1][0] == "error"
            cid = events[0][1]["conversation_id"]
            failed = (await client.get("/conversations/" + cid)).json()
            assert len(failed["tickets"]) == 1
            assert failed["messages"][0]["state"] == "failed"
            second = await client.post("/chat", json=body)
            assert sse_events(second)[-1][0] == "done"
            audit = (await client.get("/conversations/" + cid)).json()
            assert len(audit["tickets"]) == 1
            assert len(audit["messages"]) == 4
            assert audit["messages"][0]["state"] == "completed"
            assert len([c for c in calls if "tools" in c]) == 1
            assert failed["tickets"][0]["ticket_no"] in second.text
            replay = await client.post("/chat", json=body)
            assert sse_events(replay)[-1][0] == "done" and len(calls) == 3
            conflict = await client.post("/chat", json={**body, "message": "另一个问题"})
            assert conflict.status_code == 409

    asyncio.run(scenario())
