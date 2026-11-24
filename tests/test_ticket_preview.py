import asyncio
import json

import pytest
from conftest import sse_events, stream_response, tool_response


@pytest.mark.parametrize("confirmed", [True, False])
def test_chat_ticket_clarifies_then_interrupts_and_requires_frontend_resume(harness, confirmed):
    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"售后","confidence":0.99}',))
        current = next(m["content"] for m in reversed(payload["messages"]) if m["role"] == "user")
        if payload["messages"][-1]["role"] == "tool":
            result = json.loads(payload["messages"][-1]["content"])
            return stream_response(
                (
                    "工单号：" + result["data"]["ticket_no"]
                    if result["ok"]
                    else "已取消，本次未创建工单。",
                )
            )
        if "不加热" not in current:
            return stream_response(("请描述商品遇到的具体问题，以及希望如何处理。",))
        return tool_response(
            [
                {
                    "id": "ticket-call",
                    "name": "create_ticket",
                    "args": {"description": "水壶不加热，希望维修", "ticket_type": "维修"},
                }
            ]
        )

    async def scenario():
        async with harness(handler) as (client, _, _):
            first = sse_events(await client.post("/chat", json={"message": "帮我建个工单"}))
            assert first[-1][0] == "done"
            assert not any(e == "order_required" or e == "ticket_required" for e, _ in first)
            assert "请描述" in "".join(d["text"] for e, d in first if e == "token")
            cid = first[0][1]["conversation_id"]
            question = "水壶不加热，希望维修"
            preview = sse_events(
                await client.post("/chat", json={"conversation_id": cid, "message": question})
            )
            card = next(d for e, d in preview if e == "ticket_required")
            assert card["ticket_type"] == "维修" and "不加热" in card["description"]
            assert card["tool_call_id"] == "ticket-call"
            rid = preview[0][1]["request_id"]
            before = (await client.get("/conversations/" + cid)).json()
            assert not before["tickets"]
            assert before["messages"][-2]["state"] == "waiting" or any(
                m["state"] == "waiting" for m in before["messages"] if m["request_id"] == rid
            )
            body = {
                "conversation_id": cid,
                "request_id": rid,
                "message": question,
                "ticket_confirmation": confirmed,
                "ticket_tool_call_id": card["tool_call_id"],
            }
            forged = await client.post("/chat", json={**body, "ticket_tool_call_id": "wrong"})
            assert forged.status_code == 409
            wrong_type = await client.post("/chat", json={**body, "ticket_confirmation": "true"})
            assert wrong_type.status_code == 422
            resumed = await client.post("/chat", json=body)
            assert resumed.status_code == 200
            assert sse_events(resumed)[-1][0] == "done", resumed.text
            after = (await client.get("/conversations/" + cid)).json()
            logs = [r for r in after["tool_audit_logs"] if r["tool_call_id"] == "ticket-call"]
            assert len(logs) == 1
            assert logs[0]["status"] == ("成功" if confirmed else "权限拒绝")
            if confirmed:
                assert len(after["tickets"]) == 1
                assert after["tickets"][0]["ticket_no"] in resumed.text
            else:
                assert not after["tickets"]
            replay = await client.post("/chat", json=body)
            assert sse_events(replay)[-1][0] == "done"
            assert len((await client.get("/conversations/" + cid)).json()["tickets"]) == int(
                confirmed
            )

    asyncio.run(scenario())


def test_validation_feedback_keeps_original_ticket_request_for_followup(harness):
    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"售后","confidence":0.99}',))
        if payload["messages"][-1]["role"] == "tool":
            return stream_response(("请补充具体问题。",))
        return tool_response(
            [
                {
                    "id": "description-check",
                    "name": "create_ticket",
                    "args": {"description": "水壶不加热", "ticket_type": "维修"},
                }
            ]
        )

    async def scenario():
        async with harness(handler) as (client, _, _):
            first = await client.post("/chat", json={"message": "帮我建工单"})
            events = sse_events(first)
            assert events[-1][0] == "done" and "请补充" in first.text
            second = await client.post(
                "/chat",
                json={
                    "conversation_id": events[0][1]["conversation_id"],
                    "message": "水壶不加热",
                },
            )
            assert "event: ticket_required" in second.text, second.text

    asyncio.run(scenario())


@pytest.mark.parametrize("message", ["水壶不加热，帮我建工单", "水壶不加热，帮我建个维修工单"])
def test_confirmed_ticket_survives_failed_answer_and_resume_does_not_create_again(harness, message):
    import httpx

    answer_attempts = 0

    async def handler(request, payload):
        nonlocal answer_attempts
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"售后","confidence":0.99}',))
        if payload["messages"][-1]["role"] == "tool":
            answer_attempts += 1
            if answer_attempts == 1:
                return httpx.Response(503, json={"error": {"message": "temporary"}})
            result = json.loads(payload["messages"][-1]["content"])
            return stream_response(("工单号：" + result["data"]["ticket_no"],))
        return tool_response(
            [
                {
                    "id": "persistent-ticket",
                    "name": "create_ticket",
                    "args": {"description": "水壶不加热", "ticket_type": "维修"},
                }
            ]
        )

    async def scenario():
        async with harness(handler) as (client, _, _):
            preview = sse_events(await client.post("/chat", json={"message": message}))
            body = {
                **preview[0][1],
                "message": message,
                "ticket_confirmation": True,
                "ticket_tool_call_id": "persistent-ticket",
            }
            failed = await client.post("/chat", json=body)
            assert sse_events(failed)[-1][0] == "error"
            retried = await client.post("/chat", json=body)
            assert retried.status_code == 200, retried.text
            assert sse_events(retried)[-1][0] == "done", retried.text
            audit = (await client.get("/conversations/" + body["conversation_id"])).json()
            assert len(audit["tickets"]) == 1
            assert len(audit["tool_audit_logs"]) == 1

    asyncio.run(scenario())


def test_model_cannot_forge_a_ticket_without_customer_request(harness):
    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"订单","confidence":0.99}',))
        if payload["messages"][-1]["role"] == "tool":
            result = json.loads(payload["messages"][-1]["content"])
            assert result["error"]["code"] == "permission_denied"
            return stream_response(("只能查询，本次未创建工单。",))
        return tool_response(
            [
                {
                    "id": "forged",
                    "name": "create_ticket",
                    "args": {"description": "虚构问题", "ticket_type": "其他"},
                }
            ]
        )

    async def scenario():
        async with harness(handler) as (client, _, _):
            response = await client.post("/chat", json={"message": "查询订单1001"})
            events = sse_events(response)
            assert events[-1][0] == "done", response.text
            assert not any(e == "ticket_required" for e, _ in events)
            audit = (await client.get("/conversations/" + events[0][1]["conversation_id"])).json()
            assert audit["tickets"] == []
            assert audit["tool_audit_logs"][0]["status"] == "权限拒绝"

    asyncio.run(scenario())


@pytest.mark.parametrize("description", ["水壶不加热", "帮我建个工单"])
def test_missing_description_cannot_be_invented_even_after_explicit_request(harness, description):
    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"售后","confidence":0.99}',))
        if payload["messages"][-1]["role"] == "tool":
            result = json.loads(payload["messages"][-1]["content"])
            assert result["error"]["code"] == "invalid_arguments"
            return stream_response(("请补充具体的问题描述。",))
        return tool_response(
            [
                {
                    "id": "invented",
                    "name": "create_ticket",
                    "args": {"description": description, "ticket_type": "维修"},
                }
            ]
        )

    async def scenario():
        async with harness(handler) as (client, _, _):
            response = await client.post("/chat", json={"message": "帮我建个工单"})
            assert sse_events(response)[-1][0] == "done", response.text
            assert "ticket_required" not in response.text
            assert "请补充" in response.text

    asyncio.run(scenario())
