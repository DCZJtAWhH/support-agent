import asyncio
from uuid import uuid4

from conftest import sse_events, stream_response


def test_only_explicit_ticket_confirmation_creates_ticket_and_retries_are_idempotent(harness):
    async def handler(request, payload):
        return stream_response(('{"intent":"投诉"}',))

    async def scenario():
        async with harness(handler) as (client, calls, _):
            response = await client.post("/chat", json={"message": "我要投诉"})
            events = sse_events(response)
            meta = events[0][1]
            assert next(d["actions"] for e, d in events if e == "actions") == [
                "handoff",
                "create_ticket",
            ]
            cid = meta["conversation_id"]
            before = (await client.get("/conversations/" + cid)).json()
            assert not before["tickets"]
            body = {"conversation_id": cid, "request_id": meta["request_id"]}
            first = await client.post("/actions/ticket", json=body)
            assert first.status_code == 200
            second = await client.post("/actions/ticket", json=body)
            assert first.json()["ticket_no"] == second.json()["ticket_no"]
            after = (await client.get("/conversations/" + cid)).json()
            assert len(after["tickets"]) == 1 and after["status"] == "active"
            assert len(calls) == 1
            forged = await client.post("/actions/ticket", json={**body, "request_id": str(uuid4())})
            assert forged.status_code == 409

    asyncio.run(scenario())


def test_ticket_confirmation_requires_a_suggestion_and_the_original_conversation(harness):
    async def scenario():
        async with harness() as (client, _, _):
            first = sse_events(await client.post("/chat", json={"message": "你好"}))[0][1]
            second = sse_events(await client.post("/chat", json={"message": "谢谢"}))[0][1]
            denied = await client.post("/actions/ticket", json=first)
            assert denied.status_code == 409
            forged = await client.post(
                "/actions/ticket",
                json={
                    "conversation_id": second["conversation_id"],
                    "request_id": first["request_id"],
                },
            )
            assert forged.status_code == 409
            audit = (await client.get("/conversations/" + first["conversation_id"])).json()
            assert audit["tickets"] == []

    asyncio.run(scenario())
