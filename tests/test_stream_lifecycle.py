import asyncio

import httpx
import pytest
from conftest import HeldStream, sse_events, start_asgi_stream, stream_response


def test_same_conversation_rejects_concurrent_request_while_tokens_are_streaming(harness):
    async def scenario():
        held = HeldStream()

        async def handler(request, payload):
            if payload["messages"][0]["content"].startswith("将顾客"):
                return stream_response(('{"intent":"订单"}',))
            if payload["messages"][-1]["content"] == "慢请求":
                return httpx.Response(
                    200, stream=held, headers={"content-type": "text/event-stream"}
                )
            return stream_response()

        async with harness(handler) as (client, _, app):
            first = await client.post("/chat", json={"message": "已有轮次"})
            conversation_id = sse_events(first)[0][1]["conversation_id"]
            task, _, frames = await start_asgi_stream(
                app, {"message": "慢请求", "conversation_id": conversation_id}
            )
            try:
                assert not task.done(), "上游尚未结束时客户端应当已经收到文本增量"
                assert any(b"event: token" in frame.get("body", b"") for frame in frames)
                concurrent = await client.post(
                    "/chat", json={"message": "并发请求", "conversation_id": conversation_id}
                )
                assert concurrent.status_code == 409
            finally:
                held.release.set()
                await asyncio.wait_for(task, 5)

    asyncio.run(scenario())


def test_session_stays_busy_until_final_response_body_is_sent(harness):
    async def scenario():
        async with harness() as (client, _, app):
            first = await client.post("/chat", json={"message": "已有轮次"})
            conversation_id = sse_events(first)[0][1]["conversation_id"]
            reached, release = asyncio.Event(), asyncio.Event()
            task, _, _ = await start_asgi_stream(
                app,
                {"message": "末帧尚未送达", "conversation_id": conversation_id},
                final_gate=(reached, release),
            )
            try:
                await asyncio.wait_for(reached.wait(), 2)
                concurrent = await client.post(
                    "/chat", json={"message": "并发", "conversation_id": conversation_id}
                )
                assert concurrent.status_code == 409
            finally:
                release.set()
                await asyncio.wait_for(task, 2)

    asyncio.run(scenario())


@pytest.mark.parametrize("spec_version", ["2.0", "2.4"])
def test_disconnect_cancels_upstream_unlocks_session_and_discards_partial_turn(
    harness, spec_version
):
    async def scenario():
        held = HeldStream()

        async def handler(request, payload):
            if payload["messages"][0]["content"].startswith("将顾客"):
                return stream_response(('{"intent":"订单"}',))
            if payload["messages"][-1]["content"] == "将要断流":
                return httpx.Response(
                    200, stream=held, headers={"content-type": "text/event-stream"}
                )
            return stream_response()

        async with harness(handler) as (client, calls, app):
            first = await client.post("/chat", json={"message": "成功轮次"})
            conversation_id = sse_events(first)[0][1]["conversation_id"]
            task, received, _ = await start_asgi_stream(
                app, {"message": "将要断流", "conversation_id": conversation_id}, spec_version
            )
            try:
                await received.put({"type": "http.disconnect"})
                await asyncio.wait_for(asyncio.shield(task), 2)
                await asyncio.wait_for(held.closed.wait(), 2)
                retry = await client.post(
                    "/chat", json={"message": "断流后重试", "conversation_id": conversation_id}
                )
                assert retry.status_code == 200
                assert calls[-1]["messages"][1:] == [
                    {"role": "user", "content": "成功轮次"},
                    {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
                    {"role": "user", "content": "断流后重试"},
                ]
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
