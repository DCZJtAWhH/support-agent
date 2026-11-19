import asyncio

import httpx
import pytest
from conftest import sse_events, stream_response


def test_chat_streams_upstream_text_as_sse_without_tools(harness):
    async def scenario():
        async with harness() as (client, calls, _):
            response = await client.post("/chat", json={"message": "你好"})
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = sse_events(response)
            assert [name for name, _ in events] == ["meta", "token", "token", "done"]
            assert [data["text"] for name, data in events if name == "token"] == [
                "您好",
                "，请问有什么可以帮您？",
            ]
            assert events[0][1]["conversation_id"] == events[-1][1]["conversation_id"]
            assert calls[0]["model"] == "test-model"
            assert calls[0]["stream"] is True
            assert "tools" not in calls[0]
            assert calls[0]["messages"][-1] == {"role": "user", "content": "你好"}

    asyncio.run(scenario())


@pytest.mark.parametrize("parameter", ["max_completion_tokens", "max_tokens"])
def test_output_limit_uses_the_configured_openai_protocol_parameter(harness, parameter):
    async def scenario():
        async with harness(output_token_parameter=parameter, max_output_tokens=321) as (
            client,
            calls,
            _,
        ):
            await client.post("/chat", json={"message": "你好"})
            assert calls[0][parameter] == 321
            other = (
                "max_tokens" if parameter == "max_completion_tokens" else "max_completion_tokens"
            )
            assert other not in calls[0]

    asyncio.run(scenario())


def test_budget_drops_oldest_complete_turn_and_rejects_oversized_current_input(harness):
    async def scenario():
        async with harness(
            context_window_tokens=1500, max_output_tokens=100, token_safety_margin=100
        ) as (client, calls, _):
            old_message = "旧轮次" + "a" * 700
            recent_message = "新轮次" + "b" * 400
            first = await client.post("/chat", json={"message": old_message})
            assert first.status_code == 200
            conversation_id = sse_events(first)[0][1]["conversation_id"]
            await client.post(
                "/chat", json={"message": recent_message, "conversation_id": conversation_id}
            )
            assert calls[1]["messages"][1:] == [{"role": "user", "content": recent_message}]
            await client.post("/chat", json={"message": "继续", "conversation_id": conversation_id})
            assert calls[2]["messages"][1:] == [
                {"role": "user", "content": recent_message},
                {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
                {"role": "user", "content": "继续"},
            ]
            rejected = await client.post(
                "/chat", json={"message": "中" * 2000, "conversation_id": conversation_id}
            )
            assert rejected.status_code == 413
            assert len(calls) == 3

    asyncio.run(scenario())


def test_second_turn_uses_completed_history_and_new_conversations_are_isolated(harness):
    async def scenario():
        async with harness() as (client, calls, _):
            first = await client.post("/chat", json={"message": "我的订单号是 A123"})
            conversation_id = sse_events(first)[0][1]["conversation_id"]
            second = await client.post(
                "/chat",
                json={"message": "刚才的订单号是什么？", "conversation_id": conversation_id},
            )
            assert sse_events(second)[0][1]["conversation_id"] == conversation_id
            assert calls[1]["messages"][1:] == [
                {"role": "user", "content": "我的订单号是 A123"},
                {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
                {"role": "user", "content": "刚才的订单号是什么？"},
            ]
            await client.post("/chat", json={"message": "新顾客"})
            assert calls[2]["messages"][1:] == [{"role": "user", "content": "新顾客"}]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["length", "no_finish", "rate_limit", "empty", "tool_calls"])
def test_failed_turn_is_reported_and_never_enters_history(harness, failure):
    async def scenario():
        count = 0

        async def handler(request, payload):
            nonlocal count
            count += 1
            if count != 2:
                return stream_response()
            if failure == "rate_limit":
                return httpx.Response(
                    429,
                    json={"error": {"message": "private-upstream-detail", "type": "rate_limit"}},
                )
            reason = {"no_finish": None, "empty": "stop"}.get(failure, failure)
            return stream_response(() if failure == "empty" else ("半截回答",), reason)

        async with harness(handler) as (client, calls, _):
            first = await client.post("/chat", json={"message": "成功轮次"})
            conversation_id = sse_events(first)[0][1]["conversation_id"]
            failed = await client.post(
                "/chat", json={"message": "失败轮次", "conversation_id": conversation_id}
            )
            names = [name for name, _ in sse_events(failed)]
            assert names[-1] == "error"
            assert "done" not in names
            assert "private-upstream-detail" not in failed.text
            await client.post("/chat", json={"message": "重试", "conversation_id": conversation_id})
            assert calls[2]["messages"][1:] == [
                {"role": "user", "content": "成功轮次"},
                {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
                {"role": "user", "content": "重试"},
            ]

    asyncio.run(scenario())
