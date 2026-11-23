import asyncio

import httpx
import pytest
from conftest import sse_events, stream_response


def test_smalltalk_returns_fixed_reply_without_any_model_call(harness):
    async def scenario():
        async with harness() as (client, calls, _):
            response = await client.post("/chat", json={"message": "你好"})
            events = sse_events(response)
            assert events[-1][0] == "done"
            assert "本店智能客服" in response.text
            assert calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("parameter", ["max_completion_tokens", "max_tokens"])
def test_output_limit_uses_the_configured_openai_protocol_parameter(harness, parameter):
    async def scenario():
        async with harness(output_token_parameter=parameter, max_output_tokens=321) as (
            client,
            calls,
            _,
        ):
            response = await client.post("/chat", json={"message": "查订单"})
            assert sse_events(response)[-1][0] == "done"
            for call in calls:
                assert call[parameter] == 321
                other = (
                    "max_tokens"
                    if parameter == "max_completion_tokens"
                    else "max_completion_tokens"
                )
                assert other not in call

    asyncio.run(scenario())


def test_second_turn_uses_completed_state_and_other_conversations_are_isolated(harness):
    async def scenario():
        async with harness() as (client, calls, _):
            first = await client.post("/chat", json={"message": "我的订单号是 A123"})
            cid = sse_events(first)[0][1]["conversation_id"]
            second = await client.post(
                "/chat", json={"message": "刚才的订单号是什么？", "conversation_id": cid}
            )
            assert sse_events(second)[-1][0] == "done"
            agents = [c for c in calls if "tools" in c]
            assert agents[1]["messages"][1:] == [
                {"role": "user", "content": "我的订单号是 A123"},
                {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
                {"role": "user", "content": "刚才的订单号是什么？"},
            ]
            await client.post("/chat", json={"message": "新顾客"})
            assert calls[-1]["messages"][1:] == [{"role": "user", "content": "新顾客"}]

    asyncio.run(scenario())


def test_oversized_input_is_rejected_before_classification(harness):
    async def scenario():
        async with harness(
            context_window_tokens=3250, max_output_tokens=100, token_safety_margin=100
        ) as (client, calls, _):
            response = await client.post("/chat", json={"message": "中" * 5000})
            assert response.status_code == 413 and calls == []

    asyncio.run(scenario())


def test_existing_completed_conversation_is_imported_when_it_has_no_graph_checkpoint(harness):
    from uuid import uuid4

    async def scenario():
        async with harness() as (client, calls, app):
            cid, rid = str(uuid4()), str(uuid4())
            repo = app.state.repository
            await repo.begin_request(cid, rid, "我的订单号是1001")
            await repo.save_message(cid, rid, 4, "assistant", "已记下您的订单号。")
            await repo.set_request_state(rid, "completed")
            response = await client.post(
                "/chat", json={"message": "刚才的订单号是什么", "conversation_id": cid}
            )
            assert sse_events(response)[-1][0] == "done"
            assert calls[-1]["messages"][1:] == [
                {"role": "user", "content": "我的订单号是1001"},
                {"role": "assistant", "content": "已记下您的订单号。"},
                {"role": "user", "content": "刚才的订单号是什么"},
            ]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["length", "no_finish", "rate_limit", "empty", "tool_calls"])
def test_failed_generation_is_reported_and_never_enters_state_history(harness, failure):
    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"订单"}',))
        if payload["messages"][-1]["content"] == "失败轮次":
            if failure == "rate_limit":
                return httpx.Response(429, json={"error": {"message": "private-upstream-detail"}})
            reason = {"no_finish": None, "empty": "stop"}.get(failure, failure)
            return stream_response(() if failure == "empty" else ("半截回答",), reason)
        return stream_response()

    async def scenario():
        async with harness(handler) as (client, calls, _):
            first = await client.post("/chat", json={"message": "成功轮次"})
            cid = sse_events(first)[0][1]["conversation_id"]
            failed = await client.post(
                "/chat", json={"message": "失败轮次", "conversation_id": cid}
            )
            assert sse_events(failed)[-1][0] == "error"
            assert "private-upstream-detail" not in failed.text
            await client.post("/chat", json={"message": "重试", "conversation_id": cid})
            assert calls[-1]["messages"][1:] == [
                {"role": "user", "content": "成功轮次"},
                {"role": "assistant", "content": "您好，请问有什么可以帮您？"},
                {"role": "user", "content": "重试"},
            ]

    asyncio.run(scenario())
