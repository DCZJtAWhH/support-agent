import asyncio
import json

import httpx
import pytest
from conftest import sse_events, stream_response, tool_response


def protocol_response(*, reasoning, call=False):
    response = (
        tool_response([{"id": "model-call", "name": "query_order", "args": {"order_id": "1001"}}])
        if call
        else stream_response(("演示订单已核对。",))
    )
    text = response.text
    prefix, tail = text.split("\n\n", 1)
    row = json.loads(prefix.removeprefix("data: "))
    row["choices"][0]["delta"]["reasoning_content"] = reasoning
    return httpx.Response(
        200,
        text="data: " + json.dumps(row) + "\n\n" + tail,
        headers={"content-type": "text/event-stream"},
    )


def test_deepseek_tool_roundtrip_preserves_protocol_and_checkpoint_without_exposing_it(
    harness, caplog
):
    opaque = "protocol-test-marker"

    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"订单","confidence":0.99}',))
        assistants = [m for m in payload["messages"] if m["role"] == "assistant"]
        if payload["messages"][-1]["role"] == "tool":
            assert assistants[-1].get("reasoning_content") == opaque
            return protocol_response(reasoning=opaque + "-final")
        return protocol_response(reasoning=opaque, call=True)

    async def scenario():
        async with harness(handler, model_provider="deepseek") as (client, _, app):
            response = await client.post("/chat", json={"message": "查询订单1001"})
            events = sse_events(response)
            assert events[-1][0] == "done", response.text
            cid = events[0][1]["conversation_id"]
            assert opaque not in response.text
            assert opaque not in caplog.text
            checkpoint = await app.state.chat.workflow.graph.aget_state(
                {"configurable": {"thread_id": cid}}
            )
            ai = [m for m in checkpoint.values["history"] if m.type == "ai"]
            assert ai[0].additional_kwargs["reasoning_content"] == opaque
            assert ai[-1].additional_kwargs["reasoning_content"] == opaque + "-final"
            repeated = await client.post(
                "/chat", json={"conversation_id": cid, "message": "再查订单1001"}
            )
            assert sse_events(repeated)[-1][0] == "done", repeated.text

    asyncio.run(scenario())


@pytest.mark.parametrize("provider", ["auto", "openai"])
def test_generic_openai_compatible_tool_roundtrip_has_no_vendor_fields(harness, provider):
    async def handler(request, payload):
        assert payload["model"] == "test-model"
        assert "thinking" not in payload
        assert all("reasoning_content" not in message for message in payload["messages"])
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"订单","confidence":0.99}',))
        if payload["messages"][-1]["role"] == "tool":
            return stream_response(("已查到演示订单。",))
        return tool_response(
            [{"id": "generic", "name": "query_order", "args": {"order_id": "1001"}}]
        )

    async def scenario():
        async with harness(handler, model_provider=provider) as (client, _, _):
            response = await client.post("/chat", json={"message": "查询订单1001"})
            assert sse_events(response)[-1][0] == "done", response.text
            assert "已查到" in response.text

    asyncio.run(scenario())
