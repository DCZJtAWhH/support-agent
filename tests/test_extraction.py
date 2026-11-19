import asyncio
import json

import httpx
import pytest
from conftest import sse_events, stream_response


@pytest.mark.parametrize("method", ["json_schema", "json_mode"])
def test_extracts_fixed_fields_without_tool_calls(harness, method):
    expected = {"order_id": "A123", "request_type": "换货", "expected_solution": "换一个新的"}

    async def scenario():
        async def handler(request, payload):
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-extract",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(expected, ensure_ascii=False),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        async with harness(handler, structured_output_method=method) as (client, calls, _):
            response = await client.post(
                "/after-sales/extract",
                json={"description": "订单 A123 的杯子碎了，我想换一个新的。"},
            )
            assert response.status_code == 200
            assert response.json() == expected
            assert calls[0]["response_format"]["type"] == (
                "json_schema" if method == "json_schema" else "json_object"
            )
            assert "tools" not in calls[0]
            assert calls[0]["stream"] is False
            assert calls[0]["messages"][-1]["content"] == "订单 A123 的杯子碎了，我想换一个新的。"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "content",
    [
        "not JSON",
        '{"order_id": "A123"}',
        '{"order_id": null, "request_type": "瞎编分类", "expected_solution": null}',
        '{"order_id": 123, "request_type": "退款", "expected_solution": null}',
        '{"order_id": null, "request_type": "未知", "expected_solution": null, "extra": true}',
    ],
)
def test_invalid_structured_content_returns_clear_error(harness, content):
    async def scenario():
        async def handler(request, payload):
            return httpx.Response(
                200,
                json={
                    "id": "bad-result",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        async with harness(handler) as (client, _, _):
            response = await client.post("/after-sales/extract", json={"description": "商品有问题"})
            assert response.status_code == 502
            assert response.json()["detail"]["code"] == "invalid_model_response"
            assert "not JSON" not in response.text

    asyncio.run(scenario())


def test_unsupported_structured_method_never_falls_back(harness):
    async def scenario():
        async def handler(request, payload):
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "response_format unsupported",
                        "type": "invalid_request_error",
                    }
                },
            )

        async with harness(handler) as (client, calls, _):
            response = await client.post("/after-sales/extract", json={"description": "我想退款"})
            assert response.status_code == 502
            assert response.json()["detail"]["code"] == "structured_output_rejected"
            assert response.json()["detail"]["upstream_status"] == 400
            assert "STRUCTURED_OUTPUT_METHOD=json_mode" in response.json()["detail"]["message"]
            assert len(calls) == 1
            assert calls[0]["response_format"]["type"] == "json_schema"
            assert "tools" not in calls[0]

    asyncio.run(scenario())


def test_langchain_legacy_model_downgrade_is_blocked_before_any_tool_request(harness):
    async def scenario():
        async with harness(openai_model="gpt-3.5-turbo") as (client, calls, _):
            response = await client.post("/after-sales/extract", json={"description": "我想退款"})
            assert response.status_code == 502
            assert response.json()["detail"]["code"] == "invalid_model_response"
            assert calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["json_schema", "json_mode"])
def test_chat_can_work_when_deepseek_chat_endpoint_rejects_json_schema(harness, method):
    expected = {"order_id": None, "request_type": "退款", "expected_solution": "退款"}

    async def scenario():
        async def handler(request, payload):
            if payload.get("stream"):
                return stream_response()
            if payload["response_format"]["type"] == "json_schema":
                return httpx.Response(
                    400,
                    json={"error": {"message": "Invalid response_format type: json_schema"}},
                )
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-json-mode",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": json.dumps(expected)},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        async with harness(handler, structured_output_method=method) as (client, calls, _):
            chat = await client.post("/chat", json={"message": "你好"})
            assert sse_events(chat)[-1][0] == "done"
            extraction = await client.post("/after-sales/extract", json={"description": "我想退款"})
            if method == "json_schema":
                assert extraction.status_code == 502
                assert extraction.json()["detail"]["code"] == "structured_output_rejected"
            else:
                assert extraction.status_code == 200
                assert extraction.json() == expected
            assert len(calls) == 2
            assert all("tools" not in call for call in calls)

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [401, 429, 500])
def test_other_upstream_errors_keep_status_without_leaking_error_body(harness, status):
    async def scenario():
        async def handler(request, payload):
            return httpx.Response(
                status,
                json={
                    "error": {
                        "message": "response_format json_schema secret-value https://private.invalid",
                        "param": "response_format",
                    }
                },
            )

        async with harness(handler) as (client, calls, _):
            response = await client.post("/after-sales/extract", json={"description": "我想退款"})
            assert response.status_code == 502
            assert response.json()["detail"]["code"] == "upstream_error"
            assert response.json()["detail"]["upstream_status"] == status
            assert response.json()["detail"]["error_type"]
            assert "secret-value" not in response.text
            assert "private.invalid" not in response.text
            assert len(calls) == 1

    asyncio.run(scenario())
