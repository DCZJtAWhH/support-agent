import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from langchain_core.tools import tool


@tool("query_logistics")
async def mock_logistics(order_id: str) -> dict:
    """替代外部 MCP 物流边界；真实传输与发现由独立进程集成用例验证。"""
    return {"order_id": order_id, "location": "演示分拨中心", "status": "运输中", "simulated": True}


@pytest.fixture(autouse=True)
def disable_sdk_dotenv_loading(monkeypatch):
    """阻止 pymilvus 导入时将真实 .env 注入进程；Pydantic 显式 dotenv 测试照常运行。"""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")


def stream_response(parts=("您好", "，请问有什么可以帮您？"), finish_reason="stop"):
    records = []
    for part in parts:
        records.append({"delta": {"content": part}, "finish_reason": None})
    records.append({"delta": {}, "finish_reason": finish_reason})
    records[0]["delta"]["role"] = "assistant"
    body = "".join(
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "test-model",
                "choices": [{"index": 0, **record}],
            },
            ensure_ascii=False,
        )
        + "\n\n"
        for record in records
    )
    return httpx.Response(
        200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
    )


def sse_events(response):
    events = []
    for block in response.text.replace("\r\n", "\n").strip().split("\n\n"):
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


class HeldStream(httpx.AsyncByteStream):
    def __init__(self):
        self.release = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        body = stream_response(("已收到问题",)).content
        first, rest = body.split(b"\n\n", 1)
        yield first + b"\n\n"
        await self.release.wait()
        yield rest

    async def aclose(self):
        self.closed.set()


async def start_asgi_stream(app, body, spec_version="2.0", final_gate=None):
    received = asyncio.Queue()
    await received.put(
        {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
    )
    frames = []
    token_seen = asyncio.Event()

    async def send(message):
        if final_gate and message["type"] == "http.response.body" and not message["more_body"]:
            reached, release = final_gate
            reached.set()
            await release.wait()
        frames.append(message)
        if b"event: token\n" in message.get("body", b""):
            token_seen.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat",
        "raw_path": b"/chat",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("app.test", 80),
    }
    task = asyncio.create_task(app(scope, received.get, send))
    try:
        await asyncio.wait_for(token_seen.wait(), timeout=5)
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    return task, received, frames


@pytest.fixture
def harness():
    @asynccontextmanager
    async def open_app(handler=None, **overrides):
        from support_agent.app import create_app
        from support_agent.platform.config import Settings

        calls = []

        async def handle(request):
            assert str(request.url) == "https://upstream.invalid/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer test-key"
            payload = json.loads(request.content)
            calls.append(payload)
            if "问题理解器" in payload["messages"][0]["content"]:
                current = next(
                    m
                    for m in reversed(payload["messages"])
                    if m["role"] == "user" and not m["content"].startswith("以下仅为")
                )
                return stream_response((json.dumps({"question": current["content"]}),))
            if handler:
                return await handler(request, payload)
            if payload["messages"][0]["content"].startswith("将顾客"):
                return stream_response(('{"intent":"订单","confidence":0.99}',))
            return stream_response()

        settings = Settings(
            _env_file=None,
            **{
                "openai_base_url": "https://upstream.invalid/v1",
                "openai_api_key": "test-key",
                "openai_model": "test-model",
                "database_url": "sqlite+aiosqlite:///:memory:",
                "checkpoint_path": ":memory:",
                **overrides,
            },
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as upstream:
            app = create_app(settings=settings, upstream_client=upstream)
            async with app.router.lifespan_context(app):
                from module_support import TestServices

                app.state.repository = TestServices.from_app(app)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://app.test"
                ) as client:
                    yield client, calls, app

    return open_app


def tool_response(calls):
    records = [
        {
            "delta": {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["args"], ensure_ascii=False),
                        },
                    }
                    for index, call in enumerate(calls)
                ]
            },
            "finish_reason": None,
        },
        {"delta": {}, "finish_reason": "tool_calls"},
    ]
    body = "".join(
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-tools",
                "object": "chat.completion.chunk",
                "model": "test-model",
                "choices": [{"index": 0, **record}],
            },
            ensure_ascii=False,
        )
        + "\n\n"
        for record in records
    )
    return httpx.Response(
        200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
    )
