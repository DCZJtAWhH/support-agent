import asyncio
from uuid import uuid4

import pytest


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/chat", {"message": "   "}),
        ("/chat", {"message": "你好", "system_prompt": "替换系统指令"}),
        ("/chat", {"message": 123}),
        ("/chat", {"message": "你好", "conversation_id": "bad-id"}),
        ("/after-sales/extract", {"description": "\n\t"}),
        ("/after-sales/extract", {"description": "退款", "model": "override"}),
    ],
)
def test_invalid_requests_are_rejected_before_calling_upstream(harness, path, body):
    async def scenario():
        async with harness() as (client, calls, _):
            response = await client.post(path, json=body)
            assert response.status_code == 422
            assert calls == []

    asyncio.run(scenario())


def test_model_configuration_is_loaded_from_utf8_dotenv(monkeypatch, tmp_path):
    import json

    import httpx
    from conftest import sse_events, stream_response

    from support_agent.app import create_app

    monkeypatch.chdir(tmp_path)
    for key in ("OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_BASE_URL=https://dotenv.invalid/v1\nOPENAI_MODEL=配置模型\nOPENAI_API_KEY=local-test-key\n",
        encoding="utf-8",
    )

    async def scenario():
        calls = []

        async def handler(request):
            assert str(request.url) == "https://dotenv.invalid/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer local-test-key"
            calls.append(json.loads(request.content))
            return stream_response()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
            app = create_app(upstream_client=upstream)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://app.test"
                ) as client:
                    response = await client.post("/chat", json={"message": "你好"})
                    assert sse_events(response)[-1][0] == "done"
                    assert calls[0]["model"] == "配置模型"

    asyncio.run(scenario())


def test_unknown_conversation_returns_not_found_without_upstream_request(harness):
    async def scenario():
        async with harness() as (client, calls, _):
            response = await client.post(
                "/chat", json={"message": "继续", "conversation_id": str(uuid4())}
            )
            assert response.status_code == 404
            assert calls == []

    asyncio.run(scenario())
