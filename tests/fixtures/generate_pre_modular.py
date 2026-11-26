"""用改造前代码的真实 ASGI 入口生成离线持久夹具；不作为自动化测试运行。"""

import argparse
import asyncio
import gzip
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import httpx


async def generate(source, output):
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path[:0] = [str(source / "src"), str(source / "tests")]
    from conftest import sse_events, stream_response, tool_response
    from support_agent.config import Settings

    from support_agent.app import create_app

    async def handler(request):
        payload = json.loads(request.content)
        prompt = payload["messages"][0]["content"]
        if "问题理解器" in prompt:
            question = next(
                m["content"] for m in reversed(payload["messages"]) if m["role"] == "user"
            )
            return stream_response((json.dumps({"question": question}),))
        if prompt.startswith("将顾客"):
            question = payload["messages"][-1]["content"]
            intent = "投诉" if "投诉" in question else "退款退货"
            return stream_response((json.dumps({"intent": intent, "confidence": 0.99}),))
        if payload["messages"][-1]["role"] == "tool":
            return httpx.Response(503, json={"error": {"message": "offline interruption"}})
        return tool_response(
            [
                {
                    "id": "legacy-ticket",
                    "name": "create_ticket",
                    "args": {"description": "水壶不加热", "ticket_type": "维修"},
                }
            ]
        )

    with TemporaryDirectory() as directory:
        directory = Path(directory)
        config = Settings(
            _env_file=None,
            openai_base_url="https://legacy.invalid/v1",
            openai_api_key="offline",
            openai_model="offline",
            database_url=f"sqlite+aiosqlite:///{directory / 'business.db'}",
            checkpoint_path=str(directory / "checkpoint.db"),
            flywheel_interval_seconds=0,
            eval_interval_seconds=0,
            langfuse_enabled=False,
        )
        cases = {}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
            app = create_app(settings=config, upstream_client=upstream)
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app), base_url="http://legacy.test"
                ) as client:
                    for index, (name, question) in enumerate(
                        [
                            ("completed", "我要投诉"),
                            ("ticket_waiting", "水壶不加热，帮我建工单"),
                            ("ticket_failed", "水壶不加热，帮我建工单"),
                            ("order_waiting", "这个能退吗"),
                        ],
                        start=1,
                    ):
                        body = {"message": question, "request_id": str(UUID(int=index))}
                        response = await client.post("/chat", json=body)
                        events = sse_events(response)
                        assert events[-1][0] == "done", response.text
                        body["conversation_id"] = events[0][1]["conversation_id"]
                        if name == "ticket_failed":
                            body.update(
                                ticket_confirmation=True, ticket_tool_call_id="legacy-ticket"
                            )
                            failed = await client.post("/chat", json=body)
                            assert sse_events(failed)[-1][0] == "error", failed.text
                        cases[name] = body
                    audit = (
                        await client.get(
                            "/conversations/" + cases["ticket_failed"]["conversation_id"]
                        )
                    ).json()
                    assert len(audit["tickets"]) == 1
                    ticket_no = audit["tickets"][0]["ticket_no"]
        fixture = {
            "baseline": "510e000d2ede7ad2ca2ab2749b0b626177667edd",
            "cases": cases,
            "ticket_no": ticket_no,
        }
        for name in ("business", "checkpoint"):
            with closing(sqlite3.connect(directory / f"{name}.db")) as connection:
                fixture[name] = "\n".join(connection.iterdump())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(
            gzip.compress(json.dumps(fixture, ensure_ascii=False).encode("utf-8"), mtime=0)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    asyncio.run(generate(args.source.resolve(), args.output.resolve()))
