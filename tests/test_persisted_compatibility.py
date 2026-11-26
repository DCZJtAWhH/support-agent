import asyncio
import gzip
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from conftest import sse_events, stream_response


@pytest.mark.parametrize("case", ["completed", "ticket_waiting", "ticket_failed", "order_waiting"])
def test_pre_modular_requests_replay_or_resume_through_existing_http(harness, tmp_path, case):
    fixture = json.loads(
        gzip.decompress((Path(__file__).parent / "fixtures/pre-modular.json.gz").read_bytes())
    )
    for name in ("business", "checkpoint"):
        with closing(sqlite3.connect(tmp_path / f"{name}.db")) as connection:
            connection.executescript(fixture[name])

    class EmptyKnowledge:
        async def search_many(self, *args, **kwargs):
            return []

    async def handler(request, payload):
        if "查询扩写器" in payload["messages"][0]["content"]:
            return stream_response(('{"queries":["退货条件","退货期限"]}',))
        result = json.loads(payload["messages"][-1]["content"])
        return stream_response(("已登记待人工跟进，工单号：" + result["data"]["ticket_no"],))

    async def scenario():
        async with harness(
            handler,
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'business.db'}",
            checkpoint_path=str(tmp_path / "checkpoint.db"),
            flywheel_interval_seconds=0,
            eval_interval_seconds=0,
        ) as (client, calls, app):
            app.state.knowledge.retriever = EmptyKnowledge()
            body = dict(fixture["cases"][case])
            if case == "ticket_waiting":
                body.update(ticket_confirmation=True, ticket_tool_call_id="legacy-ticket")
            if case == "order_waiting":
                body["selected_order_id"] = "1001"
            response = await client.post("/chat", json=body)
            assert response.status_code == 200, response.text
            assert sse_events(response)[-1][0] == "done", response.text
            audit = (await client.get("/conversations/" + body["conversation_id"])).json()
            original = [m for m in audit["messages"] if m["position"] == 0]
            assert len(original) == 1 and original[0]["state"] == "completed"
            if case.startswith("ticket"):
                assert len(audit["tickets"]) == 1
                assert audit["tickets"][0]["ticket_no"] in response.text
                if case == "ticket_failed":
                    assert audit["tickets"][0]["ticket_no"] == fixture["ticket_no"]
            if case == "completed":
                assert calls == []
            before = len(calls)
            replay = await client.post("/chat", json=body)
            assert sse_events(replay)[-1][0] == "done"
            assert len(calls) == before
            changed = await client.post("/chat", json={**body, "message": "另一个问题"})
            assert changed.status_code == 409

    asyncio.run(scenario())
