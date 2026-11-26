from decimal import Decimal

import pytest


@pytest.fixture(autouse=True)
def fixed_pricing_clock(monkeypatch):
    from datetime import datetime

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 24, 10, tzinfo=tz)

    monkeypatch.setattr("support_agent.platform.model_budget.datetime", FixedDatetime)


def test_budget_is_reserved_before_spending_and_persists_across_runs(tmp_path):
    from support_agent.platform.model_budget import BudgetExceeded, BudgetLedger

    path = tmp_path / "budget.sqlite"
    first = BudgetLedger(path)
    reservation = first.reserve("deepseek-flash", Decimal("0.70"))
    assert first.report()["committed_cny"] == "0.70"
    first.settle(reservation, Decimal("0.40"))
    restarted = BudgetLedger(path)
    with pytest.raises(BudgetExceeded):
        restarted.reserve("deepseek-flash", Decimal("0.61"))
    assert restarted.report()["committed_cny"] == "0.40"
    reservation = restarted.reserve("deepseek-flash", Decimal("0.60"))
    restarted.settle(reservation, Decimal("0.60"))
    with pytest.raises(BudgetExceeded):
        restarted.reserve("deepseek-flash", Decimal("0.000001"))
    assert restarted.report()["committed_cny"] == "1.00"


def test_http_budget_stops_unknown_usage_and_blocks_later_calls(tmp_path):
    import asyncio

    import httpx

    from support_agent.platform.model_budget import BudgetExceeded, BudgetLedger, BudgetTransport

    async def scenario():
        sent = []

        async def handler(request):
            sent.append(1)
            return httpx.Response(200, json={"choices": []})

        ledger = BudgetLedger(tmp_path / "unknown.sqlite")
        transport = BudgetTransport(httpx.MockTransport(handler), ledger)
        payload = {
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 10,
        }
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(BudgetExceeded):
                await client.post("https://api.deepseek.com/chat/completions", json=payload)
            with pytest.raises(BudgetExceeded):
                await client.post("https://api.deepseek.com/chat/completions", json=payload)
        assert len(sent) == 1
        assert ledger.report()["halted"]
        assert Decimal(ledger.report()["committed_cny"]) > 0

    asyncio.run(scenario())


@pytest.mark.parametrize("data_prefix", ["data: ", "data:"])
def test_streaming_sdk_records_usage_before_closing_after_done(tmp_path, data_prefix):
    import asyncio
    import json

    import httpx
    from conftest import stream_response

    from support_agent.platform.config import Settings
    from support_agent.platform.model_budget import BudgetLedger, BudgetTransport
    from support_agent.platform.model_gateway import create_chat_model

    async def scenario():
        response = stream_response(("您好",))
        blocks = response.text.split("\n\n")
        final = json.loads(blocks[-3].removeprefix("data: "))
        final["usage"] = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        blocks[-3] = "data: " + json.dumps(final)
        body = "\n\n".join(blocks)
        body = body.replace("data: ", data_prefix)

        async def handler(request):
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        ledger = BudgetLedger(tmp_path / "sdk-stream.sqlite")
        config = Settings(
            _env_file=None,
            openai_base_url="https://api.deepseek.com",
            openai_api_key="test",
            openai_model="deepseek-flash",
            database_url="sqlite+aiosqlite:///:memory:",
        )
        async with httpx.AsyncClient(
            transport=BudgetTransport(httpx.MockTransport(handler), ledger)
        ) as client:
            model = create_chat_model(config, client)
            _ = [chunk async for chunk in model.astream("你好")]
        assert not ledger.report()["halted"]
        assert Decimal(ledger.report()["committed_cny"]) == Decimal("0.00028")

    asyncio.run(scenario())


def test_model_sdk_uses_budgeted_client_and_cannot_bypass_the_gate(tmp_path):
    import asyncio

    import httpx

    from support_agent.platform.config import Settings
    from support_agent.platform.model_budget import BudgetExceeded, BudgetLedger, BudgetTransport
    from support_agent.platform.model_gateway import create_chat_model

    async def scenario():
        sent = []

        async def handler(request):
            sent.append(1)
            pytest.fail("模型SDK不得绕过费用闸")

        ledger = BudgetLedger(tmp_path / "sdk.sqlite")
        receipt = ledger.reserve("deepseek-flash", Decimal("0.99"))
        ledger.settle(receipt, Decimal("0.99"))
        config = Settings(
            _env_file=None,
            openai_base_url="https://api.deepseek.com",
            openai_api_key="test",
            openai_model="deepseek-flash",
            database_url="sqlite+aiosqlite:///:memory:",
        )
        async with httpx.AsyncClient(
            transport=BudgetTransport(httpx.MockTransport(handler), ledger)
        ) as client:
            model = create_chat_model(config, client)
            with pytest.raises(BudgetExceeded):
                await model.ainvoke("你好")
        assert not sent

    asyncio.run(scenario())


def test_http_over_budget_is_rejected_before_network_and_stale_prices_are_rejected(tmp_path):
    import asyncio
    from datetime import date

    import httpx

    from support_agent.platform.model_budget import BudgetExceeded, BudgetLedger, BudgetTransport

    async def scenario():
        sent = []

        async def handler(request):
            sent.append(1)
            pytest.fail("超预算不得发送网络请求")

        ledger = BudgetLedger(tmp_path / "nearly-full.sqlite")
        receipt = ledger.reserve("deepseek-flash", Decimal("0.99"))
        ledger.settle(receipt, Decimal("0.99"))
        for pricing_day in [date(2026, 11, 24), date(2000, 1, 1)]:
            transport = BudgetTransport(
                httpx.MockTransport(handler), ledger, pricing_verified_on=pricing_day
            )
            async with httpx.AsyncClient(transport=transport) as client:
                with pytest.raises(BudgetExceeded):
                    await client.post(
                        "https://api.deepseek.com/chat/completions",
                        json={
                            "model": "deepseek-flash",
                            "messages": [{"role": "user", "content": "你好"}],
                            "max_tokens": 10,
                        },
                    )
        assert not sent
        assert ledger.report()["committed_cny"] == "0.99"

    asyncio.run(scenario())


@pytest.mark.parametrize("complete", [True, False])
def test_streaming_usage_is_charged_once_and_incomplete_stream_keeps_reservation(
    tmp_path, complete
):
    import asyncio
    import json

    import httpx

    from support_agent.platform.model_budget import BudgetExceeded, BudgetLedger, BudgetTransport

    async def scenario():
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "completion_tokens_details": {"reasoning_tokens": 6},
        }
        body = "data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n"
        if complete:
            body += "data: [DONE]\n\n"

        class SplitStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for byte in body.encode():
                    yield bytes([byte])

        async def handler(request):
            return httpx.Response(200, stream=SplitStream())

        ledger = BudgetLedger(tmp_path / "stream.sqlite")
        async with httpx.AsyncClient(
            transport=BudgetTransport(httpx.MockTransport(handler), ledger)
        ) as client:
            call = client.post(
                "https://api.deepseek.com/chat/completions",
                json={
                    "model": "deepseek-flash",
                    "messages": [{"role": "user", "content": "你好"}],
                    "max_tokens": 10,
                    "stream": True,
                },
            )
            if complete:
                await call
                assert Decimal(ledger.report()["committed_cny"]) == Decimal("0.00028")
                assert not ledger.report()["halted"]
            else:
                with pytest.raises(BudgetExceeded):
                    await call
                assert ledger.report()["halted"]

    asyncio.run(scenario())
