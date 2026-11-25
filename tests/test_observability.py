import asyncio
import json

from conftest import stream_response


def test_cost_api_groups_actual_model_usage_by_final_intent(harness):
    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            content = '{"intent":"订单","confidence":0.99}'
        else:
            content = "这是演示订单数据。"
        response = stream_response((content,))
        usage = {
            "id": "usage",
            "object": "chat.completion.chunk",
            "model": "test-model",
            "choices": [],
            "usage": {"prompt_tokens": 112, "completion_tokens": 8, "total_tokens": 120},
        }
        body = response.text.replace(
            "data: [DONE]", "data: " + json.dumps(usage) + "\n\ndata: [DONE]"
        )
        return type(response)(200, text=body, headers=response.headers)

    async def scenario():
        async with harness(handler) as (client, _, app):
            response = await client.post("/chat", json={"message": "查询订单"})
            assert "event: done" in response.text
            costs = await client.get("/api/costs")
            assert costs.status_code == 200
            # harness 的问题理解响应没有 usage；分类与生成各返回 120 个真实 token。
            assert costs.json() == [
                {
                    "intent": "订单",
                    "input_tokens": 224,
                    "output_tokens": 16,
                    "total_tokens": 240,
                    "request_count": 1,
                    "missing_usage_count": 1,
                }
            ]

    asyncio.run(scenario())


def test_explicit_non_thinking_mode_reaches_existing_model_without_switching_provider(harness):
    async def scenario():
        async with harness(model_provider="deepseek", model_thinking="disabled") as context:
            client, calls, _ = context
            response = await client.post("/chat", json={"message": "查询订单"})
            assert "event: done" in response.text
            assert all(c["thinking"] == {"type": "disabled"} for c in calls)
            assert {c["model"] for c in calls} == {"test-model"}

    asyncio.run(scenario())


def test_workflow_emits_nested_langfuse_nodes_prompts_usage_and_intent(harness):
    from uuid import uuid4

    from langfuse import Langfuse
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from pydantic import SecretStr

    from support_agent.workflow import SupportWorkflow

    async def scenario():
        exporter = InMemorySpanExporter()
        public_key = "pk-lf-" + uuid4().hex
        trace_client = Langfuse(
            public_key=public_key,
            secret_key="test",
            base_url="http://127.0.0.1:3000",
            span_exporter=exporter,
        )
        try:
            async with harness() as (_, _, app):
                repo = app.state.repository
                cid, rid = str(uuid4()), str(uuid4())
                await repo.begin_request(cid, rid, "查询订单")
                async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
                    config = app.state.chat.config.model_copy(
                        update={
                            "langfuse_public_key": SecretStr(public_key),
                            "langfuse_secret_key": SecretStr("test"),
                        }
                    )
                    workflow = SupportWorkflow(
                        app.state.model, repo, config, saver, trace_client=trace_client
                    )
                    events = [e async for e in workflow.stream(cid, rid, "查询订单")]
                    assert any(e["event"] == "token" for e in events)
            trace_client.flush()
            spans = exporter.get_finished_spans()
            root = next(s for s in spans if s.name == "support-request")
            assert root.attributes["langfuse.observation.metadata.intent"] == "订单"
            assert {"resolve", "classify", "agent", "log"} <= {s.name for s in spans}
            assert len({s.context.trace_id for s in spans}) == 1
            assert any(
                "查询订单"
                in str(json.loads(s.attributes.get("langfuse.observation.input", "null")))
                for s in spans
                if s.attributes.get("langfuse.observation.type") == "generation"
            )
            assert all(s.end_time >= s.start_time for s in spans)
        finally:
            trace_client.shutdown()

    asyncio.run(scenario())
