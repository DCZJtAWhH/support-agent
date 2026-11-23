import asyncio

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from test_session_context import config

from support_agent.routing import RouterLanguage


def test_json_router_receives_historical_tools_as_data_without_reentering_tool_protocol():
    class JsonModel:
        def bind(self, **kwargs):
            return self

        async def astream(self, messages):
            # JSON 任务不绑定工具定义；历史工具协议会使上游回到客服角色并耗尽推理预算。
            assert not any(m.type == "tool" or getattr(m, "tool_calls", None) for m in messages)
            assert "1001" in messages[1].content and "已签收" in messages[1].content
            yield AIMessageChunk(
                content='{"intent":"订单","confidence":0.99}',
                response_metadata={"finish_reason": "stop"},
            )

    async def scenario():
        history = [
            HumanMessage("订单1001状态"),
            AIMessage(
                "", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "q"}]
            ),
            ToolMessage("已签收", tool_call_id="q"),
            AIMessage("演示订单已签收"),
        ]
        result = await RouterLanguage(JsonModel(), config()).classify(
            "订单1002状态", history=history
        )
        assert result.intent == "订单"
        assert history[2].type == "tool"

    asyncio.run(scenario())


def test_json_router_retries_one_blank_stop_response_without_relaxing_the_schema():
    class BlankOnce:
        def __init__(self):
            self.calls = 0

        def bind(self, **kwargs):
            return self

        async def astream(self, messages):
            self.calls += 1
            yield AIMessageChunk(
                content="  " if self.calls == 1 else '{"intent":"订单","confidence":0.99}',
                response_metadata={"finish_reason": "stop"},
            )

    async def scenario():
        model = BlankOnce()
        result = await RouterLanguage(model, config()).classify("订单1001状态")
        assert result.intent == "订单"
        assert model.calls == 2

    asyncio.run(scenario())
