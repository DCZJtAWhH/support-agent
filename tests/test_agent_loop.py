import asyncio

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from support_agent.agent import run_bare_agent
from support_agent.tools import ToolRegistry, build_tools


class ScriptedModel:
    """替代外部模型端点，按工具结果返回下一次可观察决策。"""

    def __init__(self):
        self.inputs = []

    def bind_tools(self, tools):
        assert "create_ticket" not in [t.name for t in tools]
        return self

    async def astream(self, messages):
        self.inputs.append(list(messages))
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if len(results) < 2:
            name = "query_order" if not results else "query_logistics"
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": name,
                        "args": '{"order_id":"1001"}',
                        "id": f"step_{len(results)}",
                        "index": 0,
                    }
                ],
                response_metadata={"finish_reason": "tool_calls"},
            )
        else:
            yield AIMessageChunk(content="演示物流：", response_metadata={})
            yield AIMessageChunk(content="运输中", response_metadata={"finish_reason": "stop"})


def test_bare_loop_uses_observations_for_multiple_tool_steps_then_streams_answer():
    async def scenario():
        model = ScriptedModel()
        registry = ToolRegistry(build_tools(None, "conversation", "request", "先查订单再查物流"))
        events = [
            event
            async for event in run_bare_agent(
                model,
                registry,
                [HumanMessage("先查订单1001，已发货再查物流")],
            )
        ]
        assert "".join(e["text"] for e in events if e["event"] == "token") == "演示物流：运输中"
        assert [
            e["name"] for e in events if e["event"] == "tool_status" and e["state"] == "succeeded"
        ] == ["query_order", "query_logistics"]
        assert len(model.inputs) == 3

    asyncio.run(scenario())


def test_loop_stops_without_calling_model_when_turn_budget_cannot_cover_a_call():
    async def scenario():
        model = ScriptedModel()
        registry = ToolRegistry(build_tools(None, "c", "r", "查订单"))
        events = [
            e
            async for e in run_bare_agent(
                model,
                registry,
                [HumanMessage("查订单")],
                token_budget=10,
            )
        ]
        assert not model.inputs
        assert events[-1]["stop_reason"] == "budget"
        assert events[-1]["actions"] == ["handoff", "create_ticket"]

    asyncio.run(scenario())


def test_loop_can_ask_for_missing_information_without_using_a_tool():
    class ClarifyingModel(ScriptedModel):
        async def astream(self, messages):
            self.inputs.append(messages)
            yield AIMessageChunk(
                content="请提供订单号。",
                response_metadata={"finish_reason": "stop"},
            )

    async def scenario():
        registry = ToolRegistry(build_tools(None, "c", "r", "查物流"))
        events = [
            e
            async for e in run_bare_agent(
                ClarifyingModel(),
                registry,
                [HumanMessage("查物流")],
            )
        ]
        assert events[-1]["answer"] == "请提供订单号。"
        assert not any(e["event"] == "tool_status" for e in events)

    asyncio.run(scenario())
