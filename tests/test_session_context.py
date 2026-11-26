from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from support_agent.conversation import ContextBudget, project_history
from support_agent.dialogue.context import assemble_context, limit_tool_results
from support_agent.platform.config import Settings
from support_agent.platform.context import estimate_tokens


def config(**overrides):
    return Settings(
        _env_file=None,
        openai_base_url="https://unused.invalid/v1",
        openai_api_key="test",
        openai_model="test",
        database_url="sqlite+aiosqlite:///:memory:",
        **overrides,
    )


def test_demo_budget_is_derived_from_the_entire_react_configuration():
    settings = config(
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
    )
    budget = ContextBudget.from_settings(settings)
    assert (budget.history, budget.layer1, budget.layer2) == (5650, 3954, 1695)
    larger_tools = ContextBudget.from_settings(
        settings.model_copy(update={"tool_result_max_tokens": 1400})
    )
    assert larger_tools.history == 5250


def test_layer1_keeps_original_turns_layer2_only_shortens_assistant_and_tools():
    settings = config(model_context_window=18000, max_output_tokens=2000, max_agent_steps=3)
    history = []
    for turn in range(1, 8):
        meta = {"turn_id": turn * 10}
        history.extend(
            [
                HumanMessage("订单1001的诉求不能改" + "甲" * 150, additional_kwargs=meta),
                AIMessage(
                    "",
                    tool_calls=[{"name": "query_order", "args": {}, "id": str(turn)}],
                    additional_kwargs=meta,
                ),
                ToolMessage(
                    "工具结果" + "乙" * 1200, tool_call_id=str(turn), additional_kwargs=meta
                ),
                AIMessage("客服答复" + "丙" * 600, additional_kwargs=meta),
            ]
        )
    before = [m.model_dump() for m in history]
    view = project_history(history, settings, summary_upto=10, layer1_from=20, current_id=80)
    assert view.layer1_from > 20
    assert view.layer1 == history[-len(view.layer1) :]
    assert estimate_tokens(view.layer1, settings) <= 3954
    assert all(m.content == history[4 + i * 4].content for i, m in enumerate(view.layer2[::4]))
    assert len(view.layer2[3].content) <= 61
    assert view.layer2[2].content.startswith("[工具结果")
    assert [m.model_dump() for m in history] == before


def test_summary_and_evidence_follow_current_input_and_system_stays_constant():
    current = HumanMessage("最初那个订单怎么说")
    result = assemble_context(
        "固定人设",
        [HumanMessage("旧问题"), AIMessage("旧回复")],
        current,
        "订单1001要求换货，尚未解决。",
        {"evidence": []},
    )
    assert [m.type for m in result] == ["system", "human", "ai", "human", "human"]
    assert result[-2].content == current.content
    assert "订单1001" in result[-1].content
    assert result[0].content == "固定人设"


def test_transient_tool_limit_preserves_the_full_observation_for_checkpoint():
    settings = config(tool_result_max_tokens=100)
    original = ToolMessage("完整原文" * 500, tool_call_id="large")
    projected = limit_tool_results([original], settings)
    assert len(original.content) == 2000
    assert estimate_tokens(projected, settings) <= 100
    assert "tool_result_too_large" in projected[0].content
