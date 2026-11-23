from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from support_agent.context import fit_messages


def test_trimming_removes_a_whole_old_tool_turn_without_orphaning_observations():
    old = [
        HumanMessage("旧问题" + "甲" * 500),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_order",
                    "args": {"order_id": "1001"},
                    "id": "old",
                }
            ],
        ),
        ToolMessage("旧订单观察" + "乙" * 500, tool_call_id="old"),
        AIMessage("旧回答"),
    ]
    recent = [HumanMessage("新问题"), AIMessage("新回答")]
    messages = fit_messages(SystemMessage("系统"), [*old, *recent], HumanMessage("继续"), 200)
    assert [m.content for m in messages] == ["系统", "新问题", "新回答", "继续"]
