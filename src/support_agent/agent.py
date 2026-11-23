"""不依赖 LangGraph 的裸工具循环，以及图节点复用的模型与工具单步。"""

import json
from contextlib import aclosing

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from support_agent.errors import ModelResponseError

STOP_REPLY = (
    "本轮查询已达到处理步数或 token 预算上限。请缩小问题范围后继续，也可以选择转人工或建工单。"
)


def agent_tools(registry):
    """只授予现有只读业务工具；检索由图强制执行，建单仅由用户确认接口执行。"""
    return [t for name, t in registry.tools.items() if name not in {"create_ticket", "query_faq"}]


def suggested_actions(text):
    """把模型回复中的人工跟进建议转成无副作用的独立选项。"""
    actions = []
    if "转人工" in text or "人工客服" in text:
        actions.append("handoff")
    if "工单" in text:
        actions.append("create_ticket")
    return actions


async def model_step(model, registry, messages, *, input_budget=7000, output_budget=1024):
    """流式执行一次模型决策，最后返回完整消息及保守 token 估算；不执行工具。"""
    tools = agent_tools(registry)
    schema_cost = len(json.dumps([convert_to_openai_tool(t) for t in tools], ensure_ascii=False))
    input_tokens = int(count_tokens_approximately(messages, chars_per_token=1.0)) + schema_cost
    if input_tokens > input_budget:
        yield {"event": "limit"}
        return
    aggregate = None
    output_chars = 0
    async with aclosing(model.bind_tools(tools).astream(messages)) as chunks:
        async for chunk in chunks:
            if not isinstance(chunk.content, str) or chunk.additional_kwargs.get("refusal"):
                raise ModelResponseError("模型返回了不支持的内容或拒绝响应。")
            aggregate = chunk if aggregate is None else aggregate + chunk
            output_chars += len(chunk.content) + sum(
                len(c.get("args") or "") for c in chunk.tool_call_chunks
            )
            if output_chars > output_budget * 4:
                raise ModelResponseError("模型输出超过本轮预算。")
            if chunk.content:
                yield {"event": "token", "text": chunk.content}
    if aggregate is None or aggregate.invalid_tool_calls:
        raise ModelResponseError("模型没有返回有效内容，或工具参数 JSON 无效。")
    calls = aggregate.tool_calls
    reason = aggregate.response_metadata.get("finish_reason")
    if calls:
        ids = [c["id"] for c in calls]
        if reason != "tool_calls" or any(not i for i in ids) or len(ids) != len(set(ids)):
            raise ModelResponseError("工具调用申请不完整或标识重复。")
    elif reason != "stop" or not aggregate.content.strip():
        raise ModelResponseError("模型回复为空、被截断或未正常结束。")
    usage = aggregate.usage_metadata or {}
    yield {
        "event": "decision",
        "message": AIMessage(content=aggregate.content, tool_calls=calls),
        # 上游每次受 max_output_tokens 约束；无 usage 时用保守输入估算与输出上限计账。
        "tokens": max(input_tokens + output_budget, usage.get("total_tokens", 0)),
    }


async def tool_step(registry, decision):
    """顺序执行一批只读工具并产出轨迹和观察结果；拒绝越权的建单与检索调用。"""
    allowed = {t.name for t in agent_tools(registry)}
    for call in decision.tool_calls:
        yield {
            "event": "tool_status",
            "name": call["name"],
            "tool_call_id": call["id"],
            "state": "running",
        }
        if call["name"] not in allowed:
            result = registry.failure("forbidden_tool", "此工具不能由 Agent 执行。", 0)
        else:
            result = await registry.execute(call["name"], call["args"])
        yield {
            "event": "tool_status",
            "name": call["name"],
            "tool_call_id": call["id"],
            "state": "succeeded" if result["ok"] else "failed",
            "attempts": result["attempts"],
        }
        yield {
            "event": "observation",
            "message": ToolMessage(
                json.dumps(result, ensure_ascii=False),
                tool_call_id=call["id"],
            ),
        }


async def run_bare_agent(
    model,
    registry,
    messages,
    *,
    max_steps=6,
    token_budget=30000,
    input_budget=7000,
    output_budget=1024,
):
    """手写调用模型→执行工具→回喂结果的循环；无工具、步数或累计预算耗尽时停止。"""
    messages = list(messages)
    consumed = 0
    for _ in range(max_steps):
        # 每次调用前预留一次最大输入与输出，避免到达上限后才发现超支。
        if consumed + input_budget + output_budget > token_budget:
            break
        decision = None
        async for event in model_step(
            model,
            registry,
            messages,
            input_budget=input_budget,
            output_budget=output_budget,
        ):
            if event["event"] == "decision":
                decision = event["message"]
                consumed += event["tokens"]
            elif event["event"] != "limit":
                yield event
        if decision is None:
            break
        messages.append(decision)
        if not decision.tool_calls:
            yield {
                "event": "result",
                "answer": decision.content,
                "tokens": consumed,
                "actions": suggested_actions(decision.content),
            }
            return
        if len(decision.tool_calls) > 4:
            break
        async for event in tool_step(registry, decision):
            if event["event"] == "observation":
                messages.append(event["message"])
            else:
                yield event
    yield {"event": "token", "text": STOP_REPLY}
    yield {
        "event": "result",
        "answer": STOP_REPLY,
        "tokens": consumed,
        "actions": ["handoff", "create_ticket"],
        "stop_reason": "budget",
    }
