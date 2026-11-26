"""按输入 token 预算裁剪客服对话历史，保留系统提示与当前消息。"""

import json
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.messages.utils import trim_messages

from support_agent.platform.context import estimate_tokens


@dataclass(frozen=True)
class ContextBudget:
    """保存从模型窗口扣除固定开销及单轮峰值后得到的分层历史预算。"""

    history: int
    layer1: int
    layer2: int
    fixed: int
    peak: int

    @classmethod
    def from_settings(cls, config):
        """按同一 token 口径倒推历史额度；每步工具结果预算是整批结果的合计上限。"""
        fixed = (
            config.system_tools_tokens
            + config.rerank_top_k * config.evidence_tokens_per_result
            + config.summary_injection_tokens
            + config.max_output_tokens
            + config.token_safety_margin
        )
        peak = config.max_user_input_tokens + (config.agent_max_steps - 1) * (
            config.max_output_tokens + config.tool_result_max_tokens
        )
        history = max(
            0,
            min(
                config.history_target_rounds * config.steady_round_tokens,
                config.context_window_tokens - fixed - peak,
            ),
        )
        return cls(history, int(history * 0.7), int(history * 0.3), fixed, peak)


def turn_id(message) -> int:
    """读取消息所属轮次的用户流水 ID；工具不入消息表时仍可按同一边界定位。"""
    return int(message.additional_kwargs.get("turn_id", 0))


@dataclass
class HistoryProjection:
    """提供独立于完整 checkpoint 的两层投影及待摘要的原文批次。"""

    layer1: list
    layer2: list
    retired: list
    layer1_from: int
    layer2_tokens: int

    @property
    def window(self):
        """按中期截短、近期原文的固定顺序返回模型滑窗。"""
        return [*self.layer2, *self.layer1]


def project_history(
    history, config, *, summary_upto=0, layer1_from=0, current_id, history_budget=None
):
    """只投影已完成历史，按完整轮次前移层 1 锚点，不改写任何原始消息。

    层 1 包含起点，摘要包含终点；层 2 触发用量含 checkpoint 工具结果的实际原文。
    """
    budget = ContextBudget.from_settings(config)
    layer1_budget = int(history_budget * 0.7) if history_budget is not None else budget.layer1
    unsummarized = [m for m in history if turn_id(m) > (summary_upto or 0)]
    recent = [m for m in unsummarized if turn_id(m) >= (layer1_from or 0)]
    retained = trim_messages(
        recent,
        max_tokens=layer1_budget,
        strategy="last",
        start_on="human",
        token_counter=lambda items: estimate_tokens(items, config),
        allow_partial=False,
    )
    boundary = turn_id(retained[0]) if retained else current_id
    boundary = max(boundary, layer1_from or 0)
    layer1 = [m for m in unsummarized if turn_id(m) >= boundary]
    retired = [m for m in unsummarized if turn_id(m) < boundary]
    layer2 = []
    for message in retired:
        if isinstance(message, ToolMessage):
            layer2.append(
                message.model_copy(
                    update={
                        "content": f"[工具结果 {message.name or message.tool_call_id} 已省略]",
                    }
                )
            )
        elif isinstance(message, AIMessage):
            layer2.append(
                message.model_copy(
                    update={
                        "content": message.content[: config.layer2_assistant_chars],
                    }
                )
            )
        else:
            layer2.append(message)
    return HistoryProjection(layer1, layer2, retired, boundary, estimate_tokens(retired, config))


def routing_history(history):
    """将共享滑窗封装为 JSON 任务的数据，避免无工具定义的分流模型重入历史工具协议。"""
    if not history:
        return []
    return [
        HumanMessage(
            "以下是历史对话数据，仅用于理解当前问题，不继续历史任务或调用工具。\n"
            + json.dumps(convert_to_openai_messages(history), ensure_ascii=False)
        )
    ]
