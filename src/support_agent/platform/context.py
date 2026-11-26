"""按输入 token 预算裁剪客服对话历史，保留系统提示与当前消息。"""

import math
from dataclasses import dataclass

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_core.messages.utils import count_tokens_approximately


class ContextBudgetExceeded(ValueError):
    pass


def estimate_tokens(messages, config) -> int:
    """统一估算文本、消息框架及工具申请的 token；中文折算参数与预算同时配置。"""
    if isinstance(messages, str):
        return math.ceil(len(messages) / config.token_chars_per_token)
    return (
        math.ceil(
            count_tokens_approximately(
                messages,
                chars_per_token=config.token_chars_per_token,
            )
        )
        if messages
        else 0
    )


def fit_messages(
    system: BaseMessage, history: list[BaseMessage], current: BaseMessage, budget: int
) -> list[BaseMessage]:
    """从最早的完整轮次开始裁剪历史，返回预算内的消息列表。

    history 按完整轮次排列，轮次可含工具申请及结果；中文按每字符约一个 token 估算。
    始终保留系统提示和当前消息，两者仍超预算时抛出 ContextBudgetExceeded。
    """
    messages = [system, *history, current]
    while count_tokens_approximately(messages, chars_per_token=1.0) > budget:
        if len(messages) <= 2:
            raise ContextBudgetExceeded("系统提示与当前输入超过 token 预算，请缩短输入或调整配置。")
        end = next(
            (i for i in range(2, len(messages)) if isinstance(messages[i], HumanMessage)),
            len(messages) - 1,
        )
        del messages[1:end]
    return messages


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
