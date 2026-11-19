"""按输入 token 预算裁剪客服对话历史，保留系统提示与当前消息。"""

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately


class ContextBudgetExceeded(ValueError):
    pass


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
