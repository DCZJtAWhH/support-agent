"""模型请求与历史日志的消息序列化。"""

from langchain_core.messages import convert_to_openai_messages


def context_data(messages):
    """转换正文与工具申请用于可读日志；私有协议字段仅存检查点并在出站时回传。"""
    rows = convert_to_openai_messages(messages)
    for row in rows:
        if row["role"] == "assistant" and row.get("tool_calls"):
            # langchain-openai 将空的工具申请正文规范为 null，日志与线上请求保持一致。
            row["content"] = row.get("content") or None
        if row["role"] == "tool":
            row.pop("name", None)
    return rows
