from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def completed_history(rows):
    """恢复已完成轮次的消息，首次升级没有图检查点时保留旧会话上下文。"""
    completed = {r["request_id"] for r in rows if r["position"] == 0 and r["state"] == "completed"}
    messages = []
    origins = {r["request_id"]: r["id"] for r in rows if r["position"] == 0}
    for row in rows:
        if row["request_id"] not in completed:
            continue
        meta = {
            "turn_id": origins[row["request_id"]],
            "msg_id": row["id"],
            "request_id": row["request_id"],
        }
        options = {"id": f"import:{row['id']}", "additional_kwargs": meta}
        if row["role"] == "user":
            messages.append(HumanMessage(row["content"], **options))
        elif row["role"] == "tool":
            messages.append(
                ToolMessage(row["content"], tool_call_id=row["tool_call_id"], **options)
            )
        elif row["role"] == "assistant":
            messages.append(
                AIMessage(row["content"], tool_calls=row["tool_calls"] or [], **options)
            )
    return messages
