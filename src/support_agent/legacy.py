"""集中兼容旧轮次审计和检查点，正常反馈只读取持久召回快照。"""

import json


def legacy_retrieval_reader(read_checkpoint):
    """绑定对话公开兼容查询，返回按原轮次恢复旧证据的适配函数。"""

    async def recover(conversation_id, request_id, messages):
        """只恢复指定轮次；空召回是有效快照，绝不以最近轮次代替。"""
        snapshot = None
        for message in messages:
            if message["role"] not in {"workflow", "tool"}:
                continue
            try:
                data = json.loads(message["content"])
            except (ValueError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            if data.get("retrieval_snapshot") is not None:
                snapshot = data["retrieval_snapshot"]
            elif data.get("sources"):
                snapshot = data["sources"]
            elif isinstance(data.get("data"), dict) and "matches" in data["data"]:
                snapshot = data["data"]["matches"]
            elif "retrieve" in data.get("trace", []):
                snapshot = []
        if snapshot is None and read_checkpoint:
            snapshot = await read_checkpoint(conversation_id, request_id)
        return snapshot

    return recover
