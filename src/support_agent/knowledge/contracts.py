"""知识模块对外提供的版本快照。"""

from typing import TypedDict


class CorpusVersion(TypedDict):
    """一次查询中的有效知识数量、内容摘要及待同步数，不包含存储对象。"""

    size: int
    hash: str
    pending_count: int
