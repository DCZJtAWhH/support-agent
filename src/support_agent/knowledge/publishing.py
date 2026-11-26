"""以稳定来源身份发布核准知识，失败保留 pending 原文供重试。"""

from .chunks import KnowledgeDraft
from .store import synchronize


class KnowledgePublisher:
    def __init__(self, store, embeddings, vectors):
        """绑定本模块原文存储与索引边界，不接收其他模块的事务。"""
        self.store, self.embeddings, self.vectors = store, embeddings, vectors

    async def publish_answer(self, source, question, answer):
        """保存人工核准内容并同步；同来源同内容可安全重试并复用知识主键。"""
        await self.store.replace_source(
            source, [KnowledgeDraft("客服知识", [question], answer, question)]
        )
        await synchronize(self.store, self.embeddings, self.vectors, source_ref=source)
