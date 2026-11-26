"""对话采集边界只记录证据和未解决问题，不执行审核归并发布。"""

import logging

logger = logging.getLogger(__name__)


class ImprovementCollector:
    def __init__(self, store):
        """绑定问题池存储，故障由该边界统一降级。"""
        self.store = store

    async def record_retrieval_snapshot(self, conversation_id, request_id, snapshot):
        """保存执行时召回快照；写入失败记录轮次日志，不阻断顾客回答。"""
        try:
            await self.store.record_retrieval_snapshot(conversation_id, request_id, snapshot)
        except Exception:
            logger.exception("召回采集失败 request=%s", request_id)

    async def record_low_confidence(
        self, conversation_id, request_id, question, entry_point, reason
    ):
        """幂等采集未解决原问；后台故障明确降级，不带入实时对话控制。"""
        try:
            await self.store.record_low_confidence(
                conversation_id, request_id, question, entry_point, reason
            )
        except Exception:
            logger.exception("问题采集失败 request=%s entry=%s", request_id, entry_point)
