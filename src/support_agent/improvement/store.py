from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .models import LowConfidenceQuestion, TurnRetrieval


class ImprovementStore:
    def __init__(self, database):
        """绑定本模块存储，每次操作使用独立短事务。"""
        self.database = database

    async def record_low_confidence(
        self, conversation_id, request_id, question, entry_point, reason
    ):
        """按请求幂等入池并回捞本轮快照；同轮多个入口保留来源但不重复计数。"""
        try:
            async with self.database.sessions.begin() as session:
                existing = await session.scalar(
                    select(LowConfidenceQuestion)
                    .where(
                        LowConfidenceQuestion.request_id == request_id,
                    )
                    .with_for_update()
                )
                evidence = await session.get(TurnRetrieval, request_id)
                if existing is None:
                    session.add(
                        LowConfidenceQuestion(
                            conversation_id=conversation_id,
                            request_id=request_id,
                            question=question,
                            entry_point=entry_point,
                            reason=reason,
                            retrieval_snapshot=evidence.snapshot if evidence else None,
                            entry_points=[entry_point],
                        )
                    )
                else:
                    existing.entry_points = list(
                        dict.fromkeys(
                            [
                                *(existing.entry_points or [existing.entry_point]),
                                entry_point,
                            ]
                        )
                    )
                    if existing.retrieval_snapshot is None and evidence:
                        existing.retrieval_snapshot = evidence.snapshot
        except IntegrityError:
            # 并发重复提交只忽略已存在的唯一请求，其他完整性错误继续向上报告。
            async with self.database.sessions() as session:
                if not await session.scalar(
                    select(LowConfidenceQuestion).where(
                        LowConfidenceQuestion.request_id == request_id,
                    )
                ):
                    raise

    async def record_retrieval_snapshot(self, conversation_id, request_id, snapshot):
        """在检索完成后立即保存该轮 Top 片段原文与得分，不依赖最终回答或检查点存续。"""
        try:
            async with self.database.sessions.begin() as session:
                if not await session.get(TurnRetrieval, request_id):
                    session.add(
                        TurnRetrieval(
                            conversation_id=conversation_id,
                            request_id=request_id,
                            snapshot=snapshot,
                        )
                    )
                pooled = await session.scalar(
                    select(LowConfidenceQuestion).where(
                        LowConfidenceQuestion.request_id == request_id,
                    )
                )
                if pooled and pooled.retrieval_snapshot is None:
                    pooled.retrieval_snapshot = snapshot
        except IntegrityError:
            async with self.database.sessions() as session:
                if not await session.get(TurnRetrieval, request_id):
                    raise

    async def low_confidence_questions(self) -> list[dict]:
        """按入池时间顺序返回问题，供离线复盘与验证使用。"""
        async with self.database.sessions() as session:
            rows = await session.scalars(
                select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id)
            )
            return [
                {
                    "id": r.id,
                    "request_id": r.request_id,
                    "question": r.question,
                    "conversation_id": r.conversation_id,
                    "entry_point": r.entry_point,
                    "reason": r.reason,
                    "retrieval_snapshot": r.retrieval_snapshot,
                    "matched_review_id": r.matched_review_id,
                    "entry_points": r.entry_points or [r.entry_point],
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]
