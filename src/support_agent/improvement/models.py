from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from support_agent.platform.models import Base, utc_now


class LowConfidenceQuestion(Base):
    """保存无法用现有证据回答的问题；同请求重放最多保留一条。"""

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    question: Mapped[str] = mapped_column(Text)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    entry_point: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text)
    retrieval_snapshot: Mapped[list | None] = mapped_column(JSON, nullable=True)
    matched_review_id: Mapped[int | None] = mapped_column(
        ForeignKey("review_queue.id"), nullable=True, index=True
    )
    entry_points: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class TurnRetrieval(Base):
    """按轮次保存过滤前的召回快照；存在但为空表示检索过却未召回。"""

    __tablename__ = "turn_retrievals"
    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    snapshot: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class ReviewQueue(Base):
    """一行代表一个去重知识缺口，核准答案由人工填写后按稳定来源发布。"""

    __tablename__ = "review_queue"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    normalized_question: Mapped[str] = mapped_column(Text)
    example_answer: Mapped[str] = mapped_column(Text)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class AnswerFeedback(Base):
    """按轮次持久化满意度选择，重复点击返回原选择，不重复落池。"""

    __tablename__ = "answer_feedback"
    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    value: Mapped[str] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
