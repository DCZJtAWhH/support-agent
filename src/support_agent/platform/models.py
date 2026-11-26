"""共享元数据和通用观测数据；不包含功能模块的业务表。"""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """生成统一存储为 UTC 的无时区时间，兼容 MySQL DATETIME。"""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """提供客服业务表与知识库表共享的 SQLAlchemy 元数据。"""


class RequestIntent(Base):
    """保存请求的最终意图，识别前失败的调用仍可按未识别归集用量。"""

    __tablename__ = "request_intents"
    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    intent: Mapped[str] = mapped_column(String(32), default="未识别")
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class ModelUsage(Base):
    """逐模型调用记录提供方真实用量，未知用量单独标记，不用估算值冒充。"""

    __tablename__ = "model_usage"
    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    usage_known: Mapped[bool] = mapped_column(default=False)
    duration_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
