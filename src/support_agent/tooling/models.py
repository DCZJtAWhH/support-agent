from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from support_agent.platform.models import Base, utc_now


class ToolAuditLog(Base):
    """独立保存每次工具调用的终态；不挂外键，业务记录缺失也能记录拒绝。"""

    __tablename__ = "tool_audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    tool_call_id: Mapped[str] = mapped_column(String(128), index=True)
    tool_name: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(255))
    arguments: Mapped[dict] = mapped_column(JSON)
    result_summary: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
