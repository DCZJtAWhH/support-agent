from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from support_agent.platform.models import Base, utc_now


class Ticket(Base):
    """记录需人工跟进的问题；创建工单不代表问题已解决或售后已获批准。"""

    __tablename__ = "tickets"  # 人工工单表。

    # 人工工单号，唯一主键，由 T 前缀和 UUID 十六进制串组成。
    ticket_no: Mapped[str] = mapped_column(String(40), primary_key=True)
    # 来源客服会话标识，外键关联 conversations.id。
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    # 建单请求标识，使用 UUID 字符串；唯一约束保证同轮重试最多创建一张工单。
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    # 需人工跟进的问题描述及顾客诉求。
    description: Mapped[str] = mapped_column(Text)
    # 工单诉求类型：退款、退货退款、换货、维修、其他或未知。
    ticket_type: Mapped[str] = mapped_column(String(32))
    # 工单处理状态：pending 待处理；尚未提供后续处理流转。
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # 工单创建时间，按 UTC 存储。
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class RefundApplication(Base):
    """保存用户显式提交的退款申请；pending 仅表示待核实，不代表已退款。"""

    __tablename__ = "refund_applications"
    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    application_no: Mapped[str] = mapped_column(String(40), unique=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    order_id: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
