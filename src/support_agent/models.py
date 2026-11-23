"""定义常见问题、客服会话、消息流水、工单、退款申请及低置信问题业务表。"""

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """生成统一存储为 UTC 的无时区时间，兼容 MySQL DATETIME。"""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """提供客服业务表与知识库表共享的 SQLAlchemy 元数据。"""


class FAQ(Base):
    """维护商家购物及售后常见问题的标准答复。"""

    __tablename__ = "faq"  # 常见问题表。

    # 常见问题编号，自增主键。
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 常见问题文本，内容唯一。
    question: Mapped[str] = mapped_column(String(255), unique=True)
    # 商家维护的标准答复。
    answer: Mapped[str] = mapped_column(Text)
    # 常见问题分类，例如售后、配送。
    category: Mapped[str] = mapped_column(String(50))


class Conversation(Base):
    """保存客服会话所属顾客、处理状态和创建时间。"""

    __tablename__ = "conversations"  # 客服会话表。

    # 客服会话唯一标识，使用 UUID 字符串。
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # 顾客标识，本章固定使用演示用户 demo-user，不代表已完成身份认证。
    user_id: Mapped[str] = mapped_column(String(64), default="demo-user")
    # 会话处理状态：active 对话中；旧版 waiting_human 值保留，新建工单不再转接会话。
    status: Mapped[str] = mapped_column(String(32), default="active")
    # 会话创建时间，按 UTC 存储。
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class Message(Base):
    """保存用户发言、客服回复及工具申请和结果，保留失败轮次供审计。"""

    __tablename__ = "messages"  # 消息流水表。
    # 同一请求的同一消息位置只保存一条记录，重试时更新原记录。
    __table_args__ = (UniqueConstraint("request_id", "position"),)

    # 消息流水编号，自增主键。
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 所属客服会话标识，外键关联 conversations.id。
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    # 对话轮次请求标识，使用 UUID 字符串；同轮失败重试复用该值。
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    # 新版位置：0 用户，1 Workflow 快照，2起每步预留10位保存工具链，1000000最终回答。
    position: Mapped[int] = mapped_column(Integer)
    # 消息角色：user 顾客，assistant 模型客服，tool 工具，workflow 路由与建议审计。
    role: Mapped[str] = mapped_column(String(16))
    # 消息文本；工具结果为 JSON 文本，失败回答可能不完整。
    content: Mapped[str] = mapped_column(Text, default="")
    # 模型工具申请数组，包含工具名、参数和调用标识；非申请消息为空。
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # 工具结果对应的申请标识，匹配 tool_calls 中的 id；其他消息为空。
    tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # 消息保存/生成状态：running 执行中，completed 完成，failed 失败。
    # 用户消息的状态代表整轮状态；工具业务是否成功由 content JSON 中的 ok 表示。
    state: Mapped[str] = mapped_column(String(16), default="running")
    # 消息首次写入时间，按 UTC 存储；重试更新原消息时不改变此时间。
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


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
    # 工单处理状态：pending 待处理；本章未提供后续处理流转。
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # 工单创建时间，按 UTC 存储。
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class LowConfidenceQuestion(Base):
    """保存无法用现有证据回答的问题；同请求重放最多保留一条。"""

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True)
    question: Mapped[str] = mapped_column(Text)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    entry_point: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text)
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
