from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from support_agent.platform.models import Base, utc_now


class Conversation(Base):
    """保存客服会话所属顾客、处理状态和创建时间。"""

    __tablename__ = "conversations"  # 客服会话表。

    # 客服会话唯一标识，使用 UUID 字符串。
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # 顾客标识，本章固定使用演示用户 demo-user，不代表已完成身份认证。
    user_id: Mapped[str] = mapped_column(String(64), default="demo-user")
    # 会话处理状态：active 对话中；旧版 waiting_human 值保留，新建工单不再转接会话。
    status: Mapped[str] = mapped_column(String(32), default="active")
    # 摘要覆盖终点（包含）和原文层起点（包含），只前移、不搬动消息。
    summary_upto_msg_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 会话创建时间，按 UTC 存储。
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class ConversationSummary(Base):
    """只追加保存每批历史的一次性事实梗概，不合并或重写旧段落。"""

    __tablename__ = "conversation_summaries"
    __table_args__ = (UniqueConstraint("conversation_id", "seq"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    from_msg_id: Mapped[int] = mapped_column(Integer)
    upto_msg_id: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)


class Message(Base):
    """保存用户及客服原文和工作流审计；旧工具行保留，新工具结果只存 checkpoint。"""

    __tablename__ = "messages"  # 消息流水表。
    # 同一请求的同一消息位置只保存一条记录，重试时更新原记录。
    __table_args__ = (UniqueConstraint("request_id", "position"),)

    # 消息流水编号，自增主键。
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 所属客服会话标识，外键关联 conversations.id。
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    # 对话轮次请求标识，使用 UUID 字符串；同轮失败重试复用该值。
    request_id: Mapped[str] = mapped_column(String(36), index=True)
    # 新版位置：0 用户，1 Workflow 快照，1000000最终回答；旧版中间工具位置保留。
    position: Mapped[int] = mapped_column(Integer)
    # 消息角色：user 顾客，assistant 客服，workflow 路由审计；tool 仅兼容旧数据。
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
