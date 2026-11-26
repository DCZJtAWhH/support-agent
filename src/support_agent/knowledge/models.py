"""知识模块独立维护旧 FAQ、权威知识、候选问答和抽取批次表。"""

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.orm import Mapped, mapped_column

from support_agent.platform.models import Base, utc_now


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


ID_TYPE = BigInteger().with_variant(Integer, "sqlite").with_variant(BIGINT(unsigned=True), "mysql")


class KnowledgeChunk(Base):
    """保存权威原文；来源键保证重复导入稳定，摘要防止更新期间读取旧向量。"""

    __tablename__ = "knowledge_chunks"
    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    prev_chunk_id: Mapped[int | None] = mapped_column(
        ID_TYPE,
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        ID_TYPE,
        ForeignKey("knowledge_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done"),
        default="pending",
        server_default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)
    source_key: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    source_ref: Mapped[str | None] = mapped_column(String(255), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), default="", server_default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")


class QAStaging(Base):
    """保存抽取候选与去重结论；冲突仍为 extracted 并通过 reason 解释。"""

    __tablename__ = "qa_extraction_staging"
    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded"),
        default="extracted",
        server_default="extracted",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    category: Mapped[str] = mapped_column(String(255), default="其他", server_default="其他")
    extraction_key: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    knowledge_id: Mapped[int | None] = mapped_column(ID_TYPE, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class ExtractionBatch(Base):
    """已提交批次即为抽取检查点，输入摘要使定时重复扫描不重复调用模型。"""

    __tablename__ = "knowledge_extraction_batches"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
