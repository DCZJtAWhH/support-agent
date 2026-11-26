"""用短事务保存知识原文和来源身份，并重放尚未完成的向量写入。"""

import hashlib

from sqlalchemy import inspect, select, true, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateColumn

from support_agent.knowledge.chunks import KnowledgeDraft
from support_agent.knowledge.models import KnowledgeChunk, QAStaging
from support_agent.platform.database import Database
from support_agent.platform.models import Base

from .contracts import CorpusVersion


class SimulatedInterruption(RuntimeError):
    """表示演示专用的向量写后中断，已提交的原文仍保持 pending。"""


def digest(text: str) -> str:
    """计算稳定摘要，用于来源幂等、内容版本及批次身份。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def migrate_knowledge_schema(connection):
    """给已建表增量补充缺失列和索引；可重跑，不删除已有列或数据。"""
    for model in (KnowledgeChunk, QAStaging):
        table = Base.metadata.tables[model.__tablename__]
        existing = {c["name"] for c in inspect(connection).get_columns(table.name)}
        for column in table.columns:
            if column.name not in existing:
                definition = str(CreateColumn(column).compile(dialect=connection.dialect))
                connection.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {definition}")
        for index in table.indexes:
            index.create(connection, checkfirst=True)


def draft_from_row(row: KnowledgeChunk) -> KnowledgeDraft:
    """将数据库换行问法还原成知识草稿，复用唯一嵌入文本格式。"""
    return KnowledgeDraft(
        row.category,
        row.questions.splitlines(),
        row.answer,
        row.section_path or "",
        row.content_type or "policy",
        row.is_key_clause,
    )


class KnowledgeStore:
    def __init__(self, database: Database):
        """绑定数据库；所有网络向量调用均在数据库事务外进行。"""
        self.database = database

    async def save_publication(self, source: str, draft: KnowledgeDraft) -> int:
        """首次发布保存稳定来源；重试核对原输入，内容冲突不覆盖 pending 或已发布原文。

        只提交本模块短事务；向量同步由调用方在事务结束后执行。
        """
        if not source or len(source) > 255:
            raise ValueError("知识发布来源标识无效。")
        key = digest(f"{source}:0")
        content_hash = digest(draft.embedding_text())
        try:
            async with self.database.sessions.begin() as session:
                row = await session.scalar(
                    select(KnowledgeChunk).where(KnowledgeChunk.source_key == key).with_for_update()
                )
                if row is not None:
                    self.check_publication(row, source, content_hash)
                    return row.id
                row = KnowledgeChunk(
                    source_key=key,
                    source_ref=source,
                    category=draft.category,
                    questions="\n".join(draft.questions),
                    answer=draft.answer,
                    section_path=draft.section_path,
                    content_type=draft.content_type,
                    is_key_clause=draft.is_key_clause,
                    content_hash=content_hash,
                    is_active=True,
                    vectorize_status="pending",
                )
                session.add(row)
                await session.flush()
                return row.id
        except IntegrityError:
            # 首次发布并发相撞时只回读同一来源，不能用另一个身份重新写入。
            async with self.database.sessions() as session:
                row = await session.scalar(
                    select(KnowledgeChunk).where(KnowledgeChunk.source_key == key)
                )
                if row is None:
                    raise
                self.check_publication(row, source, content_hash)
                return row.id

    @staticmethod
    def check_publication(row: KnowledgeChunk, source: str, content_hash: str):
        """拒绝稳定发布身份上的不同核准输入，不影响普通知识来源的更新语义。"""
        if row.source_ref != source or row.content_hash != content_hash or not row.is_active:
            raise ValueError("该发布标识已用于不同的核准内容。")

    async def replace_source(self, source: str, drafts: list[KnowledgeDraft]) -> list[int]:
        """按来源和块序号幂等替换，内容变化转 pending，移除的块失效并待删除向量。"""
        if len(source) > 255:
            raise ValueError("知识来源标识不能超过255字。")
        async with self.database.sessions.begin() as session:
            old = list(
                await session.scalars(
                    select(KnowledgeChunk).where(KnowledgeChunk.source_ref == source),
                )
            )
            by_key = {r.source_key: r for r in old}
            rows = []
            for index, draft in enumerate(drafts):
                key = digest(f"{source}:{index}")
                row = by_key.get(key)
                if row is None:
                    row = KnowledgeChunk(source_key=key, source_ref=source)
                    session.add(row)
                questions = list(draft.questions)
                if row.answer == draft.answer and row.category == draft.category and row.questions:
                    questions = list(dict.fromkeys([*questions, *row.questions.splitlines()]))
                effective = KnowledgeDraft(
                    draft.category,
                    questions,
                    draft.answer,
                    draft.section_path,
                    draft.content_type,
                    draft.is_key_clause,
                )
                content_hash = digest(effective.embedding_text())
                if row.content_hash != content_hash or not row.is_active:
                    row.vectorize_status, row.vector_id = "pending", None
                row.category, row.questions, row.answer = (
                    draft.category,
                    "\n".join(questions),
                    draft.answer,
                )
                row.section_path, row.content_type = draft.section_path, draft.content_type
                row.is_key_clause, row.is_active, row.content_hash = (
                    draft.is_key_clause,
                    True,
                    content_hash,
                )
                rows.append(row)
            await session.flush()
            ids = [row.id for row in rows]
            for index, row in enumerate(rows):
                row.prev_chunk_id = ids[index - 1] if index else None
                row.next_chunk_id = ids[index + 1] if index + 1 < len(ids) else None
            for row in old:
                if row.id not in ids and row.is_active:
                    row.is_active, row.vectorize_status = False, "pending"
                    row.prev_chunk_id = row.next_chunk_id = None
            return ids

    async def get(self, chunk_id: int) -> KnowledgeChunk | None:
        """读取权威知识与状态，供检索回查和命令检查。"""
        async with self.database.sessions() as session:
            return await session.get(KnowledgeChunk, chunk_id)

    async def active(self) -> list[KnowledgeChunk]:
        """返回全部生效知识，供 demo 的整体离线去重使用。"""
        async with self.database.sessions() as session:
            return list(
                await session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.is_active.is_(True))
                    .order_by(KnowledgeChunk.id),
                )
            )

    async def pending(
        self, limit: int = 16, *, source_ref: str | None = None
    ) -> list[KnowledgeChunk]:
        """按主键扫描未完成块；上传可限定来源，包含待删除向量的失效记录。"""
        async with self.database.sessions() as session:
            return list(
                await session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.vectorize_status == "pending")
                    .where(
                        KnowledgeChunk.source_ref == source_ref
                        if source_ref is not None
                        else true()
                    )
                    .order_by(KnowledgeChunk.id)
                    .limit(limit),
                )
            )

    async def mark_done(self, row: KnowledgeChunk):
        """仅当原文摘要和生效状态未变化时回填，避免旧任务覆盖新版本。"""
        async with self.database.sessions.begin() as session:
            await session.execute(
                update(KnowledgeChunk)
                .where(
                    KnowledgeChunk.id == row.id,
                    KnowledgeChunk.content_hash == row.content_hash,
                    KnowledgeChunk.is_active == row.is_active,
                )
                .values(vectorize_status="done", vector_id=str(row.id) if row.is_active else None)
            )

    async def stats(self) -> dict:
        """返回状态统计，便于演示中断前后漏向量块的补齐。"""
        async with self.database.sessions() as session:
            rows = list(await session.scalars(select(KnowledgeChunk)))
            return {
                "total": len(rows),
                "active": sum(r.is_active for r in rows),
                "pending": sum(r.vectorize_status == "pending" for r in rows),
                "done": sum(r.vectorize_status == "done" for r in rows),
            }

    async def requeue(self):
        """将生效知识重新标为 pending，供显式重建向量与可重复的中断演示。"""
        async with self.database.sessions.begin() as session:
            await session.execute(
                update(KnowledgeChunk)
                .where(
                    KnowledgeChunk.is_active.is_(True),
                )
                .values(vectorize_status="pending", vector_id=None)
            )

    async def corpus(self) -> CorpusVersion:
        """返回当前知识身份和待同步数量，供评估核对版本，不暴露 ORM 与数据库。"""
        rows = await self.active()
        return {
            "size": len(rows),
            "hash": digest("\n".join(f"{r.id}:{r.content_hash}" for r in rows)),
            "pending_count": sum(r.vectorize_status != "done" for r in rows),
        }


async def synchronize(
    store,
    embeddings,
    vectors,
    *,
    interrupt_after_upsert: bool = False,
    source_ref: str | None = None,
) -> int:
    """重放全部或指定来源 pending；先写 Milvus 再回填，允许注入双写窗口中断。"""
    count = 0
    options = {"source_ref": source_ref} if source_ref is not None else {}
    while rows := await store.pending(**options):
        active = [r for r in rows if r.is_active]
        if active:
            embedded = await embeddings.embed([draft_from_row(r).embedding_text() for r in active])
            await vectors.upsert(
                [
                    {
                        "id": row.id,
                        "vector": vector,
                        "content_hash": row.content_hash,
                        "text": draft_from_row(row).embedding_text(),
                        "category": row.category,
                    }
                    for row, vector in zip(active, embedded, strict=True)
                ]
            )
            if interrupt_after_upsert:
                raise SimulatedInterruption("模拟中断：Milvus已写入，MySQL尚未回填；请重跑 sync。")
        inactive = [r.id for r in rows if not r.is_active]
        if inactive:
            await vectors.delete(inactive)
        for row in rows:
            await store.mark_done(row)
            count += 1
    return count
