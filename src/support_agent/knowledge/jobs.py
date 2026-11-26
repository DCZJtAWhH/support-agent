"""提供文档扫描、FAQ 迁移和北京时间每日知识挖掘任务。"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from support_agent.knowledge.chunks import KnowledgeDraft, split_markdown
from support_agent.knowledge.models import FAQ, KnowledgeChunk
from support_agent.knowledge.store import digest, synchronize

BEIJING = timezone(timedelta(hours=8))


def scan_documents(directory: Path) -> tuple[str, dict]:
    """在线程中完整读取并解析文档，返回来源前缀与草稿，解析失败不会先改库。"""
    root = directory.resolve()
    if not root.is_dir():
        raise ValueError("知识文档目录不存在；为避免误删，终止扫描。")
    prefix = "markdown:" + digest(root.as_posix())[:12] + ":"
    parsed = {
        prefix + path.relative_to(root).as_posix(): split_markdown(
            path.read_text(encoding="utf-8"),
            title=path.stem,
        )
        for path in sorted(root.rglob("*.md"))
    }
    return prefix, parsed


async def import_documents(store, directory: Path) -> int:
    """扫描指定目录的 Markdown；先完整解析，成功后更新来源并失效已删除文件。"""
    prefix, parsed = await asyncio.to_thread(scan_documents, directory)
    for source, drafts in parsed.items():
        await store.replace_source(source, drafts)
    async with store.database.sessions() as session:
        sources = set(
            await session.scalars(
                select(KnowledgeChunk.source_ref).where(
                    KnowledgeChunk.source_ref.startswith(prefix),
                    KnowledgeChunk.is_active.is_(True),
                )
            )
        )
    for source in sources - parsed.keys():
        await store.replace_source(source, [])
    return sum(len(chunks) for chunks in parsed.values())


async def import_faq(store) -> int:
    """离线迁移旧 FAQ 至权威知识表；旧表只作为种子来源，不参与在线召回。"""
    async with store.database.sessions() as session:
        rows = list(await session.scalars(select(FAQ).order_by(FAQ.id)))
    for row in rows:
        await store.replace_source(
            f"faq:{row.id}",
            [
                KnowledgeDraft(
                    row.category,
                    [row.question],
                    row.answer,
                    content_type="faq",
                )
            ],
        )
    return len(rows)


def read_history(directory: Path) -> list[tuple[str, list[dict]]]:
    """在线程中读取 UTF-8 历史导出，使用根目录与相对路径区分同名文件。"""
    if not directory.is_dir():
        raise ValueError("历史对话目录不存在。")
    root = directory.resolve()
    exports = []
    for path in sorted(directory.rglob("*.jsonl")):
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        source = (
            "history:" + digest(root.as_posix())[:12] + ":" + path.relative_to(directory).as_posix()
        )
        exports.append((source, records))
    return exports


async def mine_directory(pipeline, directory: Path, *, retry_conflicts: bool = False) -> dict:
    """读取显式已审核 JSONL，完成全部抽取后再整体发布，失败由批次检查点恢复。"""
    extracted = 0
    for source, records in await asyncio.to_thread(read_history, directory):
        extracted += await pipeline.extract_records(records, source)
    return {"extracted": extracted, **await pipeline.publish(retry_conflicts=retry_conflicts)}


def seconds_until_next_run(now: datetime | None = None) -> float:
    """计算距离下次北京时间 02:00 的秒数，恰好在执行时刻调用则安排到次日。"""
    current = (now or datetime.now(BEIJING)).astimezone(BEIJING)
    target = current.replace(hour=2, minute=0, second=0, microsecond=0)
    if target <= current:
        target += timedelta(days=1)
    return (target - current).total_seconds()


async def schedule(pipeline, directory, vectors, *, run_now: bool = False):
    """在独立单进程每日挖掘并补齐向量；失败一分钟后重试，重跑复用已提交批次。"""
    due = run_now
    remaining = seconds_until_next_run()
    while True:
        if not due:
            delay = min(60, remaining)
            await asyncio.sleep(delay)
            remaining -= delay
            if remaining > 0:
                continue
        try:
            result = await mine_directory(pipeline, directory)
            result["synchronized"] = await synchronize(pipeline.store, pipeline.embeddings, vectors)
            print(json.dumps({"job": "knowledge-mining", **result}, ensure_ascii=False), flush=True)
            remaining, due = seconds_until_next_run(), False
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "job": "knowledge-mining",
                        "error": type(exc).__name__,
                        "message": "任务失败，60秒后按检查点重试。",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            remaining, due = 60, False
