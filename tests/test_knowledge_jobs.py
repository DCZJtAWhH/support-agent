import asyncio
from datetime import datetime, timedelta, timezone

from support_agent.knowledge.jobs import import_documents, seconds_until_next_run
from support_agent.knowledge.store import KnowledgeStore
from support_agent.platform.database import Database


def test_document_scan_updates_same_source_and_deactivates_removed_files(tmp_path):
    async def scenario():
        docs = tmp_path / "docs"
        docs.mkdir()
        file = docs / "policy.md"
        file.write_text("# 配送\n## 运费\n满99元免运费。", encoding="utf-8")
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}")
        try:
            await db.initialize()
            store = KnowledgeStore(db)
            await import_documents(store, docs)
            original = (await store.active())[0]
            file.write_text("# 配送\n## 运费\n满199元免运费。", encoding="utf-8")
            await import_documents(store, docs)
            updated = (await store.active())[0]
            assert updated.id == original.id
            assert updated.answer == "满199元免运费。"
            file.unlink()
            await import_documents(store, docs)
            assert await store.active() == []
            assert (await store.get(original.id)).vectorize_status == "pending"
        finally:
            await db.close()

    asyncio.run(scenario())


def test_daily_schedule_targets_next_two_am_in_beijing():
    tz = timezone(timedelta(hours=8))
    assert seconds_until_next_run(datetime(2026, 11, 20, 1, 59, tzinfo=tz)) == 60
    assert seconds_until_next_run(datetime(2026, 11, 20, 2, 0, tzinfo=tz)) == 86400
