import asyncio

import pytest

from support_agent.knowledge.chunks import KnowledgeDraft
from support_agent.knowledge.store import KnowledgeStore, synchronize
from support_agent.platform.database import Database


class DemoEmbedding:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class DemoVectors:
    def __init__(self):
        self.rows = {}

    async def upsert(self, rows):
        self.rows.update({r["id"]: r for r in rows})

    async def delete(self, ids):
        for key in ids:
            self.rows.pop(key, None)

    async def search(self, vector, limit):
        return [(r["id"], 0.9, r["content_hash"]) for r in self.rows.values()][:limit]


def test_interrupted_vector_upsert_is_resumed_by_same_primary_key_and_source_is_idempotent(
    tmp_path,
):
    async def scenario():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'knowledge.db'}")
        try:
            await db.initialize()
            store, vectors = KnowledgeStore(db), DemoVectors()
            draft = KnowledgeDraft("配送", ["运费标准是什么"], "满99元免运费，否则8元。")
            first = await store.replace_source("faq:1", [draft])
            assert await store.replace_source("faq:1", [draft]) == first
            with pytest.raises(RuntimeError, match="模拟中断"):
                await synchronize(store, DemoEmbedding(), vectors, interrupt_after_upsert=True)
            assert len(vectors.rows) == 1
            assert (await store.stats())["pending"] == 1
            assert await synchronize(store, DemoEmbedding(), vectors) == 1
            assert (await store.stats())["pending"] == 0
            assert len(vectors.rows) == 1
            assert (await store.get(first[0])).vector_id == str(first[0])
            assert await synchronize(store, DemoEmbedding(), vectors) == 0
        finally:
            await db.close()

    asyncio.run(scenario())
