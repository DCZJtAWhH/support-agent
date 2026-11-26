import asyncio

import pytest
from test_flywheel import Embeddings, Vectors

from support_agent.knowledge import KnowledgePublisher
from support_agent.knowledge.store import KnowledgeStore
from support_agent.platform.database import Database


def test_publication_retries_keep_original_content_and_identity_after_restart():
    async def scenario():
        database = Database("sqlite+aiosqlite:///:memory:")
        await database.initialize()
        store, vectors = KnowledgeStore(database), Vectors()
        try:
            vectors.fail = True
            publisher = KnowledgePublisher(store, Embeddings(), vectors)
            with pytest.raises(RuntimeError):
                await publisher.publish_answer("review:77", "杯盖如何清洗？", "请用温水手洗。")
            original = (await store.active())[0]
            assert original.vectorize_status == "pending"
            publisher = KnowledgePublisher(store, Embeddings(), vectors)
            vectors.fail = False
            with pytest.raises(ValueError, match="不同"):
                await publisher.publish_answer("review:77", "杯盖如何清洗？", "改成其他答案。")
            assert (await store.active())[0].answer == "请用温水手洗。"
            assert vectors.rows == {}
            for _ in range(2):
                await publisher.publish_answer("review:77", "杯盖如何清洗？", "请用温水手洗。")
            saved = await store.active()
            assert len(saved) == 1 and saved[0].id == original.id
            assert saved[0].vectorize_status == "done"
        finally:
            await database.close()

    asyncio.run(scenario())
