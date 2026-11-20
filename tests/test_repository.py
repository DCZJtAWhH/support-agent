import asyncio
from uuid import uuid4

from test_knowledge_store import DemoEmbedding, DemoVectors

from support_agent.database import Database
from support_agent.knowledge_chunks import KnowledgeDraft
from support_agent.knowledge_store import KnowledgeStore, synchronize
from support_agent.repositories import ChatRepository
from support_agent.vectors import KnowledgeRetriever


def test_faq_semantic_results_use_mysql_text_and_preserve_contract(tmp_path):
    async def scenario():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'faq.db'}")
        try:
            await db.initialize(seed=True)
            store, embeddings, vectors = KnowledgeStore(db), DemoEmbedding(), DemoVectors()
            ids = await store.replace_source(
                "faq:freight",
                [
                    KnowledgeDraft("配送", ["运费标准是什么"], "满99元免运费，否则8元。"),
                ],
            )
            await synchronize(store, embeddings, vectors)
            repo = ChatRepository(db, KnowledgeRetriever(store, embeddings, vectors))
            assert await repo.query_faq("邮费是多少") == [
                {
                    "question": "运费标准是什么",
                    "answer": "满99元免运费，否则8元。",
                    "category": "配送",
                }
            ]
            await store.replace_source(
                "faq:freight",
                [
                    KnowledgeDraft("配送", ["运费标准是什么"], "新政策：满199元免运费。"),
                ],
            )
            assert await repo.query_faq("邮费是多少") == []
            await synchronize(store, embeddings, vectors)
            assert "199元" in (await repo.query_faq("邮费是多少"))[0]["answer"]
            assert (await store.get(ids[0])).vector_id == str(ids[0])
            await store.replace_source("faq:freight", [])
            assert await repo.query_faq("邮费是多少") == []
            await synchronize(store, embeddings, vectors)
            assert vectors.rows == {}
        finally:
            await db.close()

    asyncio.run(scenario())


def test_ticket_retry_returns_original_and_conversation_survives_restart(tmp_path):
    async def scenario():
        url = f"sqlite+aiosqlite:///{tmp_path / 'tickets.db'}"
        conversation_id, request_id = str(uuid4()), str(uuid4())
        db = Database(url)
        await db.initialize()
        repo = ChatRepository(db)
        await repo.begin_request(conversation_id, request_id, "请人工处理商品损坏")
        first = await repo.create_ticket(conversation_id, request_id, "商品损坏", "维修")
        await db.close()
        db = Database(url)
        try:
            repo = ChatRepository(db)
            second = await repo.create_ticket(conversation_id, request_id, "商品损坏", "维修")
            assert second == first
            audit = await repo.read_conversation(conversation_id)
            assert audit["status"] == "waiting_human"
            assert len(audit["tickets"]) == 1
            assert audit["messages"][0]["content"] == "请人工处理商品损坏"
            assert await repo.resolve_request(request_id) == (conversation_id, "请人工处理商品损坏")
        finally:
            await db.close()

    asyncio.run(scenario())
