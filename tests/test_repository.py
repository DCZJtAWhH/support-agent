import asyncio
from uuid import uuid4

from support_agent.database import Database
from support_agent.repositories import ChatRepository


def test_faq_literal_keywords_find_return_policy_but_miss_postage(tmp_path):
    async def scenario():
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'faq.db'}")
        try:
            await db.initialize(seed=True)
            repo = ChatRepository(db)
            matches = await repo.query_faq("退货")
            assert matches[0]["question"] == "退货政策是什么"
            assert "7天" in matches[0]["answer"]
            assert await repo.query_faq("邮费") == []
            assert await repo.query_faq("%") == []
            await db.initialize(seed=True)
            assert len(await repo.query_faq("退货")) == 1
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
