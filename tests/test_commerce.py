import asyncio

import pytest

from support_agent.commerce import CommerceService
from support_agent.conversation import ConversationService
from support_agent.platform.database import Database


def test_ticket_public_entry_requires_confirmation_and_rejects_changed_retry():
    async def scenario():
        database = Database("sqlite+aiosqlite:///:memory:")
        await database.initialize()
        commerce = CommerceService(database)
        try:
            await ConversationService(database).begin_request("conversation", "request", "商品损坏")
            with pytest.raises(ValueError, match="确认"):
                await commerce.create_ticket("conversation", "request", "商品损坏", "维修")
            assert (await commerce.read_operations("conversation"))["tickets"] == []
            first = await commerce.create_ticket(
                "conversation", "request", "商品损坏", "维修", confirmed=True
            )
            assert (
                await commerce.create_ticket(
                    "conversation", "request", "商品损坏", "维修", confirmed=True
                )
                == first
            )
            with pytest.raises(ValueError, match="不同"):
                await commerce.create_ticket(
                    "conversation", "request", "更换了问题", "退款", confirmed=True
                )
            assert (await commerce.read_operations("conversation"))["tickets"] == [first]
        finally:
            await database.close()

    asyncio.run(scenario())
