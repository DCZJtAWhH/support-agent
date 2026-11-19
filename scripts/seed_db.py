"""建立四张表并写入可重复执行的 FAQ 和会话、消息、工单样例。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import Settings  # noqa: E402
from support_agent.database import Database  # noqa: E402
from support_agent.repositories import ChatRepository  # noqa: E402

DEMO_CONVERSATION = "00000000-0000-4000-8000-000000000002"
DEMO_REQUEST = "00000000-0000-4000-8000-000000000003"


async def main():
    """只补充缺失样例，不清空或覆盖已有客服记录，完成后释放连接池。"""
    db = Database(Settings().database_url.get_secret_value())
    try:
        await db.initialize(seed=True)
        repo = ChatRepository(db)
        if not await repo.resolve_request(DEMO_REQUEST):
            await repo.begin_request(DEMO_CONVERSATION, DEMO_REQUEST, "演示：商品损坏，请人工处理")
            ticket = await repo.create_ticket(
                DEMO_CONVERSATION, DEMO_REQUEST, "演示商品损坏", "维修"
            )
            await repo.save_message(
                DEMO_CONVERSATION,
                DEMO_REQUEST,
                4,
                "assistant",
                "演示工单已登记：" + ticket["ticket_no"],
            )
            await repo.set_request_state(DEMO_REQUEST, "completed")
        print("四张表及演示数据已就绪；样例会话：" + DEMO_CONVERSATION)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
