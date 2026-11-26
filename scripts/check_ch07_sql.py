"""在当前 MySQL 数据库的独立空表上验证 ch7.sql，收尾只删除本脚本创建的验证表。"""

import asyncio
import re
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import text  # noqa: E402

from support_agent.platform.config import Settings  # noqa: E402
from support_agent.platform.database import Database  # noqa: E402


async def main():
    """克隆无数据会话表并执行迁移原文，验证列与外键兼容，不触碰真实会话。"""
    config = Settings()
    database = Database(config.database_url.get_secret_value())
    suffix = uuid4().hex[:12]
    conversations = f"ch07_check_conversations_{suffix}"
    summaries = f"ch07_check_summaries_{suffix}"
    assert conversations.startswith("ch07_check_") and summaries.startswith("ch07_check_")
    try:
        async with database.engine.begin() as connection:
            await connection.execute(text(f"CREATE TABLE {conversations} LIKE conversations"))
            for column in ("summary_upto_msg_id", "layer1_from_msg_id"):
                await connection.execute(text(f"ALTER TABLE {conversations} DROP COLUMN {column}"))
            sql = (ROOT / "ch7.sql").read_text(encoding="utf-8-sig")
            sql = re.sub(r"--[^\n]*", "", sql)
            sql = re.sub(r"\bconversations\b", conversations, sql)
            sql = re.sub(r"\bconversation_summaries\b", summaries, sql)
            sql = sql.replace("fk_summary_conversation", f"fk_ch07_check_{suffix}")
            for statement in sql.split(";"):
                if statement.strip():
                    await connection.execute(text(statement))
            await connection.execute(
                text(
                    f"INSERT INTO {conversations} (id, user_id, status, created_at) "
                    "VALUES ('sql-check','demo-user','active',NOW())"
                )
            )
            await connection.execute(
                text(
                    f"INSERT INTO {summaries} "
                    "(conversation_id,seq,from_msg_id,upto_msg_id,content) "
                    "VALUES ('sql-check',1,1,2,'订单1001换货待核实')"
                )
            )
            assert await connection.scalar(text(f"SELECT COUNT(*) FROM {summaries}")) == 1
            print("ch7.sql: ALTER / CREATE / UUID外键 / 中文摘要写入验证通过")
    finally:
        async with database.engine.begin() as connection:
            await connection.execute(text(f"DROP TABLE IF EXISTS {summaries}"))
            await connection.execute(text(f"DROP TABLE IF EXISTS {conversations}"))
        await database.close()


if __name__ == "__main__":
    asyncio.run(main())
