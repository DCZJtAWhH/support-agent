"""管理 SQLAlchemy 异步连接与数据库初始化，不在模型生成时持有事务。"""

import json
from functools import partial

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from support_agent.models import FAQ, Base

FAQ_SEEDS = [
    (
        "退货政策是什么",
        "签收后7天内，商品完好且不影响二次销售可申请无理由退货。质量问题请联系人工核实。",
        "售后",
    ),
    (
        "运费标准是什么",
        "普通商品订单满99元免运费，未满99元收取8元运费；偏远地区需人工确认。",
        "配送",
    ),
    ("如何申请换货", "请提供订单号和商品问题，客服登记后由人工确认换货条件。", "售后"),
]


class Database:
    def __init__(self, url: str):
        """创建异步引擎；生产使用 mysql+aiomysql，测试可注入独立 SQLite 地址。"""
        self.engine = create_async_engine(
            url,
            pool_pre_ping=True,
            json_serializer=partial(json.dumps, ensure_ascii=False),
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self, *, seed: bool = False):
        """建立业务表并增量补齐既有知识表，按需初始化 FAQ 演示数据。"""
        from support_agent.knowledge_store import migrate_knowledge_schema
        from support_agent.summaries import migrate_context_schema

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.run_sync(migrate_knowledge_schema)
            await connection.run_sync(migrate_context_schema)
        if seed:
            async with self.sessions.begin() as session:
                for question, answer, category in FAQ_SEEDS:
                    if not await session.scalar(select(FAQ).where(FAQ.question == question)):
                        session.add(FAQ(question=question, answer=answer, category=category))

    async def close(self):
        """释放连接池，供应用退出和测试清理调用。"""
        await self.engine.dispose()
