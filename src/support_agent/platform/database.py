"""管理 SQLAlchemy 异步连接与数据库初始化，不在模型生成时持有事务。"""

import json
from functools import partial

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .models import Base

schema_migrations = []
seed_initializers = []


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
        """建立业务表并增量补齐知识、会话与问题池旧表，按需初始化 FAQ 演示数据。"""
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            for migrate in schema_migrations:
                await connection.run_sync(migrate)
        if seed:
            for initialize in seed_initializers:
                await initialize(self)

    async def close(self):
        """释放连接池，供应用退出和测试清理调用。"""
        await self.engine.dispose()
