from sqlalchemy import inspect
from sqlalchemy.schema import CreateColumn

from support_agent.platform.models import Base

from .models import LowConfidenceQuestion


def migrate_flywheel_schema(connection):
    """只增补问题池缺失列及索引，兼容既有 MySQL 与测试 SQLite 数据。"""
    table = Base.metadata.tables[LowConfidenceQuestion.__tablename__]
    existing = {c["name"] for c in inspect(connection).get_columns(table.name)}
    for column in table.columns:
        if column.name not in existing:
            definition = str(CreateColumn(column).compile(dialect=connection.dialect))
            connection.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {definition}")
    for index in table.indexes:
        index.create(connection, checkfirst=True)
