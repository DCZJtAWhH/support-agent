from sqlalchemy import select

from .models import ToolAuditLog


class ToolAuditStore:
    def __init__(self, database):
        """绑定本模块存储，每次操作使用独立短事务。"""
        self.database = database

    async def record_tool_audit(self, **values):
        """以独立短事务追加调用终态；失败交由执行引擎降级，不回滚业务写入。"""
        async with self.database.sessions.begin() as session:
            session.add(ToolAuditLog(**values))

    async def tool_audit_logs(self, conversation_id):
        """顺序读取会话调用审计，供验收及排障；不要求会话业务记录仍然存在。"""
        async with self.database.sessions() as session:
            rows = await session.scalars(
                select(ToolAuditLog)
                .where(ToolAuditLog.conversation_id == conversation_id)
                .order_by(ToolAuditLog.id)
            )
            return [
                {
                    **{c.name: getattr(row, c.name) for c in ToolAuditLog.__table__.columns},
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]
