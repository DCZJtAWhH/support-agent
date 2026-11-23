"""通过短事务读写常见问题、客服对话及人工工单。"""

from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from support_agent.database import Database
from support_agent.models import (
    Conversation,
    LowConfidenceQuestion,
    Message,
    RefundApplication,
    Ticket,
)


class ChatRepository:
    def __init__(self, database: Database, retriever=None):
        """保存会话工厂和语义检索入口，每个持久化操作使用独立短事务。"""
        self.database = database
        self.retriever = retriever

    async def query_faq(self, keyword: str, *, category=None, prepared=False) -> list[dict]:
        """用完整问题检索；Workflow 已归一的问题跳过旧的同义词扩展。"""
        if self.retriever is None:
            raise RuntimeError("知识检索尚未配置，请先建库。")
        options = {"category": category} if category else {}
        if prepared:
            options["prepared"] = {"normalized": keyword, "retrieval_text": keyword}
        return await self.retriever.search(keyword, **options)

    async def query_policies(self, question, queries, *, category=None) -> list[dict]:
        """将即时扩写的全部查询交给检索侧合并，强制只使用政策或手册条款。"""
        if self.retriever is None:
            raise RuntimeError("政策检索尚未配置，请先建库。")
        return await self.retriever.search_many(question, queries, category=category)

    async def record_low_confidence(
        self, conversation_id, request_id, question, entry_point, reason
    ):
        """按请求幂等记录拒答原因，断流重试不重复入池。"""
        try:
            async with self.database.sessions.begin() as session:
                existing = await session.scalar(
                    select(LowConfidenceQuestion).where(
                        LowConfidenceQuestion.request_id == request_id,
                    )
                )
                if existing is None:
                    session.add(
                        LowConfidenceQuestion(
                            conversation_id=conversation_id,
                            request_id=request_id,
                            question=question,
                            entry_point=entry_point,
                            reason=reason,
                        )
                    )
        except IntegrityError:
            # 并发重复提交只忽略已存在的唯一请求，其他完整性错误继续向上报告。
            async with self.database.sessions() as session:
                if not await session.scalar(
                    select(LowConfidenceQuestion).where(
                        LowConfidenceQuestion.request_id == request_id,
                    )
                ):
                    raise

    async def low_confidence_questions(self) -> list[dict]:
        """按入池时间顺序返回问题，供离线复盘与验证使用。"""
        async with self.database.sessions() as session:
            rows = await session.scalars(
                select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id)
            )
            return [
                {
                    "id": r.id,
                    "request_id": r.request_id,
                    "question": r.question,
                    "conversation_id": r.conversation_id,
                    "entry_point": r.entry_point,
                    "reason": r.reason,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def resolve_request(self, request_id: str) -> tuple[str, str] | None:
        """根据稳定请求标识找到原会话和输入，让未收到 meta 的重试也能恢复。"""
        async with self.database.sessions() as session:
            row = await session.scalar(
                select(Message).where(Message.request_id == request_id, Message.position == 0)
            )
            return (row.conversation_id, row.content) if row else None

    async def begin_request(self, conversation_id: str, request_id: str, content: str):
        """保存用户输入或将失败请求恢复为执行中；同轮重试不重复插入用户消息。"""
        async with self.database.sessions.begin() as session:
            if not await session.get(Conversation, conversation_id):
                session.add(Conversation(id=conversation_id))
                await session.flush()
            row = await session.scalar(
                select(Message).where(Message.request_id == request_id, Message.position == 0)
            )
            if not row:
                session.add(
                    Message(
                        conversation_id=conversation_id,
                        request_id=request_id,
                        position=0,
                        role="user",
                        content=content,
                    )
                )
            elif row.state != "completed":
                row.state = "running"

    async def read_conversation(self, conversation_id: str) -> dict | None:
        """返回会话、含失败轮次的消息审计及工单列表；未知会话返回 None。"""
        async with self.database.sessions() as session:
            conversation = await session.get(Conversation, conversation_id)
            if not conversation:
                return None
            messages = list(
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.id)
                )
            )
            origins = {row.request_id: row.id for row in messages if row.position == 0}
            # 先按顾客发言顺序、再按轮次内位置排序，避免重试补写的结果落在回答之后。
            messages.sort(key=lambda row: (origins[row.request_id], row.position))
            tickets = await session.scalars(
                select(Ticket)
                .where(Ticket.conversation_id == conversation_id)
                .order_by(Ticket.created_at)
            )
            refunds = await session.scalars(
                select(RefundApplication)
                .where(RefundApplication.conversation_id == conversation_id)
                .order_by(RefundApplication.created_at)
            )
            return {
                "id": conversation.id,
                "user_id": conversation.user_id,
                "status": conversation.status,
                "created_at": conversation.created_at.isoformat(),
                "messages": [self.message_data(row) for row in messages],
                "tickets": [self.ticket_data(row) for row in tickets],
                "refunds": [self.refund_data(row) for row in refunds],
            }

    @staticmethod
    def refund_data(row):
        """返回可重放的申请凭据，不承诺退款已经执行。"""
        return {
            "application_no": row.application_no,
            "order_id": row.order_id,
            "reason": row.reason,
            "status": row.status,
            "created_at": row.created_at.isoformat(),
            "simulated": True,
        }

    async def create_refund(self, conversation_id, request_id, order_id, reason):
        """按原请求唯一保存用户选择的原因，重复点击返回原单，变更内容明确拒绝。"""
        try:
            async with self.database.sessions.begin() as session:
                row = await session.get(RefundApplication, request_id)
                if row is None:
                    row = RefundApplication(
                        request_id=request_id,
                        application_no="R" + uuid4().hex,
                        conversation_id=conversation_id,
                        order_id=order_id,
                        reason=reason,
                    )
                    session.add(row)
                    await session.flush()
                if (row.conversation_id, row.order_id, row.reason) != (
                    conversation_id,
                    order_id,
                    reason,
                ):
                    raise ValueError("该轮次已提交退款申请，不能更改订单或原因。")
                return self.refund_data(row)
        except IntegrityError:
            async with self.database.sessions() as session:
                row = await session.get(RefundApplication, request_id)
                if row is None:
                    raise
                if (row.conversation_id, row.order_id, row.reason) != (
                    conversation_id,
                    order_id,
                    reason,
                ):
                    raise ValueError("该轮次已提交不同的退款申请。") from None
                return self.refund_data(row)

    @staticmethod
    def message_data(row: Message) -> dict:
        """将流水转换为可序列化数据，包含工具关联和轮次完成状态。"""
        return {
            "id": row.id,
            "request_id": row.request_id,
            "position": row.position,
            "role": row.role,
            "content": row.content,
            "tool_calls": row.tool_calls,
            "tool_call_id": row.tool_call_id,
            "state": row.state,
            "created_at": row.created_at.isoformat(),
        }

    @staticmethod
    def ticket_data(row: Ticket) -> dict:
        """将人工工单转换为工具结果和审计接口共用的公开数据。"""
        return {
            "ticket_no": row.ticket_no,
            "conversation_id": row.conversation_id,
            "description": row.description,
            "ticket_type": row.ticket_type,
            "status": row.status,
            "created_at": row.created_at.isoformat(),
        }

    async def create_ticket(
        self, conversation_id: str, request_id: str, description: str, ticket_type: str
    ) -> dict:
        """每个请求最多创建一张工单；唯一约束保护超时重试和并发重放。"""
        try:
            async with self.database.sessions.begin() as session:
                existing = await session.scalar(
                    select(Ticket).where(Ticket.request_id == request_id)
                )
                if existing:
                    return self.ticket_data(existing)
                row = Ticket(
                    ticket_no="T" + uuid4().hex,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    description=description,
                    ticket_type=ticket_type,
                )
                session.add(row)
                await session.flush()
                result = self.ticket_data(row)
            return result
        except IntegrityError:
            # 提交已成功但结果丢失或同时重放时，读取唯一请求对应的原工单。
            async with self.database.sessions() as session:
                existing = await session.scalar(
                    select(Ticket).where(Ticket.request_id == request_id)
                )
                if existing:
                    return self.ticket_data(existing)
            raise

    async def save_message(
        self,
        conversation_id: str,
        request_id: str,
        position: int,
        role: str,
        content: str,
        *,
        tool_calls: list | None = None,
        tool_call_id: str | None = None,
        state: str = "completed",
    ):
        """按请求和固定位置保存检查点；重试更新同一条消息，不重复记录执行结果。"""
        async with self.database.sessions.begin() as session:
            row = await session.scalar(
                select(Message).where(
                    Message.request_id == request_id, Message.position == position
                )
            )
            if not row:
                row = Message(
                    conversation_id=conversation_id,
                    request_id=request_id,
                    position=position,
                    role=role,
                )
                session.add(row)
            row.content, row.tool_calls, row.tool_call_id = content, tool_calls, tool_call_id
            row.state = state

    async def set_request_state(self, request_id: str, state: str):
        """更新用户消息上的轮次状态；工具检查点保留自己的执行状态。"""
        async with self.database.sessions.begin() as session:
            await session.execute(
                update(Message)
                .where(Message.request_id == request_id, Message.position == 0)
                .values(state=state)
            )
