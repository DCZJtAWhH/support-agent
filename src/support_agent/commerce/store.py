from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .models import RefundApplication, Ticket
from .orders import get_order
from .schemas import REFUND_REASONS


class CommerceService:
    def __init__(self, database):
        """绑定本模块存储，每次操作使用独立短事务。"""
        self.database = database

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

    async def create_refund(
        self, conversation_id, request_id, order_id, reason, *, confirmed=False
    ):
        """按原请求唯一保存用户选择的原因，重复点击返回原单，变更内容明确拒绝。"""
        if confirmed is not True:
            raise ValueError("退款申请必须经过顾客显式确认。")
        if reason not in REFUND_REASONS or get_order(order_id) is None:
            raise ValueError("退款原因或已核实订单无效。")
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
        self,
        conversation_id: str,
        request_id: str,
        description: str,
        ticket_type: str,
        *,
        confirmed: bool = False,
    ) -> dict:
        """显式确认后按请求创建工单；相同输入安全重放，更改输入报冲突。"""
        if confirmed is not True:
            raise ValueError("人工工单必须经过顾客显式确认。")
        if not description.strip() or ticket_type not in {
            "退款",
            "退货退款",
            "换货",
            "维修",
            "其他",
            "未知",
        }:
            raise ValueError("工单描述或诉求类型无效。")
        try:
            async with self.database.sessions.begin() as session:
                existing = await session.scalar(
                    select(Ticket).where(Ticket.request_id == request_id)
                )
                if existing:
                    self.check_ticket_input(existing, conversation_id, description, ticket_type)
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
                    self.check_ticket_input(existing, conversation_id, description, ticket_type)
                    return self.ticket_data(existing)
            raise

    @staticmethod
    def check_ticket_input(row, conversation_id, description, ticket_type):
        """比较业务幂等输入，避免稳定请求标识被用于另一笔写入。"""
        if (row.conversation_id, row.description, row.ticket_type) != (
            conversation_id,
            description,
            ticket_type,
        ):
            raise ValueError("该请求已经创建了不同内容的人工工单。")

    async def read_operations(self, conversation_id):
        """读取本会话工单及退款凭据，供入口组合审计，不返回 ORM 对象。"""
        async with self.database.sessions() as session:
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
                "tickets": [self.ticket_data(row) for row in tickets],
                "refunds": [self.refund_data(row) for row in refunds],
            }
