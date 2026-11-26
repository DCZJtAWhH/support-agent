import json
from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import func, or_, select, update

from support_agent.platform.capabilities import CapabilityError

from .models import Conversation, ConversationSummary, Message


@dataclass(frozen=True)
class AcceptedRequest:
    """保存请求身份、既有轮次及重放结果；不包含图或数据库私有状态。"""

    conversation_id: str
    request_id: str
    original: tuple[str, str] | None
    audit: dict | None
    metadata: dict
    final: dict | None


class ConversationService:
    def __init__(self, database):
        """绑定本模块存储，每次操作使用独立短事务。"""
        self.database = database
        self._busy: set[str] = set()
        self._requests: set[str] = set()

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
                row = Message(
                    conversation_id=conversation_id,
                    request_id=request_id,
                    position=0,
                    role="user",
                    content=content,
                )
                session.add(row)
            elif row.state != "completed":
                row.state = "running"
            await session.flush()
            return row.id

    async def read_conversation(self, conversation_id: str) -> dict | None:
        """返回本模块会话、含失败轮次的完整原文与摘要；未知会话返回 None。"""
        async with self.database.sessions() as session:
            conversation = await session.get(Conversation, conversation_id)
            if not conversation or conversation.user_id != "demo-user":
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
            return {
                "id": conversation.id,
                "user_id": conversation.user_id,
                "status": conversation.status,
                "created_at": conversation.created_at.isoformat(),
                "summary_upto_msg_id": conversation.summary_upto_msg_id,
                "layer1_from_msg_id": conversation.layer1_from_msg_id,
                "summaries": [
                    {
                        "seq": s.seq,
                        "from_msg_id": s.from_msg_id,
                        "upto_msg_id": s.upto_msg_id,
                        "content": s.content,
                    }
                    for s in await session.scalars(
                        select(ConversationSummary)
                        .where(
                            ConversationSummary.conversation_id == conversation_id,
                        )
                        .order_by(ConversationSummary.seq)
                    )
                ],
                "messages": [self.message_data(row) for row in messages],
            }

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
            await session.flush()
            return row.id

    async def move_layer1(self, conversation_id, from_id):
        """以条件更新单调前移原文层起点，不删除历史或修改消息内容。"""
        async with self.database.sessions.begin() as session:
            await session.execute(
                update(Conversation)
                .where(
                    Conversation.id == conversation_id,
                    or_(
                        Conversation.layer1_from_msg_id.is_(None),
                        Conversation.layer1_from_msg_id < from_id,
                    ),
                )
                .values(layer1_from_msg_id=from_id)
            )

    async def list_conversations(self, user_id="demo-user"):
        """按创建时间倒序列出当前演示用户的全部会话、首问预览及摘要状态。"""
        async with self.database.sessions() as session:
            preview = (
                select(Message.content)
                .where(
                    Message.conversation_id == Conversation.id,
                    Message.position == 0,
                )
                .order_by(Message.id)
                .limit(1)
                .correlate(Conversation)
                .scalar_subquery()
            )
            rows = await session.execute(
                select(Conversation, preview)
                .where(
                    Conversation.user_id == user_id,
                )
                .order_by(Conversation.created_at.desc(), Conversation.id.desc())
            )
            return [
                {
                    "id": c.id,
                    "created_at": c.created_at.isoformat(),
                    "preview": (question or "新对话")[:60],
                    "has_summary": c.summary_upto_msg_id is not None,
                }
                for c, question in rows
            ]

    async def append_summary(self, conversation_id, from_id, upto_id, content, *, expected_upto):
        """在同一事务比较旧锚点、追加摘要并推进覆盖边界；过期任务返回 None。

        模型调用必须在此方法之外执行，避免持有数据库事务等待网络。
        """
        if not content.strip() or from_id > upto_id or from_id <= (expected_upto or 0):
            raise ValueError("摘要内容及覆盖边界无效。")
        async with self.database.sessions.begin() as session:
            changed = await session.execute(
                update(Conversation)
                .where(
                    Conversation.id == conversation_id,
                    Conversation.summary_upto_msg_id == expected_upto,
                    Conversation.layer1_from_msg_id > upto_id,
                )
                .values(summary_upto_msg_id=upto_id)
            )
            if not changed.rowcount:
                return None
            seq = (
                await session.scalar(
                    select(func.max(ConversationSummary.seq)).where(
                        ConversationSummary.conversation_id == conversation_id,
                    )
                )
                or 0
            ) + 1
            session.add(
                ConversationSummary(
                    conversation_id=conversation_id,
                    seq=seq,
                    from_msg_id=from_id,
                    upto_msg_id=upto_id,
                    content=content.strip(),
                )
            )
            return {
                "seq": seq,
                "from_msg_id": from_id,
                "upto_msg_id": upto_id,
                "content": content.strip(),
            }

    async def set_request_state(self, request_id: str, state: str):
        """更新用户消息上的轮次状态；工具检查点保留自己的执行状态。"""
        async with self.database.sessions.begin() as session:
            await session.execute(
                update(Message)
                .where(Message.request_id == request_id, Message.position == 0)
                .values(state=state)
            )

    async def read_turn(self, conversation_id, request_id):
        """读取指定会话的原轮次，含完成状态和原始审计；不存在时返回 None。"""
        audit = await self.read_conversation(conversation_id)
        rows = [m for m in (audit or {}).get("messages", []) if m["request_id"] == request_id]
        return rows or None

    async def accept_request(
        self, content, *, conversation_id=None, request_id=None, category=None
    ):
        """定位或恢复原请求并锁定会话；同标识不同输入报冲突，失败自动释放占用。

        成功后调用方必须在完整发送或取消后调用 release_request；本方法不提交回答。
        """
        rid = str(request_id or uuid4())
        original = await self.resolve_request(rid)
        cid = str(conversation_id or (original[0] if original else uuid4()))
        if original and original != (cid, content):
            raise CapabilityError("conflict", "请求标识已用于其他会话或问题。")
        if cid in self._busy or rid in self._requests:
            raise CapabilityError("conflict", "本次会话或请求正在执行，请等待结束后重试。")
        self._busy.add(cid)
        self._requests.add(rid)
        try:
            audit = await self.read_conversation(cid)
            if (conversation_id or original) and audit is None:
                raise CapabilityError("not_found", "会话不存在。")
            turn = [m for m in (audit or {}).get("messages", []) if m["request_id"] == rid]
            metadata = next((json.loads(m["content"]) for m in turn if m["role"] == "workflow"), {})
            if turn and metadata.get("category") != category:
                raise CapabilityError("conflict", "同一请求不能更改检索品类。")
            final = next(
                (m for m in turn if m["position"] == 1_000_000 and m["state"] == "completed"), None
            )
            return AcceptedRequest(cid, rid, original, audit, metadata, final)
        except BaseException:
            self.release_request(cid, rid)
            raise

    def release_request(self, conversation_id, request_id):
        """释放进程内会话与请求占用；持久状态和已写入的业务结果不回滚。"""
        self._busy.discard(conversation_id)
        self._requests.discard(request_id)

    def claim_action(self, conversation_id, *, busy_message):
        """在确认业务动作期间互斥会话；已占用时返回明确冲突，不暴露内部锁集合。"""
        if conversation_id in self._busy:
            raise CapabilityError("conflict", busy_message)
        self._busy.add(conversation_id)

    def release_action(self, conversation_id):
        """释放已完成或失败的业务确认占用，不改变轮次或已提交业务结果。"""
        self._busy.discard(conversation_id)
