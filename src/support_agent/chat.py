"""连接 HTTP 流与持久化 Workflow，管理请求幂等、点选续跑及显式提交。"""

import json
import logging
from contextlib import aclosing
from uuid import uuid4

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from support_agent.agent import agent_tools
from support_agent.context import ContextBudgetExceeded, fit_messages
from support_agent.errors import model_error
from support_agent.schemas import REFUND_REASONS
from support_agent.streaming import ChatStreamResponse
from support_agent.tools import ToolRegistry, build_tools
from support_agent.workflow import AGENT_PROMPT, FINAL_POSITION, SupportWorkflow

logger = logging.getLogger(__name__)


def sse(event: str, **data) -> str:
    """编码一条 SSE 事件，中文和换行通过 JSON 安全传递。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def completed_history(rows):
    """恢复已完成轮次的消息，首次升级没有图检查点时保留旧会话上下文。"""
    completed = {r["request_id"] for r in rows if r["position"] == 0 and r["state"] == "completed"}
    messages = []
    for row in rows:
        if row["request_id"] not in completed:
            continue
        if row["role"] == "user":
            messages.append(HumanMessage(row["content"]))
        elif row["role"] == "tool":
            messages.append(ToolMessage(row["content"], tool_call_id=row["tool_call_id"]))
        elif row["role"] == "assistant":
            messages.append(AIMessage(row["content"], tool_calls=row["tool_calls"] or []))
    return messages


class ChatService:
    def __init__(self, model, repo, config, *, checkpointer):
        """绑定图及仓储；按单工作进程在整个响应发送期间互斥同会话请求。"""
        self.repo, self.config = repo, config
        self.workflow = SupportWorkflow(model, repo, config, checkpointer)
        self.busy: set[str] = set()
        self.requests: set[str] = set()

    async def respond(self, body):
        """校验归属和输入预算后执行图，已完成结果直接重放，不再次调用模型。"""
        rid = str(body.request_id or uuid4())
        original = await self.repo.resolve_request(rid)
        cid = str(body.conversation_id or (original[0] if original else uuid4()))
        if original and original != (cid, body.message):
            raise HTTPException(409, "请求标识已用于其他会话或问题。")
        if cid in self.busy or rid in self.requests:
            raise HTTPException(409, "本次会话或请求正在执行，请等待结束后重试。")
        self.busy.add(cid)
        self.requests.add(rid)
        try:
            audit = await self.repo.read_conversation(cid)
            if body.conversation_id and audit is None:
                raise HTTPException(404, "会话不存在。")
            turn = [m for m in (audit or {}).get("messages", []) if m["request_id"] == rid]
            metadata = next((json.loads(m["content"]) for m in turn if m["role"] == "workflow"), {})
            if turn and metadata.get("category") != body.category:
                raise HTTPException(409, "同一请求不能更改检索品类。")
            final = next(
                (m for m in turn if m["position"] == FINAL_POSITION and m["state"] == "completed"),
                None,
            )
            if body.selected_order_id:
                if not original:
                    raise HTTPException(409, "请先发起问题，再从订单卡片中选择。")
                snapshot = await self.workflow.graph.aget_state(
                    {"configurable": {"thread_id": cid}}
                )
                if final:
                    if metadata.get("order_id") != body.selected_order_id:
                        raise HTTPException(409, "已完成请求不能更换订单。")
                elif snapshot.values.get("request_id") != rid:
                    raise HTTPException(409, "该订单选择已过期，请重新描述问题。")
                elif snapshot.interrupts:
                    allowed = {o["order_id"] for o in snapshot.interrupts[0].value["orders"]}
                    if body.selected_order_id not in allowed:
                        raise HTTPException(409, "请选择本轮列表中的订单。")
                elif snapshot.values.get("order_id") != body.selected_order_id:
                    raise HTTPException(409, "该请求不能更换订单。")
            registry = ToolRegistry(build_tools(self.repo, cid, rid, body.message))
            schema_cost = len(
                json.dumps(
                    [convert_to_openai_tool(t) for t in agent_tools(registry)],
                    ensure_ascii=False,
                )
            )
            fit_messages(
                SystemMessage(AGENT_PROMPT),
                [],
                HumanMessage(body.message),
                self.config.input_token_budget - schema_cost,
            )
            await self.repo.begin_request(cid, rid, body.message)
            if not metadata:
                await self.repo.save_message(
                    cid, rid, 1, "workflow", json.dumps({"category": body.category})
                )
        except ContextBudgetExceeded as exc:
            self.busy.discard(cid)
            self.requests.discard(rid)
            raise HTTPException(413, str(exc)) from exc
        except BaseException:
            self.busy.discard(cid)
            self.requests.discard(rid)
            raise
        successful = False

        async def stream():
            """透传图中的节点、工具和文本事件；错误不进入后续完整对话历史。"""
            nonlocal successful
            yield sse("meta", conversation_id=cid, request_id=rid)
            parts = []
            try:
                if final:
                    if metadata.get("resolved_question"):
                        yield sse(
                            "understanding",
                            question=metadata["resolved_question"],
                            intent=metadata["intent"],
                            confidence=metadata["confidence"],
                        )
                    yield sse("sources", sources=metadata.get("sources", []))
                    yield sse("token", text=final["content"])
                    yield sse("actions", actions=metadata.get("actions", []))
                    if metadata.get("refund_available"):
                        yield sse(
                            "refund_form", order_id=metadata["order_id"], reasons=REFUND_REASONS
                        )
                else:
                    async with aclosing(
                        self.workflow.stream(
                            cid,
                            rid,
                            body.message,
                            category=body.category,
                            initial_history=completed_history((audit or {}).get("messages", [])),
                            selected_order_id=body.selected_order_id,
                        )
                    ) as events:
                        async for event in events:
                            data = dict(event)
                            name = data.pop("event")
                            if name == "token":
                                parts.append(data["text"])
                            yield sse(name, **data)
                successful = True
                yield sse("done", conversation_id=cid, request_id=rid)
            except Exception as exc:
                logger.exception("聊天流程失败 request=%s", rid)
                await self.repo.save_message(
                    cid, rid, FINAL_POSITION, "assistant", "".join(parts), state="failed"
                )
                yield sse("error", **model_error(exc))

        async def commit():
            """最后一帧完整发送后提交图历史与轮次状态，建议选项此后才可确认。"""
            if successful:
                completed = await self.workflow.commit(cid, rid)
                await self.repo.set_request_state(rid, "completed" if completed else "waiting")

        async def close():
            """断流标记失败但保留审计；无论数据库操作结果如何都释放会话锁。"""
            try:
                current = await self.repo.read_conversation(cid)
                user = next(
                    m for m in current["messages"] if m["request_id"] == rid and m["position"] == 0
                )
                if user["state"] not in {"completed", "waiting"}:
                    await self.repo.set_request_state(rid, "failed")
            finally:
                self.busy.discard(cid)
                self.requests.discard(rid)

        return ChatStreamResponse(stream(), commit, close)

    async def submit_refund(self, body):
        """校验完成轮次与服务端订单快照后幂等保存退款申请，不执行资金操作。"""
        cid, rid = str(body.conversation_id), str(body.request_id)
        original = await self.repo.resolve_request(rid)
        if not original or original[0] != cid:
            raise HTTPException(409, "请求不属于该会话。")
        if cid in self.busy:
            raise HTTPException(409, "请等待本轮回复完成。")
        self.busy.add(cid)
        try:
            audit = await self.repo.read_conversation(cid)
            turn = [m for m in audit["messages"] if m["request_id"] == rid]
            user = next(m for m in turn if m["position"] == 0)
            metadata = next((json.loads(m["content"]) for m in turn if m["role"] == "workflow"), {})
            if user["state"] != "completed" or not metadata.get("refund_available"):
                raise HTTPException(409, "请先完成该订单的政策核对。")
            return await self.repo.create_refund(cid, rid, metadata["order_id"], body.reason)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        finally:
            self.busy.discard(cid)

    async def confirm_ticket(self, body):
        """仅显式点击才能调用第二章建单工具，校验原轮次的归属、完成和建议权限。"""
        cid, rid = str(body.conversation_id), str(body.request_id)
        original = await self.repo.resolve_request(rid)
        if original is None or original[0] != cid:
            raise HTTPException(409, "请求不属于该会话。")
        if cid in self.busy:
            raise HTTPException(409, "请等待本轮回复完成后再确认。")
        self.busy.add(cid)
        try:
            audit = await self.repo.read_conversation(cid)
            turn = [m for m in audit["messages"] if m["request_id"] == rid]
            user = next(m for m in turn if m["position"] == 0)
            metadata = next((json.loads(m["content"]) for m in turn if m["role"] == "workflow"), {})
            if user["state"] != "completed" or "create_ticket" not in metadata.get("actions", []):
                raise HTTPException(409, "该轮次尚未完成或没有建单建议。")
            registry = ToolRegistry(
                build_tools(self.repo, cid, rid, original[1]),
                timeout=self.config.tool_timeout_seconds,
                retries=self.config.tool_max_retries,
            )
            result = await registry.execute(
                "create_ticket",
                {
                    "description": original[1][:2000],
                    "ticket_type": "未知",
                },
            )
            if not result["ok"]:
                raise HTTPException(503, result["error"]["message"])
            return result["data"]
        finally:
            self.busy.discard(cid)
