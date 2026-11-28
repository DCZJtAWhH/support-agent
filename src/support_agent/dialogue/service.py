"""连接 HTTP 流与持久化 Workflow，管理请求幂等、点选续跑及显式提交。"""

import json
import logging
from contextlib import aclosing

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from support_agent.commerce import REFUND_REASONS
from support_agent.conversation import completed_history
from support_agent.dialogue.agent import agent_tools
from support_agent.dialogue.streaming import ChatStreamResponse
from support_agent.dialogue.workflow import AGENT_PROMPT, FINAL_POSITION
from support_agent.platform.capabilities import CapabilityError
from support_agent.platform.context import ContextBudgetExceeded, estimate_tokens, fit_messages
from support_agent.platform.errors import model_error

logger = logging.getLogger(__name__)


def sse(event: str, **data) -> str:
    """编码一条 SSE 事件，中文和换行通过 JSON 安全传递。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatService:
    def __init__(self, workflow, conversation, commerce, config):
        """绑定图及仓储；按单工作进程在整个响应发送期间互斥同会话请求。"""
        self.repo, self.commerce, self.config = conversation, commerce, config
        self.workflow = workflow

    async def respond(self, body):
        """校验归属和输入预算后执行图，已完成结果直接重放，不再次调用模型。"""
        accepted = await self.repo.accept_request(
            body.message,
            conversation_id=body.conversation_id,
            request_id=body.request_id,
            category=body.category,
        )
        cid, rid = accepted.conversation_id, accepted.request_id
        original, audit = accepted.original, accepted.audit
        metadata, final = accepted.metadata, accepted.final
        try:
            if estimate_tokens(body.message, self.config) > self.config.max_user_input_tokens:
                raise ContextBudgetExceeded("当前输入超过 MAX_USER_INPUT_TOKENS，请缩短后重试。")
            if body.selected_order_id:
                if not original:
                    raise CapabilityError("conflict", "请先发起问题，再从订单卡片中选择。")
                snapshot = await self.workflow.graph.aget_state(
                    {"configurable": {"thread_id": cid}}
                )
                if final:
                    if metadata.get("order_id") != body.selected_order_id:
                        raise CapabilityError("conflict", "已完成请求不能更换订单。")
                elif snapshot.values.get("request_id") != rid:
                    raise CapabilityError("conflict", "该订单选择已过期，请重新描述问题。")
                elif snapshot.interrupts:
                    allowed = {
                        o["order_id"] for o in snapshot.interrupts[0].value.get("orders", [])
                    }
                    if body.selected_order_id not in allowed:
                        raise CapabilityError("conflict", "请选择本轮列表中的订单。")
                elif snapshot.values.get("order_id") != body.selected_order_id:
                    raise CapabilityError("conflict", "该请求不能更换订单。")
            if body.ticket_confirmation is not None:
                if not original or body.selected_order_id:
                    raise CapabilityError("conflict", "请先获取本轮工单预览。")
                snapshot = await self.workflow.graph.aget_state(
                    {"configurable": {"thread_id": cid}}
                )
                if final:
                    if (
                        metadata.get("ticket_resolution") is not body.ticket_confirmation
                        or metadata.get("ticket_resolution_call_id") != body.ticket_tool_call_id
                    ):
                        raise CapabilityError("conflict", "已处理的工单确认不能更改。")
                else:
                    if snapshot.values.get("request_id") != rid:
                        raise CapabilityError("conflict", "该工单预览已过期。")
                    if snapshot.interrupts:
                        preview = snapshot.interrupts[0].value
                        if (
                            preview.get("kind") != "ticket"
                            or preview.get("tool_call_id") != body.ticket_tool_call_id
                        ):
                            raise CapabilityError(
                                "conflict", "请确认当前工单预览，调用标识不匹配。"
                            )
                    elif (
                        snapshot.values.get("ticket_resolution") is not body.ticket_confirmation
                        or snapshot.values.get("ticket_resolution_call_id")
                        != body.ticket_tool_call_id
                    ):
                        raise CapabilityError("conflict", "该请求没有待确认的工单预览。")
            await self.workflow.catalog.refresh()
            registry = self.workflow.catalog.bind(cid, rid, body.message)
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
            self.repo.release_request(cid, rid)
            raise CapabilityError("input_budget", str(exc)) from exc
        except BaseException:
            self.repo.release_request(cid, rid)
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
                            ticket_confirmation=body.ticket_confirmation,
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
                self.repo.release_request(cid, rid)

        return ChatStreamResponse(stream(), commit, close)

    async def submit_refund(self, body):
        """校验完成轮次与服务端订单快照后幂等保存退款申请，不执行资金操作。"""
        cid, rid = str(body.conversation_id), str(body.request_id)
        original = await self.repo.resolve_request(rid)
        if not original or original[0] != cid:
            raise CapabilityError("conflict", "请求不属于该会话。")
        self.repo.claim_action(cid, busy_message="请等待本轮回复完成。")
        try:
            audit = await self.repo.read_conversation(cid)
            turn = [m for m in audit["messages"] if m["request_id"] == rid]
            user = next(m for m in turn if m["position"] == 0)
            metadata = next((json.loads(m["content"]) for m in turn if m["role"] == "workflow"), {})
            if user["state"] != "completed" or not metadata.get("refund_available"):
                raise CapabilityError("conflict", "请先完成该订单的政策核对。")
            return await self.commerce.create_refund(
                cid, rid, metadata["order_id"], body.reason, confirmed=True
            )
        except ValueError as exc:
            raise CapabilityError("conflict", str(exc)) from exc
        finally:
            self.repo.release_action(cid)

    async def confirm_ticket(self, body):
        """仅显式点击才能调用建单工具，校验原轮次的归属、完成和建议权限。"""
        cid, rid = str(body.conversation_id), str(body.request_id)
        original = await self.repo.resolve_request(rid)
        if original is None or original[0] != cid:
            raise CapabilityError("conflict", "请求不属于该会话。")
        self.repo.claim_action(cid, busy_message="请等待本轮回复完成后再确认。")
        try:
            audit = await self.repo.read_conversation(cid)
            turn = [m for m in audit["messages"] if m["request_id"] == rid]
            user = next(m for m in turn if m["position"] == 0)
            metadata = next((json.loads(m["content"]) for m in turn if m["role"] == "workflow"), {})
            if user["state"] != "completed" or "create_ticket" not in metadata.get("actions", []):
                raise CapabilityError("conflict", "该轮次尚未完成或没有建单建议。")
            registry = self.workflow.catalog.bind(cid, rid, original[1])
            result = await registry.execute(
                "create_ticket",
                {
                    "description": original[1][:2000],
                    "ticket_type": "未知",
                },
                confirmed=True,
                tool_call_id=f"button:{rid}",
            )
            if not result["ok"]:
                raise CapabilityError("unavailable", result["error"]["message"])
            return result["data"]
        finally:
            self.repo.release_action(cid)
