"""编排一次工具决策、顺序执行和最终流式回复，持久保存可恢复检查点。"""

import json
import re
from contextlib import aclosing
from uuid import uuid4

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from support_agent.config import Settings
from support_agent.context import ContextBudgetExceeded, fit_messages
from support_agent.errors import ModelResponseError, model_error
from support_agent.grounding import numbered_sources
from support_agent.prompts import CUSTOMER_SERVICE
from support_agent.repositories import ChatRepository
from support_agent.schemas import ChatRequest
from support_agent.streaming import ChatStreamResponse
from support_agent.tools import ToolRegistry, build_tools


def sse(event: str, **data) -> str:
    """编码一条 SSE 事件，中文和换行通过 JSON 安全传递。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def restore_message(row: dict):
    """将持久化流水恢复为 LangChain 消息，保持工具申请与结果的关联。"""
    if row["role"] == "user":
        return HumanMessage(row["content"])
    if row["role"] == "tool":
        return ToolMessage(row["content"], tool_call_id=row["tool_call_id"])
    return AIMessage(row["content"], tool_calls=row["tool_calls"] or [])


def is_social_reply(text: str) -> bool:
    """仅允许不包含业务事实的固定寒暄直接流式发送，其他无工具答复转入检索。"""
    normalized = re.sub(r"[\s，。！？、,.!?：:]", "", text)
    return normalized in {
        "你好",
        "您好",
        "您好请问有什么可以帮您",
        "请问有什么可以帮您",
        "已收到问题",
        "谢谢",
        "不客气",
        "再见",
    }


class ChatService:
    def __init__(self, model, repo: ChatRepository, config: Settings, *, answerer=None):
        """绑定模型和仓储；同会话占用覆盖整个响应发送过程，按单工作进程运行。"""
        self.model, self.repo, self.config = model, repo, config
        self.answerer = answerer
        self.busy: set[str] = set()
        self.requests: set[str] = set()

    async def respond(self, body: ChatRequest) -> ChatStreamResponse:
        """校验请求归属、并发与预算，恢复同轮检查点后返回可清理的 SSE 响应。"""
        rid = str(body.request_id or uuid4())
        original = await self.repo.resolve_request(rid)
        if rid in self.requests:
            raise HTTPException(409, "本次请求正在执行，请等待结束后重试。")
        cid = str(body.conversation_id or (original[0] if original else uuid4()))
        if original and original != (cid, body.message):
            raise HTTPException(409, "请求标识已用于其他会话或问题，请重新提交。")
        if cid in self.busy:
            raise HTTPException(409, "同一会话已有正在生成的回复，请等待结束后再发送。")
        # 数据库操作可能让出控制权，先占用会话，防止检查与占用之间并发穿透。
        self.busy.add(cid)
        self.requests.add(rid)
        try:
            audit = await self.repo.read_conversation(cid)
            if body.conversation_id and not audit:
                raise HTTPException(404, "会话不存在。")
            rows = audit["messages"] if audit else []
            # 按轮次内消息位置排序，恢复重试检查点的执行顺序。
            turn = sorted((r for r in rows if r["request_id"] == rid), key=lambda r: r["position"])
            for row in turn:
                if row["role"] == "tool":
                    checkpoint = json.loads(row["content"])
                    data = checkpoint.get("data", {})
                    if "matches" in data and data.get("category") != body.category:
                        raise HTTPException(409, "同一请求不能更改检索品类，请创建新请求。")
            completed_ids = {
                r["request_id"]
                for r in rows
                if r["position"] == 0 and r["state"] == "completed" and r["request_id"] != rid
            }
            history = [restore_message(r) for r in rows if r["request_id"] in completed_ids]
            registry = ToolRegistry(
                build_tools(self.repo, cid, rid, body.message, category=body.category),
                timeout=self.config.tool_timeout_seconds,
                retries=self.config.tool_max_retries,
            )
            tool_cost = len(
                json.dumps(
                    [convert_to_openai_tool(t) for t in registry.tools.values()], ensure_ascii=False
                )
            )
            messages = fit_messages(
                SystemMessage(CUSTOMER_SERVICE.format(store_name="本店")),
                history,
                HumanMessage(body.message),
                self.config.input_token_budget - tool_cost,
            )
            await self.repo.begin_request(cid, rid, body.message)
        except ContextBudgetExceeded as exc:
            self.busy.discard(cid)
            self.requests.discard(rid)
            raise HTTPException(413, detail=str(exc)) from exc
        except BaseException:
            self.busy.discard(cid)
            self.requests.discard(rid)
            raise

        successful = False

        async def stream():
            """恢复或执行一次工具决策，逐个发送工具状态，并以无工具模型收敛回答。"""
            nonlocal successful
            yield sse("meta", conversation_id=cid, request_id=rid)
            parts = []
            try:
                saved = {row["position"]: row for row in turn}
                if 4 in saved and saved[4]["state"] == "completed":
                    replay_matches = []
                    for row in turn:
                        if row["role"] == "tool":
                            call = next(
                                c for c in saved[1]["tool_calls"] if c["id"] == row["tool_call_id"]
                            )
                            result = json.loads(row["content"])
                            if call["name"] == "query_faq" and result.get("ok"):
                                replay_matches.extend(result["data"]["matches"])
                            yield self.tool_status(
                                call, "succeeded" if result["ok"] else "failed", reused=True
                            )
                    parts.append(saved[4]["content"])
                    yield sse("sources", sources=numbered_sources(replay_matches))
                    yield sse("token", text=parts[0])
                else:
                    if 1 in saved:
                        decision = restore_message(saved[1])
                    else:
                        aggregate = None
                        async with aclosing(
                            self.model.bind_tools(list(registry.tools.values())).astream(messages)
                        ) as chunks:
                            async for chunk in chunks:
                                self.validate_chunk(chunk, allow_tools=True)
                                aggregate = chunk if aggregate is None else aggregate + chunk
                                if chunk.content:
                                    parts.append(chunk.content)
                                    if is_social_reply("".join(parts)):
                                        yield sse("token", text=chunk.content)
                        if aggregate is None:
                            raise ModelResponseError("模型未返回内容。")
                        reason = aggregate.response_metadata.get("finish_reason")
                        if aggregate.invalid_tool_calls:
                            raise ModelResponseError("工具申请的参数 JSON 无效。")
                        decision = AIMessage(
                            content=aggregate.content, tool_calls=aggregate.tool_calls
                        )
                        if (
                            not decision.tool_calls
                            and reason == "stop"
                            and decision.content.strip()
                            and not is_social_reply(decision.content)
                        ):
                            # 模型跳过查证时由程序补上固定检索，不发送其未经核实的事实。
                            decision = AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "query_faq",
                                        "args": {"keyword": body.message[:32]},
                                        "id": "required_faq_" + rid,
                                        "type": "tool_call",
                                    }
                                ],
                            )
                            reason = "tool_calls"
                        if decision.tool_calls:
                            ids = [call["id"] for call in decision.tool_calls]
                            if (
                                reason != "tool_calls"
                                or len(ids) > 2
                                or any(not i for i in ids)
                                or len(set(ids)) != len(ids)
                            ):
                                raise ModelResponseError(
                                    "工具调用申请不完整或超过本轮两个工具的限制。"
                                )
                            await self.repo.save_message(
                                cid,
                                rid,
                                1,
                                "assistant",
                                decision.content,
                                tool_calls=decision.tool_calls,
                            )
                        elif reason != "stop" or not decision.content.strip():
                            raise ModelResponseError("模型回复为空、被截断或未正常结束。")
                    if decision.tool_calls:
                        messages.append(decision)
                        for index, call in enumerate(decision.tool_calls, start=2):
                            yield self.tool_status(call, "running")
                            if index in saved:
                                result = json.loads(saved[index]["content"])
                            else:
                                result = await registry.execute(call["name"], call["args"])
                                await self.repo.save_message(
                                    cid,
                                    rid,
                                    index,
                                    "tool",
                                    json.dumps(result, ensure_ascii=False),
                                    tool_call_id=call["id"],
                                )
                            messages.append(
                                ToolMessage(
                                    json.dumps(result, ensure_ascii=False), tool_call_id=call["id"]
                                )
                            )
                            yield self.tool_status(
                                call,
                                "succeeded" if result["ok"] else "failed",
                                attempts=result["attempts"],
                                reused=index in saved,
                            )
                        parts = []
                        reason = None
                        faq_results = [
                            json.loads(m.content)
                            for m in messages
                            if isinstance(m, ToolMessage)
                            and any(
                                c["name"] == "query_faq" and c["id"] == m.tool_call_id
                                for c in decision.tool_calls
                            )
                        ]
                        if faq_results and self.answerer:
                            result = await self.answerer.generate(
                                body.message,
                                [
                                    m
                                    for r in faq_results
                                    if r.get("ok")
                                    for m in r["data"]["matches"]
                                ],
                                verified_tools=[
                                    json.loads(m.content)
                                    for m in messages
                                    if isinstance(m, ToolMessage)
                                    and all(
                                        c["name"] != "query_faq" or c["id"] != m.tool_call_id
                                        for c in decision.tool_calls
                                    )
                                ],
                            )
                            if result["reason"]:
                                await self.repo.record_low_confidence(
                                    cid,
                                    rid,
                                    body.message,
                                    result["entry_point"],
                                    result["reason"],
                                )
                            yield sse("sources", sources=result["sources"])
                            parts = [result["answer"]]
                            yield sse("token", text=parts[0])
                            reason = "stop"
                        else:
                            # 普通回复按实际消息校验预算；知识回复由answerer校验投影后的证据输入。
                            # 工具申请和结果不可拆散，只能裁掉完整旧轮次。
                            while (
                                count_tokens_approximately(messages, chars_per_token=1.0)
                                > self.config.input_token_budget
                            ):
                                start = next(
                                    (
                                        i
                                        for i in range(2, len(messages))
                                        if isinstance(messages[i], HumanMessage)
                                    ),
                                    None,
                                )
                                if start is None:
                                    raise ModelResponseError(
                                        "工具结果超过输入预算，请缩短问题或调大上下文窗口。"
                                    )
                                del messages[1:start]
                            async with aclosing(self.model.astream(messages)) as chunks:
                                async for chunk in chunks:
                                    self.validate_chunk(chunk, allow_tools=False)
                                    if chunk.content:
                                        parts.append(chunk.content)
                                        yield sse("token", text=chunk.content)
                                    reason = chunk.response_metadata.get("finish_reason") or reason
                        if reason != "stop" or not "".join(parts).strip():
                            raise ModelResponseError("最终回复为空、被截断或未正常结束。")
                    await self.repo.save_message(cid, rid, 4, "assistant", "".join(parts))
                successful = True
                yield sse("done", conversation_id=cid, request_id=rid)
            except Exception as exc:
                await self.repo.save_message(
                    cid, rid, 4, "assistant", "".join(parts), state="failed"
                )
                yield sse("error", **model_error(exc))
            finally:
                if not successful:
                    await self.repo.save_message(
                        cid, rid, 4, "assistant", "".join(parts), state="failed"
                    )

        async def commit():
            """仅在完整成功发送响应后，将该轮次标为可供后续模型使用。"""
            if successful:
                await self.repo.set_request_state(rid, "completed")

        async def close():
            """断流时保留审计、标记失败，并在任何数据库结果下释放会话占用。"""
            try:
                current = await self.repo.read_conversation(cid)
                user = next(
                    r for r in current["messages"] if r["request_id"] == rid and r["position"] == 0
                )
                if user["state"] != "completed":
                    await self.repo.set_request_state(rid, "failed")
            finally:
                self.busy.discard(cid)
                self.requests.discard(rid)

        return ChatStreamResponse(stream(), commit, close)

    @staticmethod
    def validate_chunk(chunk, *, allow_tools: bool):
        """拒绝非文本、拒答和收敛阶段的新工具申请，防止隐式进入 Agent 循环。"""
        if not isinstance(chunk.content, str) or chunk.additional_kwargs.get("refusal"):
            raise ModelResponseError("模型返回了不支持的内容或拒绝响应。")
        if not allow_tools and chunk.tool_call_chunks:
            raise ModelResponseError("最终回答阶段不允许再次调用工具。")

    @staticmethod
    def tool_status(call: dict, state: str, **extra) -> str:
        """生成带调用关联标识的工具状态帧，不向聊天页暴露参数或数据库详情。"""
        return sse("tool_status", name=call["name"], tool_call_id=call["id"], state=state, **extra)
