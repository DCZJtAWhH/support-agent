"""组装客服对话与售后信息提取接口，管理模型客户端和进程内会话历史。"""

import json
import warnings
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from support_agent.config import Settings
from support_agent.context import ContextBudgetExceeded, fit_messages
from support_agent.errors import ModelResponseError, model_error
from support_agent.prompts import AFTER_SALES_EXTRACTION, CUSTOMER_SERVICE
from support_agent.schemas import AfterSalesRequest, ChatRequest, ExtractionRequest
from support_agent.streaming import ChatStreamResponse


def sse(event: str, **data) -> str:
    """将事件名与数据编码为一条 SSE 消息，保留 JSON 数据中的中文字符。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(
    settings: Settings | None = None, upstream_client: httpx.AsyncClient | None = None
) -> FastAPI:
    """创建应用及其独立的会话状态，缺省时从环境加载配置。

    upstream_client 可注入异步上游客户端，其关闭由调用方负责。
    """
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        """启动时初始化模型，退出时关闭应用自行创建的同步和异步 HTTP 客户端。"""
        with httpx.Client() as sync_client:
            async with httpx.AsyncClient() as owned_client:
                app.state.model = ChatOpenAI(
                    model=config.openai_model,
                    base_url=config.openai_base_url,
                    api_key=config.openai_api_key.get_secret_value(),
                    http_client=sync_client,
                    http_async_client=upstream_client or owned_client,
                    timeout=config.model_timeout_seconds,
                    max_tokens=(
                        config.max_output_tokens
                        if config.output_token_parameter == "max_completion_tokens"
                        else None
                    ),
                    extra_body=(
                        {"max_tokens": config.max_output_tokens}
                        if config.output_token_parameter == "max_tokens"
                        else None
                    ),
                    max_retries=0,
                    stream_usage=False,
                    use_responses_api=False,
                )
                yield

    app = FastAPI(title="电商智能客服", lifespan=lifespan)
    web_directory = Path(__file__).with_name("web")
    app.mount("/static", StaticFiles(directory=web_directory), name="static")

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def chat_page():
        """提供同源客服聊天页，静态资源路径基于模块位置，不依赖启动目录。"""
        return FileResponse(web_directory / "index.html", headers={"Cache-Control": "no-cache"})

    conversations: dict[str, list[BaseMessage]] = {}
    busy: set[str] = set()

    @app.post("/after-sales/extract", response_model=AfterSalesRequest)
    async def extract(body: ExtractionRequest):
        """从本次售后描述提取并校验结构化信息；超出预算返回 413，模型异常返回 502。"""
        schema = json.dumps(AfterSalesRequest.model_json_schema(), ensure_ascii=False)
        schema_cost = len(schema) if config.structured_output_method == "json_schema" else 0
        try:
            messages = fit_messages(
                SystemMessage(AFTER_SALES_EXTRACTION.format(schema=schema)),
                [],
                HumanMessage(body.description),
                config.input_token_budget - schema_cost,
            )
        except ContextBudgetExceeded as exc:
            raise HTTPException(413, detail=str(exc)) from exc
        try:
            options = {"strict": True} if config.structured_output_method == "json_schema" else {}
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", message=".*Overriding.*", category=UserWarning)
                    structured = app.state.model.with_structured_output(
                        AfterSalesRequest,
                        method=config.structured_output_method,
                        include_raw=True,
                        **options,
                    )
            except UserWarning as exc:
                raise ModelResponseError(
                    "LangChain 无法保留所选结构化方式；请在 .env 中明确选择兼容方式，"
                    "不会降级为工具调用。"
                ) from exc
            result = await structured.ainvoke(messages)
            if (
                result["parsing_error"] is not None
                or result["parsed"] is None
                or result["raw"].response_metadata.get("finish_reason") != "stop"
            ):
                raise ModelResponseError("模型未返回符合售后字段约束的完整 JSON，请检查上游能力。")
            return result["parsed"]
        except Exception as exc:
            raise HTTPException(
                502,
                detail=model_error(exc, structured_output_method=config.structured_output_method),
            ) from exc

    @app.post(
        "/chat",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def chat(body: ChatRequest):
        """校验会话与输入预算后返回 SSE 回复，并占用该会话直到响应关闭。

        成功发送完整回复后才保存本轮历史；会话不存在或正忙时分别返回 404、409。
        """
        conversation_id = str(body.conversation_id or uuid4())
        if body.conversation_id and conversation_id not in conversations:
            raise HTTPException(404, detail="会话不存在或已因服务重启失效。")
        if conversation_id in busy:
            raise HTTPException(409, detail="同一会话已有正在生成的回复，请等待结束后再发送。")
        history = conversations.get(conversation_id, [])
        try:
            messages = fit_messages(
                SystemMessage(CUSTOMER_SERVICE.format(store_name="本店")),
                history,
                HumanMessage(body.message),
                config.input_token_budget,
            )
        except ContextBudgetExceeded as exc:
            raise HTTPException(413, detail=str(exc)) from exc
        conversations.setdefault(conversation_id, [])
        busy.add(conversation_id)
        completed_history: list[BaseMessage] | None = None

        async def stream():
            """依次生成会话信息、文本增量和结束事件，仅在模型正常结束后暂存完整历史。

            模型异常时生成错误事件并结束本轮，暂存的历史由响应发送完成回调提交。
            """
            nonlocal completed_history
            yield sse("meta", conversation_id=conversation_id)
            parts = []
            finish_reason = None
            try:
                async with aclosing(app.state.model.astream(messages)) as chunks:
                    async for chunk in chunks:
                        if chunk.tool_call_chunks or chunk.additional_kwargs.get("refusal"):
                            raise ModelResponseError("模型返回了工具调用或拒绝响应，本轮未保存。")
                        if not isinstance(chunk.content, str):
                            raise ModelResponseError("模型返回了不支持的非文本内容，本轮未保存。")
                        if chunk.content:
                            parts.append(chunk.content)
                            yield sse("token", text=chunk.content)
                        finish_reason = (
                            chunk.response_metadata.get("finish_reason") or finish_reason
                        )
                if finish_reason != "stop" or not "".join(parts).strip():
                    raise ModelResponseError("模型回复为空、被截断或未正常结束，本轮未保存。")
            except Exception as exc:
                yield sse("error", **model_error(exc))
                return
            completed_history = [*messages[1:], AIMessage("".join(parts))]
            yield sse("done", conversation_id=conversation_id)

        def commit():
            """响应发送完成后，将已生成的完整对话轮次写入当前会话历史。"""
            if completed_history is not None:
                conversations[conversation_id] = completed_history

        return ChatStreamResponse(stream(), commit, lambda: busy.discard(conversation_id))

    return app
