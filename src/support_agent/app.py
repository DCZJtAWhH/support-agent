"""组装客服对话与售后信息提取接口，管理模型客户端和持久化聊天服务。"""

import json
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from support_agent.chat import ChatService
from support_agent.config import Settings
from support_agent.context import ContextBudgetExceeded, fit_messages
from support_agent.database import Database
from support_agent.errors import ModelResponseError, model_error
from support_agent.prompts import AFTER_SALES_EXTRACTION
from support_agent.repositories import ChatRepository
from support_agent.schemas import AfterSalesRequest, ChatRequest, ExtractionRequest


def create_app(
    settings: Settings | None = None, upstream_client: httpx.AsyncClient | None = None
) -> FastAPI:
    """创建应用并绑定模型与数据库配置，缺省时从环境加载。

    upstream_client 可注入异步上游客户端，其关闭由调用方负责。
    """
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        """启动模型和持久化服务，退出时释放数据库连接池及自建 HTTP 客户端。"""
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
                database = Database(config.database_url.get_secret_value())
                try:
                    await database.initialize(seed=True)
                    app.state.repository = ChatRepository(database)
                    app.state.chat = ChatService(app.state.model, app.state.repository, config)
                    yield
                finally:
                    await database.close()

    app = FastAPI(title="电商智能客服", lifespan=lifespan)
    web_directory = Path(__file__).with_name("web")
    app.mount("/static", StaticFiles(directory=web_directory), name="static")

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def chat_page():
        """提供同源客服聊天页，静态资源路径基于模块位置，不依赖启动目录。"""
        return FileResponse(web_directory / "index.html", headers={"Cache-Control": "no-cache"})

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
        """从现有聊天入口执行单轮工具决策、持久化和最终 SSE 回复。"""
        return await app.state.chat.respond(body)

    @app.get("/conversations/{conversation_id}")
    async def conversation_history(conversation_id: UUID):
        """读取演示会话的持久消息和工单审计；本章未接入用户身份认证。"""
        result = await app.state.repository.read_conversation(str(conversation_id))
        if result is None:
            raise HTTPException(404, detail="会话不存在。")
        return result

    return app
