"""组装客服对话与售后信息提取接口，管理模型客户端和持久化聊天服务。"""

import asyncio
import json
import logging
import warnings
from contextlib import AsyncExitStack, asynccontextmanager
from html import escape
from pathlib import Path
from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from support_agent.chat import ChatService
from support_agent.config import Settings
from support_agent.context import (
    ContextBudget,
    ContextBudgetExceeded,
    ensure_app_logging,
    fit_messages,
)
from support_agent.database import Database
from support_agent.errors import ModelResponseError, model_error
from support_agent.evaluation_dashboard import EvaluationDashboard
from support_agent.flywheel import FlywheelService
from support_agent.knowledge_mining import KnowledgeLLM
from support_agent.knowledge_store import KnowledgeStore
from support_agent.knowledge_uploads import (
    MAX_UPLOAD_BYTES,
    KnowledgeUploader,
    UploadIndexPending,
    UploadRejected,
)
from support_agent.management import router as management_router
from support_agent.model_budget import BudgetLedger, BudgetTransport
from support_agent.model_gateway import create_chat_model
from support_agent.prompts import AFTER_SALES_EXTRACTION
from support_agent.repositories import ChatRepository
from support_agent.retrieval import HybridRetriever, QueryUnderstanding
from support_agent.schemas import (
    AfterSalesRequest,
    ChatRequest,
    ExtractionRequest,
    RefundSubmission,
    TicketConfirmation,
)
from support_agent.vectors import BGEEmbeddings, BGEReranker, MilvusVectors


def create_app(
    settings: Settings | None = None, upstream_client: httpx.AsyncClient | None = None
) -> FastAPI:
    """创建应用并绑定模型与数据库配置，缺省时从环境加载。

    upstream_client 可注入异步上游客户端，其关闭由调用方负责。
    """
    config = settings or Settings()
    ensure_app_logging()
    budget = ContextBudget.from_settings(config)
    logging.getLogger(__name__).info(
        "context budget window=%s history=%s layer1=%s layer2=%s "
        "fixed=%s peak=%s chars_per_token=%s",
        config.context_window_tokens,
        budget.history,
        budget.layer1,
        budget.layer2,
        budget.fixed,
        budget.peak,
        config.token_chars_per_token,
    )

    @asynccontextmanager
    async def lifespan(app):
        """启动模型、存储、飞轮与定期评估，退出时取消任务并释放数据库及向量客户端。"""
        transport = None
        if config.model_test_budget_path:
            app.state.model_budget = BudgetLedger(config.model_test_budget_path)
            transport = BudgetTransport(httpx.AsyncHTTPTransport(retries=0), app.state.model_budget)
        with httpx.Client() as sync_client:
            async with (
                httpx.AsyncClient(transport=transport) as owned_client,
                AsyncExitStack() as resources,
            ):
                app.state.model = create_chat_model(
                    config,
                    upstream_client or owned_client,
                    sync_client,
                )
                database = Database(config.database_url.get_secret_value())
                vectors = MilvusVectors(config.milvus_uri, config.milvus_hybrid_collection)
                try:
                    await database.initialize(seed=True)
                    language = KnowledgeLLM(config, upstream_client or owned_client, sync_client)
                    app.state.knowledge_store = KnowledgeStore(database)
                    app.state.evaluations = EvaluationDashboard(app.state.knowledge_store)
                    retriever = HybridRetriever(
                        app.state.knowledge_store,
                        BGEEmbeddings(config.siliconflow_api_key.get_secret_value()),
                        vectors,
                        BGEReranker(config.siliconflow_api_key.get_secret_value()),
                        understanding=QueryUnderstanding(language),
                        min_score=config.rerank_min_score,
                    )
                    app.state.repository = ChatRepository(database, retriever)
                    app.state.knowledge_uploads = KnowledgeUploader(
                        app.state.knowledge_store,
                        retriever.embeddings,
                        vectors,
                    )
                    if config.checkpoint_path != ":memory:":
                        Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    checkpointer = await resources.enter_async_context(
                        AsyncSqliteSaver.from_conn_string(config.checkpoint_path)
                    )
                    app.state.chat = ChatService(
                        app.state.model,
                        app.state.repository,
                        config,
                        checkpointer=checkpointer,
                    )
                    app.state.flywheel = FlywheelService(
                        app.state.repository,
                        language,
                        store=app.state.knowledge_store,
                        embeddings=retriever.embeddings,
                        vectors=vectors,
                        workflow=app.state.chat.workflow,
                    )
                    app.state.flywheel.start_worker(config.flywheel_interval_seconds)
                    app.state.evaluations.start_scheduler(config.eval_interval_seconds)
                    yield
                finally:
                    if hasattr(app.state, "flywheel"):
                        await app.state.flywheel.close()
                    if hasattr(app.state, "chat"):
                        await app.state.chat.workflow.summaries.close()
                    if hasattr(app.state, "evaluations"):
                        await app.state.evaluations.close()
                    if hasattr(app.state, "chat") and app.state.chat.workflow.tracing.client:
                        await asyncio.to_thread(
                            app.state.chat.workflow.tracing.client.flush,
                        )
                    await vectors.close()
                    await database.close()

    app = FastAPI(title="电商智能客服", lifespan=lifespan)
    app.include_router(management_router)
    web_directory = Path(__file__).with_name("web")
    app.mount("/static", StaticFiles(directory=web_directory), name="static")

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def chat_page():
        """提供同源客服聊天页，静态资源路径基于模块位置，不依赖启动目录。"""
        return FileResponse(web_directory / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/conversations")
    async def conversations():
        """只读列出演示用户全部会话，按新在前排序，不接受客户端指定其他用户。"""
        return await app.state.repository.list_conversations()

    @app.get("/api/conversations/{conversation_id}/messages")
    async def messages(conversation_id: UUID):
        """回载本会话用户与客服原文，不暴露工作流审计或工具结果，未知会话返回404。"""
        audit = await app.state.repository.read_conversation(str(conversation_id))
        if audit is None:
            raise HTTPException(404, detail="会话不存在。")
        return [
            m
            for m in audit["messages"]
            if m["role"] in {"user", "assistant"} and not m["tool_calls"]
        ]

    @app.get("/evaluations", response_class=FileResponse, include_in_schema=False)
    async def evaluation_page():
        """提供独立的RAG评估结果页，让用户保留原聊天并查看评估证据。"""
        return FileResponse(
            web_directory / "evaluation.html", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/api/evaluations")
    async def evaluation_report(query_type: str = "", difficulty: str = ""):
        """返回真实报告、联合分桶指标、逐题证据与当前知识库版本信息。"""
        return await app.state.evaluations.overview(query_type=query_type, difficulty=difficulty)

    @app.get("/api/evaluations/status")
    async def evaluation_status():
        """轮询评估任务进度，不传输重复的检索原文与回答明细。"""
        return await app.state.evaluations.status()

    @app.post("/api/evaluations/run", status_code=202)
    async def start_evaluation():
        """针对当前知识库和固定标注集启动完整四策略评估，不接受任意命令或路径。"""
        return await app.state.evaluations.start()

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
        """从现有入口执行确定性 Workflow、ReAct 迭代和最终 SSE 回复。"""
        return await app.state.chat.respond(body)

    @app.post("/actions/ticket")
    async def confirm_ticket(body: TicketConfirmation):
        """用户点击并确认建工单后，调用既有工具并返回幂等工单结果。"""
        return await app.state.chat.confirm_ticket(body)

    @app.post("/actions/refund")
    async def submit_refund(body: RefundSubmission):
        """接收用户在固定原因表单中的显式提交，返回待核实的申请凭据。"""
        return await app.state.chat.submit_refund(body)

    @app.get("/conversations/{conversation_id}")
    async def conversation_history(conversation_id: UUID):
        """读取演示会话的持久消息和工单审计；本章未接入用户身份认证。"""
        result = await app.state.repository.read_conversation(str(conversation_id))
        if result is None:
            raise HTTPException(404, detail="会话不存在。")
        return result

    @app.get("/knowledge/chunks/{chunk_id}")
    async def knowledge_chunk(chunk_id: int):
        """展示权威来源原文与章节路径；过期或失效块不再作为有效原文返回。"""
        row = await app.state.knowledge_store.get(chunk_id)
        if row is None or not row.is_active:
            raise HTTPException(404, detail="来源知识已失效或不存在。")
        return {
            "chunk_id": row.id,
            "source_ref": row.source_ref,
            "section_path": row.section_path or row.questions,
            "question": row.questions,
            "answer": row.answer,
            "content_hash": row.content_hash,
        }

    @app.post("/knowledge/upload", status_code=201)
    async def upload_knowledge(file: UploadFile):
        """接收知识文件并同步建立索引；完整处理后才报告可供检索。"""
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            return await app.state.knowledge_uploads.import_document(file.filename or "", content)
        except UploadRejected as exc:
            raise HTTPException(exc.status_code, detail=str(exc)) from exc
        except UploadIndexPending as exc:
            return JSONResponse(status_code=503, content={"detail": exc.result})
        finally:
            await file.close()

    @app.get("/knowledge/chunks/{chunk_id}/original", response_class=HTMLResponse)
    async def original_document(chunk_id: int, version: str = ""):
        """按来源重组已入库原文并定位章节；旧引用链接明确提示内容版本变化。"""
        store = app.state.knowledge_store
        selected = await store.get(chunk_id)
        if selected is None or not selected.is_active:
            raise HTTPException(404, detail="来源知识已失效或不存在；请在聊天引用中查看原快照。")
        chunks = [r for r in await store.active() if r.source_ref == selected.source_ref]
        sections = "".join(
            f'<section id="chunk-{row.id}"><h2>{escape(row.section_path or row.questions)}</h2>'
            f"<pre>{escape(row.answer)}</pre></section>"
            for row in chunks
        )
        note = (
            "版本已更新，下方为当前入库原文；回答时的原文保留在聊天引用面板。"
            if version and version != selected.content_hash
            else "当前入库原文 · 按知识块展示所在章节"
        )
        return HTMLResponse(
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>知识来源原文</title><style>"
            "body{font:16px/1.9 system-ui;max-width:860px;margin:40px auto;padding:0 24px;"
            "color:#22332e}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}"
            "h1{font-size:24px;overflow-wrap:anywhere}h2{font-size:19px}"
            "section{padding:24px;border-bottom:1px solid #ddd;scroll-margin-top:24px}"
            "section:target{background:#edf5f0}p{color:#64726e}</style>"
            f"<h1>{escape(selected.source_ref)}</h1><p>{note}</p>{sections}</html>"
        )

    return app
