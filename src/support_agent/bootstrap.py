"""创建并连接功能模块和外部资源，不承载业务规则。"""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from pathlib import Path

import httpx
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from support_agent.commerce import CommerceService
from support_agent.commerce.authorization import valid_ticket_description
from support_agent.commerce.extraction import AfterSalesExtractor
from support_agent.conversation import ConversationHistory, ConversationService
from support_agent.dialogue.service import ChatService
from support_agent.dialogue.workflow import SupportWorkflow
from support_agent.evaluation.dashboard import EvaluationDashboard
from support_agent.evaluation.runs import EvaluationRuns
from support_agent.improvement import ImprovementCollector
from support_agent.improvement.service import FlywheelService
from support_agent.improvement.store import ImprovementStore
from support_agent.knowledge import KnowledgePublisher, KnowledgeSearch
from support_agent.knowledge.retrieval import HybridRetriever, QueryUnderstanding
from support_agent.knowledge.store import KnowledgeStore
from support_agent.knowledge.uploads import KnowledgeUploader
from support_agent.knowledge.vectors import BGEEmbeddings, BGEReranker, MilvusVectors
from support_agent.legacy import legacy_retrieval_reader
from support_agent.platform.database import Database
from support_agent.platform.model_budget import BudgetLedger, BudgetTransport
from support_agent.platform.model_gateway import create_chat_model
from support_agent.platform.observability import UsageRecorder, intent_costs, record_intent
from support_agent.platform.structured import KnowledgeLLM
from support_agent.tool_adapters import build_tools
from support_agent.tooling import ToolAuditStore, ToolCatalog
from support_agent.tooling.execution import ConfirmedWrite


def ticket_write_policy():
    """在本地装配登记工单写权限及电商验证能力，外部声明不能改变授权。"""
    return ConfirmedWrite(
        "create_ticket",
        ["ticket_no", "description", "ticket_type", "status"],
        valid_ticket_description,
        "问题描述必须引用顾客已提供的原话，请追问补齐后再发起工单预览。",
    )


def create_catalog(commerce=None, knowledge=None, audit=None, **options):
    """在装配处登记具体业务工具，tooling 只接收工具工厂与审计能力。"""
    return ToolCatalog(
        tools_factory=partial(build_tools, commerce, knowledge),
        audit=audit,
        write_policy=ticket_write_policy(),
        **options,
    )


def create_workflow(
    model,
    conversation,
    config,
    checkpointer,
    *,
    knowledge,
    improvement,
    catalog,
    database,
    trace_client=None,
):
    """注入对话所需的公开能力和观测回调，不向功能模块传递全局容器。"""
    return SupportWorkflow(
        model,
        conversation,
        config,
        checkpointer,
        knowledge=knowledge,
        improvement=ImprovementCollector(improvement),
        catalog=catalog,
        history=ConversationHistory(model, conversation, config),
        record_intent=partial(record_intent, database),
        usage_recorder=UsageRecorder(database),
        trace_client=trace_client,
    )


def create_lifespan(config, upstream_client=None):
    """创建单进程资源生命周期，网络客户端可由测试替换。"""

    @asynccontextmanager
    async def lifespan(app):
        """启动资源与模块，退出时回收后台任务、检查点和数据库连接。"""
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
                    config, upstream_client or owned_client, sync_client
                )
                database = Database(config.database_url.get_secret_value())
                vectors = MilvusVectors(config.milvus_uri, config.milvus_hybrid_collection)
                try:
                    await database.initialize(seed=True)
                    language = KnowledgeLLM(config, upstream_client or owned_client, sync_client)
                    store = app.state.knowledge_store = KnowledgeStore(database)
                    retriever = HybridRetriever(
                        store,
                        BGEEmbeddings(config.siliconflow_api_key.get_secret_value()),
                        vectors,
                        BGEReranker(config.siliconflow_api_key.get_secret_value()),
                        understanding=QueryUnderstanding(language),
                        min_score=config.rerank_min_score,
                    )
                    conversation = app.state.conversations = ConversationService(database)
                    commerce = app.state.commerce = CommerceService(database)
                    knowledge = app.state.knowledge = KnowledgeSearch(retriever)
                    improvement = app.state.improvement = ImprovementStore(database)
                    audit = app.state.tool_audit = ToolAuditStore(database)
                    app.state.evaluation_runs = EvaluationRuns(database)
                    app.state.evaluations = EvaluationDashboard(
                        store.corpus, runs=app.state.evaluation_runs
                    )
                    app.state.knowledge_uploads = KnowledgeUploader(
                        store, retriever.embeddings, vectors
                    )
                    app.state.extraction = AfterSalesExtractor(app.state.model, config)
                    app.state.read_costs = partial(intent_costs, database)

                    async def read_audit(conversation_id):
                        """通过各模块公开读取组合审计，不直接访问其他模块的数据表。"""
                        result = await conversation.read_conversation(conversation_id)
                        if result is not None:
                            result.update(await commerce.read_operations(conversation_id))
                            result["tool_audit_logs"] = await audit.tool_audit_logs(conversation_id)
                        return result

                    app.state.read_audit = read_audit
                    if config.checkpoint_path != ":memory:":
                        Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    checkpointer = await resources.enter_async_context(
                        AsyncSqliteSaver.from_conn_string(config.checkpoint_path)
                    )
                    catalog = create_catalog(
                        commerce,
                        knowledge,
                        audit,
                        servers=config.mcp_servers,
                        policy_path=config.mcp_policy_path,
                        timeout=config.tool_timeout_seconds,
                        retries=config.tool_max_retries,
                    )
                    workflow = create_workflow(
                        app.state.model,
                        conversation,
                        config,
                        checkpointer,
                        knowledge=knowledge,
                        improvement=ImprovementCollector(improvement),
                        catalog=catalog,
                        database=database,
                    )
                    app.state.chat = ChatService(workflow, conversation, commerce, config)
                    app.state.flywheel = FlywheelService(
                        database,
                        conversation,
                        language,
                        publisher=KnowledgePublisher(store, retriever.embeddings, vectors),
                        recover_legacy=legacy_retrieval_reader(workflow.read_legacy_retrieval),
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
                        await asyncio.to_thread(app.state.chat.workflow.tracing.client.flush)
                    await vectors.close()
                    await database.close()

    return lifespan
