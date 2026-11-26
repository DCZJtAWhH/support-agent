"""旧回归的测试装配：生产能力独立注入，断言和外部替身保持原样。"""

from support_agent.bootstrap import create_catalog, create_workflow
from support_agent.commerce import CommerceService
from support_agent.conversation import ConversationService
from support_agent.improvement import ImprovementStore
from support_agent.improvement.service import FlywheelService
from support_agent.knowledge import KnowledgePublisher, KnowledgeSearch
from support_agent.tool_adapters import build_tools as business_tools
from support_agent.tooling import ToolAuditStore


class TestServices:
    __test__ = False

    def __init__(self, database, retriever=None):
        self.database = database
        self.conversation = ConversationService(database)
        self.commerce = CommerceService(database)
        self.knowledge = KnowledgeSearch(retriever)
        self.improvement = ImprovementStore(database)
        self.audit = ToolAuditStore(database)

    @classmethod
    def from_app(cls, app):
        instance = cls.__new__(cls)
        instance.database = app.state.conversations.database
        instance.conversation = app.state.conversations
        instance.commerce = app.state.commerce
        instance.knowledge = app.state.knowledge
        instance.improvement = app.state.improvement
        instance.audit = app.state.tool_audit
        return instance

    @property
    def retriever(self):
        return self.knowledge.retriever

    @retriever.setter
    def retriever(self, value):
        self.knowledge.retriever = value

    def __getattr__(self, name):
        for capability in (
            self.conversation,
            self.commerce,
            self.knowledge,
            self.improvement,
            self.audit,
        ):
            if hasattr(capability, name):
                return getattr(capability, name)
        raise AttributeError(name)

    async def read_conversation(self, conversation_id):
        result = await self.conversation.read_conversation(conversation_id)
        if result is not None:
            result.update(await self.commerce.read_operations(conversation_id))
            result["tool_audit_logs"] = await self.audit.tool_audit_logs(conversation_id)
        return result

    async def create_ticket(self, *args, **kwargs):
        return await self.commerce.create_ticket(*args, confirmed=True, **kwargs)


def workflow(model, repo, config, checkpointer, **options):
    return create_workflow(
        model,
        repo.conversation,
        config,
        checkpointer,
        knowledge=repo.knowledge,
        improvement=repo.improvement,
        catalog=create_catalog(
            repo.commerce,
            repo.knowledge,
            repo.audit,
            servers=config.mcp_servers,
            policy_path=config.mcp_policy_path,
            timeout=config.tool_timeout_seconds,
            retries=config.tool_max_retries,
        ),
        database=repo.database,
        **options,
    )


def flywheel(repo, language, *, store=None, embeddings=None, vectors=None):
    publisher = KnowledgePublisher(store, embeddings, vectors) if store is not None else None
    return FlywheelService(repo.database, repo.conversation, language, publisher=publisher)


def build_tools(repo, conversation_id, request_id, user_text, **options):
    return business_tools(
        repo.commerce if repo else None,
        repo.knowledge if repo else None,
        conversation_id,
        request_id,
        user_text,
        **options,
    )


def tool_registry(*args, **options):
    from support_agent.bootstrap import ticket_write_policy
    from support_agent.tooling.execution import ToolRegistry

    return ToolRegistry(*args, write_policy=ticket_write_policy(), **options)
