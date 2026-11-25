"""编译时绑定链路与用量回调，保持请求归属和真实 token 口径。"""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from time import perf_counter

from langchain_core.callbacks import AsyncCallbackHandler
from sqlalchemy import case, func, select

from support_agent.models import ModelUsage, RequestIntent

current_root = ContextVar("support_request_observation", default=None)


class RequestTracing:
    def __init__(self, config, *, client=None):
        """只连接显式配置的自托管地址；无项目密钥时保留本地用量统计。"""
        self.client, self.callback = None, None
        public_key = config.langfuse_public_key.get_secret_value()
        secret_key = config.langfuse_secret_key.get_secret_value()
        if config.langfuse_enabled and public_key and secret_key:
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            self.client = client or Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=config.langfuse_base_url,
            )
            self.callback = CallbackHandler(public_key=public_key)

    @contextmanager
    def request(self, conversation_id, request_id, question):
        """创建请求根观察，稳定 trace_id 关联同轮暂停恢复；上下文退出自动结束观察。"""
        if self.client is None:
            yield
            return
        from langfuse import propagate_attributes

        trace_id = self.client.create_trace_id(seed=request_id)
        with (
            propagate_attributes(
                session_id=conversation_id, user_id="demo-user", metadata={"request_id": request_id}
            ),
            self.client.start_as_current_observation(
                name="support-request",
                trace_context={"trace_id": trace_id},
                input={"question": question},
                metadata={"request_id": request_id},
            ) as root,
        ):
            token = current_root.set(root)
            try:
                yield
            finally:
                current_root.reset(token)

    def set_intent(self, intent):
        """更新当前请求根观察元数据，其他并发请求的分类不会串入本请求。"""
        root = current_root.get()
        if root is not None:
            root.update(metadata={"intent": intent})

    def intent_scope(self, intent):
        """把当前图状态中的意图传播给后续模型与工具子观察。"""
        if self.client is None or not intent:
            return nullcontext()
        from langfuse import propagate_attributes

        return propagate_attributes(metadata={"intent": intent})

    def trace_id(self, request_id):
        """返回可定位同轮链路的稳定 ID，未启用 tracing 时为空。"""
        return self.client.create_trace_id(seed=request_id) if self.client else None

    def finish(self, output):
        """在根观察结束前保存最终回复及证据置信度。"""
        root = current_root.get()
        if root is not None:
            root.update(output=output)


def traced_node(node, tracing):
    """为节点增加意图上下文，实际节点与模型仍由单次绑定的图回调自动追踪。"""

    async def run(state):
        """按当前轮次意图传播属性后执行原节点，保留原异常与返回值。"""
        with tracing.intent_scope(state.get("intent")):
            return await node(state)

    return run


class UsageRecorder(AsyncCallbackHandler):
    def __init__(self, database):
        """保存运行中模型调用的时间与请求身份，按 run_id 隔离并发。"""
        self.database = database
        self.active = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        """从图继承的元数据捕获轮次身份；非客服后台模型不计入意图成本。"""
        request_id = (metadata or {}).get("request_id")
        if request_id:
            self.active[str(run_id)] = (request_id, perf_counter())

    async def save(self, run_id, usage=None):
        """以模型调用唯一标识保存真实输入输出；异常和缺 usage 均标记未知。"""
        identity = str(run_id)
        info = self.active.pop(identity, None)
        if info is None:
            return
        request_id, started = info
        usage = usage or {}
        known = all(
            type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens")
        )
        async with self.database.sessions.begin() as session:
            if not await session.get(ModelUsage, identity):
                session.add(
                    ModelUsage(
                        run_id=identity,
                        request_id=request_id,
                        input_tokens=usage["input_tokens"] if known else 0,
                        output_tokens=usage["output_tokens"] if known else 0,
                        usage_known=known,
                        duration_ms=int((perf_counter() - started) * 1000),
                    )
                )

    async def on_llm_end(self, response, *, run_id, **kwargs):
        """从 LangChain 完整聚合消息读取 usage，避免对流片段重复加计。"""
        usage = None
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    break
        await self.save(run_id, usage)

    async def on_llm_error(self, error, *, run_id, **kwargs):
        """模型失败也记录一次未知用量，避免费用统计将失败请求当作免费。"""
        await self.save(run_id)


async def record_intent(database, conversation_id, request_id, intent, trace_id=None):
    """按请求更新最终意图，所有先发生的理解与分类用量自动归入同一意图。"""
    async with database.sessions.begin() as session:
        row = await session.get(RequestIntent, request_id)
        if row is None:
            row = RequestIntent(request_id=request_id, conversation_id=conversation_id)
            session.add(row)
        row.intent = intent
        if trace_id:
            row.trace_id = trace_id


async def intent_costs(database):
    """按最终意图聚合真实 token，未知模型调用数单列，排序展示最耗 token 的意图。"""
    async with database.sessions() as session:
        rows = await session.execute(
            select(
                RequestIntent.intent,
                func.coalesce(func.sum(ModelUsage.input_tokens), 0),
                func.coalesce(func.sum(ModelUsage.output_tokens), 0),
                func.count(func.distinct(RequestIntent.request_id)),
                func.sum(case((ModelUsage.usage_known.is_(False), 1), else_=0)),
            )
            .outerjoin(ModelUsage, ModelUsage.request_id == RequestIntent.request_id)
            .group_by(RequestIntent.intent)
        )
        result = [
            {
                "intent": intent,
                "input_tokens": inputs,
                "output_tokens": outputs,
                "total_tokens": inputs + outputs,
                "request_count": requests,
                "missing_usage_count": missing or 0,
            }
            for intent, inputs, outputs, requests, missing in rows
        ]
    return sorted(result, key=lambda r: (-r["total_tokens"], r["intent"]))
