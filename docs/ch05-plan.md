# 第五章实施计划

此文是本次用户需求的实施计划，不新建或发布 GitHub Issue。

## 已确认的公开测试边界

1. 裸 Agent 流：观察工具结果驱动下一步、必要信息追问、循环与 token 停止。
2. Workflow 对话流：七类意图固定四出口；强制检索及前置置信闸、业务工具、多轮 State、持久恢复和流式输出。
3. HTTP 与浏览器：建议选项无副作用，转人工纯前端模拟，点击建单才复用已有工具，重复点击幂等，跨会话请求拒绝。

Prompt 使用七类意图标注集及真实模型评估，不以固定 Prompt 字符串断言代替语义验证。

## 阶段

1. 用 TDD 手写不依赖 LangGraph 的工具循环，保留可运行演示；只复用第二章工具，移除 Agent 的建单能力。
2. 使用 LangGraph StateGraph 编排指代透传、意图 JSON、固定路由、检索、前置闸、ReAct 模型/工具迭代、日志；共用裸循环的单步能力。
3. 应用生命周期持有内置 AsyncSqliteSaver，持久化会话 State；MySQL 继续保存审计与工单。沿用单工作进程会话互斥与请求幂等。
4. 在现有原生聊天页增加两个独立按钮和确认交互，新增显式建单接口；继续发消息不触发任何动作。
5. 回归、标注评估、浏览器验收，提供命令与结果；每阶段即时追加 dev-notes/ch05.md。

## 取舍与边界

- 闲聊固定白名单先判定，完全避免模型调用；其余请求使用简单 Prompt 输出七类 JSON。指代消解原样透传。
- 检索闸复用精排分阈值；弱证据记录现有问题池并兜底；不再等待完整生成后做结构化自评。正式可观测/飞轮入库不扩展。
- Agent 的可选动作通过无副作用建议通道返回，后端只授予已建议且已完成轮次的建单确认；建单请求的去重键来自原对话轮次。
- 使用 LangGraph 官方 SQLite checkpointer 的文件持久化，默认单进程；不声称已完成分布式部署、真实客服或真实订单系统接入。

## 官方 API 查证

Context7 查询与安装版本/签名交叉核对：LangGraph 1.2.24、checkpoint-sqlite 3.1.3、LangChain core/openai 1.8.3、FastAPI 0.165.0、SQLAlchemy 2.0.58；浏览器 Playwright 1.65.2 沿用 1.x async API。

- [图 State 与条件边](https://docs.langchain.com/oss/python/langgraph/graph-api)：StateGraph、TypedDict、compile(checkpointer=...)。
- [自定义流](https://docs.langchain.com/oss/python/langgraph/streaming)：get_stream_writer、astream(stream_mode="custom")；使用默认 v1 单模式原始数据形态。
- [持久化及恢复](https://docs.langchain.com/oss/python/langgraph/checkpointers)：AsyncSqliteSaver、thread_id；同线程输入 None 续跑未完成任务，重新传输入会从入口开始。
- [LangChain 模型与工具流](https://docs.langchain.com/oss/python/langchain/models)：bind_tools、AIMessageChunk 相加聚合、ToolMessage。
- [FastAPI 生命周期](https://fastapi.tiangolo.com/advanced/events/)：lifespan 与 asynccontextmanager 管理资源。
- [SQLAlchemy 2.0 异步会话](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)：沿用短事务和唯一约束去重，只解耦建单的会话状态副作用。
- [Playwright Python](https://github.com/microsoft/playwright-python)：真实浏览器 async_api 交互、dialog 确认和截图；复用仓库原有运行方式。

没有修改 Milvus/Langfuse API，也没有引入 MCP 业务接入。
