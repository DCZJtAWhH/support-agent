# 功能模块与恢复契约

应用按单应用、单工作进程部署，以 `support_agent.app:create_app` 为启动入口。
下表描述当前模块的职责、数据归属和允许依赖。

| 模块 | 能力及数据归属 | 允许调用的功能模块 |
| --- | --- | --- |
| `conversation` | 请求身份和输入冲突、完整消息、轮次状态、历史投影、锚点、追加摘要 | 无 |
| `dialogue` | 理解、分流、订单点选和工单确认恢复、Agent 预算、图状态与 checkpoint | conversation、commerce、knowledge、tooling、improvement |
| `commerce` | 演示订单和商品、售后提取、人工工单、退款申请及业务幂等 | 无 |
| `knowledge` | 正式知识、候选问答、抽取批次、来源导入、pending 索引同步、混合检索、证据和引用 | 无 |
| `tooling` | 工具登记、MCP 发现、本地执行授权、Schema 校验、超时重试、结果整理、调用审计 | 无 |
| `improvement` | 轮次召回快照、反馈、未解决原问、知识缺口归并、审核和发布恢复 | conversation、knowledge |
| `evaluation` | 标注集版本、评估任务、报告、指标与历史趋势 | knowledge |

`platform` 提供共享元数据、数据库连接、配置、模型协议、JSON 调用、token 估算、日志、
链路与真实用量、费用保护；不导入功能模块。历史预算由会话模块执行，逐次模型输入及
工具观察投影由对话模块执行，完整消息与工具结果不被投影改写。

跨模块从对应包的 `__init__.py` 导入公开能力。存储模型和迁移函数在模块内维护；启动装配
可以导入这些实现以创建资源。`bootstrap.py` 负责依赖注入和资源启停，`schema.py` 登记共享
元数据、兼容迁移与种子回调，`tool_adapters.py` 登记具体工具，`legacy.py` 集中兼容旧证据。
各模块的 `routes.py` 维护传输入口；业务错误保留稳定原因，入口将其映射为既有 HTTP 状态。

## 公开能力

- `ConversationService.accept_request` 核对请求、会话、原输入和品类，并持有进程内占用；调用方在
  完整发送或取消后释放。`read_turn` 只读取指定轮次；`ConversationHistory.project` 管理独立历史
  投影和后台摘要，不修改原文。图的完整工具轨迹仍由 dialogue checkpoint 保存。
- `CommerceService.create_ticket` 和 `create_refund` 要求顾客显式确认。调用方必须先核实原轮次
  的建议权限或退款政策；退款订单取该轮服务端快照。电商能力再次验证业务字段和确认，并以
  数据库唯一约束保护幂等；相同标识的不同输入报冲突。工单为待人工跟进，退款为待核实申请。
- `KnowledgeSearch` 公开完整问题与政策查询；证据过滤、引用判断归知识模块。
  `KnowledgePublisher.publish_answer` 接收稳定来源、标准问题和人工核准答案，先提交原文，
  再同步向量。当前知识版本通过 `KnowledgeStore.corpus` 的绑定查询提供给评估，不传数据库。
- `ToolCatalog` 接受启动装配提供的工具工厂。`ConfirmedWrite` 是本地配置的确认写策略，
  描述验证由 commerce 注入。MCP 的描述和 annotation 不授予权限；外部写入继续拒绝。
  只读瞬时故障受次数和逐次超时限制，总重试时间不超过所有尝试超时与固定间隔之和。
  写操作从不自动重发；结果不确定时按原业务标识查询核实。
- `ImprovementCollector` 仅允许对话记录召回和未解决原问，采集故障按轮次记录日志后降级。
  反馈和审核服务独立执行，不进入实时对话控制。审核先保存核准答案及 `publishing`，再以
  `review:<审核ID>` 发布；失败重试必须复用原答案和原来源。新反馈读取持久快照，只有缺失时
  才使用兼容适配，且不拿最近轮次代替原轮次。
- `EvaluationDashboard` 注入知识版本查询和本模块指标仓储；不读取知识模块数据库。任务完成后
  再核对知识和标注版本，全部一致才更新报告指针，失败保留上一份报告。

## 持久化与验证

存储模型和兼容迁移由所属模块维护。会话和业务写入使用各自短事务，不向其他模块传递
事务，不在事务中等待模型、MCP 或向量调用。

必须同时备份业务数据库和 SQLite checkpoint。断流不会撤销已提交业务效果；未完整发送的
回答不会进入下一轮完整上下文。

`tests/test_persisted_compatibility.py` 从改造前提交生成的持久数据恢复，分别经 HTTP 验证已完成
请求重放、工单暂停、订单暂停、已建单后回答失败的续跑及输入冲突。夹具来源见
[fixtures/README.md](../tests/fixtures/README.md)。测试组合装配集中在
`tests/module_support.py`，生产代码没有聚合业务仓储。

架构回归检查允许依赖、禁止跨模块私有导入、数据表归属和源函数中文用途注释。
自动验收使用独立数据库和外部协议替身，不读取正式密钥、不调用真实付费模型。
运行命令见[评估与验证](evaluation.md)，数据备份要求见[运维说明](operations.md)。
