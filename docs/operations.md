# 部署、备份与故障恢复

## 部署入口与运行约束

标准准备与启动见 [README](../README.md)。在仓库根目录通过 `support_agent.app:create_app` 启动，固定 `--workers 1`。会话占用锁、知识写入串行控制和后台任务管理按单进程设计，不支持仅增加 worker 数来扩容。

| 组件 | 本地默认位置 |
| --- | --- |
| 客服应用 | 推荐监听 `127.0.0.1:8010` |
| MySQL | `compose.yaml`，本机 3307，命名卷 `support-agent` |
| Milvus | `compose.milvus.yaml`，19530 数据端口、9091 健康检查端口，命名卷 `knowledge_milvus` |
| 物流／售后 MCP | 独立进程，8101 / 8102 |
| Langfuse | 可选的现有自托管服务，默认 3000；仓库不提供其部署栈 |

MySQL Compose 使用演示凭据；业务数据库在应用启动时创建所需表并执行兼容迁移和种子回调。

当前使用固定演示用户，没有身份鉴权，订单与售后 MCP 不是实际业务系统，退款不执行资金操作。应按这些边界配置访问，不把演示状态解释为真实处理结果。

## 数据职责与备份

| 数据 | 内容与恢复要求 |
| --- | --- |
| MySQL | 会话、消息、轮次状态、工单、退款申请、知识原文、问题池、审核、用量和评估指标历史。 |
| SQLite checkpoint | 由 `CHECKPOINT_PATH` 指定，保存 LangGraph 状态、新工具申请与结果、等待确认位置。缺失时不能仅凭 MySQL 完整恢复。 |
| Milvus | dense / BM25 混合索引。可从权威原文重建，但需时间与嵌入调用费用。 |
| `data/rag-evaluations/` | 本地生成的评估逐题证据、最新报告指针和任务状态；备份时保持指针与对应报告一致。没有已发布报告时页面为空。 |
| `data/reports/` | 独立脚本的验证报告、截图和费用导出，按追溯需要备份。 |
| `config/` 与 `.env` | 工具权限、证据校准和本机连接配置；恢复后需对应正确的知识与模型环境。 |
| 真实模型测试账本 | 累计费用与未确认预留，不能删除或换路径重置。 |

停止应用和写入任务后，成套备份 MySQL、checkpoint、评估报告及配置。SQLite 使用 WAL 时，不能在运行中仅复制主文件而丢失 `-wal` 内容；应停写后复制或使用一致性备份方式。Milvus 可备份卷，或保留足够配置后从 MySQL 重建。

恢复原文与 checkpoint 后，确认目标连接，再通过 `scripts/knowledge.py sync` 恢复 pending；索引丢失、集合变化或全量重建时使用 `sync --rebuild`。重建期间保持单写入任务。

## 日志与观测

业务日志自动以 UTF-8 写入 `log/app.log`，控制台格式由 `deploy/logging.json` 配置。

```powershell
Get-Content -LiteralPath log/app.log -Encoding UTF8 -Wait
Select-String -LiteralPath log/app.log -Pattern 'model_ctx|history_ctx|summary (trigger|start|done|skip|failed)' -Encoding UTF8
```

`history_ctx` 记录共享历史投影，`model_ctx` 记录每次模型输入、工具定义及估算，摘要记录触发、边界和结果。日志可能包含完整问题、上下文和工具结果，按业务数据管理。

Langfuse 配置有效公钥、私钥后向指定自托管地址发送链路。`GET /api/costs` 汇总真实模型 token 和未知用量次数；缺 usage 不代表零用量，也不等于账单金额。

## 常见故障

| 现象 | 处理方式 |
| --- | --- |
| 数据库连接失败 | 核对 `.env` 和 Compose 健康状态，尤其是模板 3306 与仓库 Compose 3307 的差异。使用已有 MySQL 时，先配置项目库及账号，再手动填写 `DATABASE_URL`。 |
| 模型名或 JSON 协议报错 | 核对当前提供方支持的 `OPENAI_MODEL`、`ROUTER_MODEL`、结构化方式和输出 token 参数；不依赖自动协议降级。 |
| 知识原文保存成功但上传返回 503 | 原文件重试导入或执行 `knowledge.py sync`，不要清库重来。 |
| 检索无结果 | 检查分类、原文生效状态、pending、集合名称与索引版本；变更集合需显式重建。 |
| MCP 工具不可用 | 检查独立进程、服务 URL、精确工具授权和参数；发现成功不代表授权成功。 |
| 写工具超时 | 结果可能已写入，先按原请求与业务标识查审计，禁止无依据地换请求重新提交。 |
| 聊天断流 | 保留原 `request_id` 重试，复用已执行效果；只有完整回复进入后续完整上下文。 |
| 评估启动返回 409 | 等待当前任务结束，或先同步 pending；避免并行 CLI 和网页任务。 |
| 审核发布返回 503 | 复用原核准答案重试，同一发布保留原来源与状态。 |
| 真实测试费用闸停止 | 核对账本、未确认 usage 及价格快照，不清除费用记录绕过。 |

`data/`、`log/`、`.env` 被 Git 忽略，克隆仓库不会恢复它们。评估和验证产物不再存入版本库；代码、标注集和校准参数仍由 Git 管理。Git 历史不能代替业务数据备份。
