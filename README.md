# 电商智能客服

面向购物咨询与售后沟通的客服应用，支持流式对话、订单与政策查询、人工工单、退款申请登记、知识上传、知识审核和检索效果评估。

后端使用 FastAPI、LangChain / LangGraph 和 SQLAlchemy；MySQL 保存业务数据与知识原文，Milvus 提供混合检索，SQLite 保存工作流 checkpoint。聊天、评估与审核页面由同一个应用提供。

## 当前能力与边界

- 同会话问题理解、意图分流、历史回载、上下文分层与追加摘要。
- 知识导入、dense + BM25 混合检索、精排、证据门控和来源引用。
- 物流、在保与退货进度通过独立 MCP 服务查询；订单与外部服务均使用演示数据。
- 工单需明确确认，退款提交只登记待核实申请，不执行资金操作；“转人工”目前是页面演示。
- 未解决问题与反馈进入问题池，经整理和人工核准后发布为知识。
- 提供检索评估、按意图统计的模型用量和可选的自托管 Langfuse 链路观测。

当前按本机单应用、单工作进程运行，使用 `demo-user`，没有用户身份鉴权或跨会话记忆。会话 ID 可用于读取会话，部署时需控制访问范围。

## 环境准备

需要 Python 3.11–3.13、uv、Docker Compose，以及支持流式工具调用和指定 JSON 输出方式的模型服务。知识检索使用硅基流动的 BGE-M3 嵌入与 BGE reranker，需要对应密钥。Node.js 22+ 仅用于前端协议测试。

以下命令均在仓库根目录的 PowerShell 中执行。

```powershell
uv sync --locked
# 仅首次配置且不存在 .env 时复制，已有配置不要覆盖。
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
docker compose -f compose.yaml up -d
docker compose -f compose.milvus.yaml up -d
```

编辑 `.env`，填写模型与硅基流动密钥。使用仓库自带 MySQL Compose 时，数据库连接应为：

```dotenv
DATABASE_URL=mysql+aiomysql://support_agent:support_agent@127.0.0.1:3306/support_agent?charset=utf8mb4
```

仓库 Compose 与 `.env.example` 统一使用本机 3306 端口，数据库、用户名和密码均为 `support_agent`；复用其他 MySQL 时按实际连接信息调整。`ROUTER_MODEL` 应为当前提供方支持的模型；不单独指定时删除该配置项，代码会沿用 `OPENAI_MODEL`。DeepSeek 官方接口应使用 `STRUCTURED_OUTPUT_METHOD=json_mode`。完整配置见[配置说明](docs/configuration.md)。

等待数据库和 Milvus 就绪后，首次导入演示知识；已有知识索引无需重复重建：

```powershell
uv run --locked python scripts/knowledge.py build --no-sync
uv run --locked python scripts/knowledge.py sync --rebuild
```

## 启动与使用

分别在三个终端启动两个 MCP 服务和客服应用：

```powershell
# 终端一：物流 MCP
uv run poe mcp-logistics
```

```powershell
# 终端二：售后 MCP
uv run poe mcp-aftersales
```

```powershell
# 终端三：客服应用
uv run poe dev
```

开发任务由 Poe the Poet 管理，定义在 `pyproject.toml`。两个 MCP 任务自动设置 `PYTHONPATH=src`，分别监听 `127.0.0.1:8101` 和 `127.0.0.1:8102`。`dev` 任务固定使用单工作进程、监听 `127.0.0.1:8010`，并设置 `MODEL_THINKING=disabled` 覆盖模型默认的思考模式。执行 `uv run poe` 可查看任务列表；MySQL、Milvus 和三个应用进程仍需按上述步骤单独启动。

| 页面 | 地址与用途 |
| --- | --- |
| 聊天 | <http://127.0.0.1:8010/>，咨询、订单选择、知识上传及工单确认 |
| RAG 评估 | <http://127.0.0.1:8010/evaluations>，查看报告或启动真实评估 |
| 知识审核 | <http://127.0.0.1:8010/reviews>，处理知识缺口、核准答案及查看运营指标 |
| 接口文档 | <http://127.0.0.1:8010/docs> |

可尝试“BX-500容量是多少”“查询订单1001物流轨迹”“这个能退吗”或“帮我建个工单”。查询和工单均以页面展示的结果、确认状态为准。

问题池默认每 10 秒整理一次，完整检索评估默认每 24 小时运行一次；这些任务可能调用付费模型。暂不自动运行时，在 `.env` 设置 `FLYWHEEL_INTERVAL_SECONDS=0` 和 `EVAL_INTERVAL_SECONDS=0`。

## 维护文档

- [模块职责与恢复契约](docs/modules.md)
- [配置与工具授权](docs/configuration.md)
- [接口与流式协议](docs/api.md)
- [知识维护与审核](docs/knowledge.md)
- [评估与验证](docs/evaluation.md)、[标注集索引](evals/README.md)
- [部署、备份与故障恢复](docs/operations.md)
- [运行与维护脚本](scripts/README.md)
- [领域词汇](GLOSSARY.md)、[架构决策](docs/adr/)

历史命名仍可能出现在脚本、数据库、checkpoint 和报告路径中；它们是现有实现使用的标识，使用时保持原名。

评估报告保存在本地 `data/rag-evaluations/`，CLI 评估报告和测试截图保存在 `data/reports/`；整个 `data/` 已被 Git 忽略，生成结果不随仓库分发。新环境没有报告时，评估页展示为空。
