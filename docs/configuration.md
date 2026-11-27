# 配置与工具授权

应用通过 `platform/config.py` 的 `Settings` 从系统环境变量和 UTF-8 `.env` 加载配置，系统环境变量优先。配置模板为 [`.env.example`](../.env.example)。修改环境配置后需重启应用；MCP 权限文件在调用绑定时热读取。

## 模型与数据连接

| 配置 | 当前行为 |
| --- | --- |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` | 必填模型接入配置；根地址不要带 `/chat/completions`。 |
| `MODEL_PROVIDER` | `auto` 默认仅识别 DeepSeek 官方域名；其他地址使用通用 OpenAI Chat Completions。可显式设为 `openai` 或 `deepseek`。 |
| `MODEL_THINKING` | `enabled` / `disabled`；未配置时沿用提供方默认行为。推荐启动方式显式关闭思考以配合小输出额度。 |
| `STRUCTURED_OUTPUT_METHOD` | 默认 `json_schema`，可设 `json_mode`；不支持时不会自动切换到工具调用。DeepSeek 官方接口使用 `json_mode`。 |
| `OUTPUT_TOKEN_PARAMETER` | 默认 `max_completion_tokens`；上游要求旧字段时显式设为 `max_tokens`。 |
| `ROUTER_MODEL` | 未设置时沿用主力模型；模板显式填了 DeepSeek 模型名，其他提供方需替换或移除。 |
| `MODEL_TIMEOUT_SECONDS` | 默认 60 秒。 |
| `DATABASE_URL` | 必填，正常使用 `mysql+aiomysql://...`；仓库 Compose 暴露本机 3307，模板示例为 3306。 |
| `SILICONFLOW_API_KEY` | BGE-M3 嵌入与 `BAAI/bge-reranker-v2-m3` 精排服务使用的密钥。 |
| `MILVUS_URI` | 默认 `http://127.0.0.1:19530`。 |
| `MILVUS_HYBRID_COLLECTION` | 在线混合集合，默认 `knowledge_hybrid_v1`。 |
| `MILVUS_COLLECTION` | 旧集合配置，不能代替在线混合集合配置。 |
| `CHECKPOINT_PATH` | 默认 `data/ch05-checkpoints.sqlite`；需持久保存。 |

模型必须支持本项目使用的流式工具调用及配置的 JSON 输出协议。网关地址不会只凭模型名称自动选择 DeepSeek 协议。

## 对话预算与检索门控

| 配置 | 默认值与用途 |
| --- | --- |
| `MODEL_CONTEXT_WINDOW` | 128000，模型窗口；优先于旧名称 `CONTEXT_WINDOW_TOKENS`。 |
| `MAX_OUTPUT_TOKENS` / `TOKEN_SAFETY_MARGIN` | 1024 / 250，输出及安全预留。 |
| `MAX_USER_INPUT_TOKENS` | 2000，当前输入预留。 |
| `MAX_AGENT_STEPS` | 6，每轮模型调用步数限制；优先于旧名称 `AGENT_MAX_STEPS`。 |
| `AGENT_TOKEN_BUDGET` | 1000000，Agent 累计输入输出预算。 |
| `TOOL_RESULT_MAX_TOKENS` | 1200，每步全部工具观察结果的合计额度。 |
| `RERANK_TOP_K` | 5，聊天上下文证据预算使用的配置。 |
| `TOKEN_CHARS_PER_TOKEN` | 2.0，字符折算估算，不是提供方精确 tokenizer。 |
| `SYSTEM_TOOLS_TOKENS` / `EVIDENCE_TOKENS_PER_RESULT` / `SUMMARY_INJECTION_TOKENS` | 1000 / 100 / 200，系统工具、单块证据及摘要注入预留。 |
| `HISTORY_TARGET_ROUNDS` / `STEADY_ROUND_TOKENS` | 40 / 1000，历史窗口目标与稳态估算。 |
| `LAYER2_ASSISTANT_CHARS` | 60，历史第二层客服回复投影长度。 |
| `INTENT_MIN_CONFIDENCE` | 0.65，意图分流阈值。 |
| `RERANK_MIN_SCORE` | 0.05，只作用于精排相关分，不能套用于 RRF 分数。 |
| `EVIDENCE_CALIBRATION_PATH` | 默认 `config/evidence-confidence.json`，证据门控校准产物。 |

历史预算取目标轮数估算与模型剩余空间中的较小值，再按约 70% / 30% 分配两层。每次模型调用仍会核对实际投影输入；配置不能保证所有长输入均可装入窗口。完整原文不会被压缩覆盖，机制见[模块说明](modules.md)。

`KNOWLEDGE_TOP_K` 和 `KNOWLEDGE_MIN_SCORE` 仍是配置字段；当前检索的主要控制是混合召回、精排与证据门控，不能将旧字段理解为在线完整策略。

## MCP 与工具权限

`.env` 中配置两个独立服务：

```dotenv
MCP_SERVERS={"logistics":{"transport":"streamable_http","url":"http://127.0.0.1:8101/mcp"},"aftersales":{"transport":"streamable_http","url":"http://127.0.0.1:8102/mcp"}}
MCP_POLICY_PATH=config/tool-permissions.json
TOOL_TIMEOUT_SECONDS=30
TOOL_MAX_RETRIES=1
```

每轮重新发现 MCP 工具；能发现不代表获得授权。[权限文件](../config/tool-permissions.json) 按服务名和工具名精确授予 `read`，`result_fields` 指定保留字段。服务端描述和 annotations 不授予权限。

新增外部只读工具时，先在权限文件登记名称，再重启对应 MCP 服务，客服应用下一轮即可发现。外部写工具始终拒绝。写权限仅由本地装配授予内置 `create_ticket`，且要求顾客确认。

工具统一经过 Schema 校验、授权、超时、结果整理和审计。只读瞬时网络故障可按次数重试；写操作不自动重发。`TOOL_MAX_RETRIES` 为 0–3，表示首次尝试后的重试次数。

## 后台任务与观测

| 配置 | 当前行为 |
| --- | --- |
| `FLYWHEEL_INTERVAL_SECONDS` | 默认 10 秒；设为 0 关闭自动问题整理，仍可手动触发。 |
| `EVAL_INTERVAL_SECONDS` | 默认 86400 秒；设为 0 关闭定时完整评估。 |
| `EVAL_MAX_OUTPUT_TOKENS` | 默认 4096，评估生成及逐事实判断的独立输出额度。 |
| `LANGFUSE_ENABLED` | 默认 `true`；同时提供公钥、私钥时才创建链路客户端。 |
| `LANGFUSE_BASE_URL` | 默认 `http://127.0.0.1:3000`，指向自行部署的 Langfuse。仓库 Compose 不包含它。 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 自托管项目密钥。无需链路观测时可设 `LANGFUSE_ENABLED=false`。 |

## 真实模型测试费用保护

`MODEL_TEST_BUDGET_PATH` 是可选的测试专用配置；费用闸仅接受已核价的 DeepSeek 官方纯文本接口，累计上限为人民币 1 元。它不是其他提供方的通用预算控制。

请求在 HTTP 出口预留保守费用，正常完成后按可信 usage 结算；未知费用、断流和不完整用量会停止后续调用。价格快照必须当天有效，源码内固定快照跨日会拒绝测试。

本机已有账本 `data/ch08-real-budget.sqlite` 记录未确认扣费，费用快照现保存于 `data/reports/tool-cost-report.json`，需先核账。这些是本地运行记录，不随 Git 分发。不要删除账本、换路径或清空记录绕过停止状态。普通业务运行可以不设置此测试选项，但不因此视为历史扣费已确认。
