# 电商智能客服 · 纯对话

Python + FastAPI + LangChain，实现多轮对话、SSE 文本增量和售后信息提取。使用 LangChain Core 的 PromptTemplate 与 langchain-openai 的 ChatOpenAI，不包含工具调用或 Agent 循环。

## 本地启动

需要 Python 3.11–3.13 和 uv。在仓库根目录执行：

```powershell
uv sync --locked
Copy-Item -LiteralPath .env.example -Destination .env
# 编辑 .env，填写 OPENAI_BASE_URL、OPENAI_MODEL、OPENAI_API_KEY。
uv run --locked uvicorn support_agent.app:create_app --factory --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

`.env` 按 UTF-8 读取并被 Git 忽略。已有 `.env` 时直接编辑，跳过复制。标准环境变量优先于 `.env`；修改配置后重启服务。接口文档位于 http://127.0.0.1:8000/docs 。

启动后在浏览器打开 **http://127.0.0.1:8000/** 即可使用客服聊天页，无需安装前端依赖或单独启动前端服务。输入问题并发送，客服回复会逐字显示；继续追问时自动携带当前会话 ID。Enter 发送，Shift + Enter 换行，中文输入法确认候选词不会触发发送。

页面支持等待状态、失败提示和重试本轮；生成期间不会重复提交。点击「新建对话」会中止当前请求并开始独立会话。页面刷新也会开始新会话，聊天内容和会话 ID 不写入浏览器存储。服务重启导致会话失效时，页面会提示新建对话。

模型接入统一使用 OpenAI Chat Completions 协议，地址填 API 根路径，例如 `https://服务地址/v1`，不包含 `/chat/completions`。GPT、Claude、DeepSeek 或 Ollama 对应的服务必须提供这一兼容端点；原生不同协议的 URL 无法仅靠改模型名接入。不增加应用侧协议转换层。

`STRUCTURED_OUTPUT_METHOD` 默认为 `json_schema`，可明确改为 `json_mode`。两者均调用 `with_structured_output`，不使用 `function_calling`；不支持时明确报错，不自动切换。`json_mode` 的字段约束由 Prompt 和本地 Pydantic 校验共同执行。某些旧模型名会触发 LangChain 自动覆盖结构化方式，本应用会阻断该覆盖。Ollama 等无需鉴权的兼容服务也需在 `.env` 提供一个非空占位密钥。

输出上限的请求参数由 `OUTPUT_TOKEN_PARAMETER` 指定，默认 `max_completion_tokens`；兼容服务若只接受 `max_tokens`，可在 `.env` 中明确选择后者。两种模式均受 `MAX_OUTPUT_TOKENS` 控制，每次请求只发送所选参数。

DeepSeek 官方 `/chat/completions` 的 response_format 支持 `text` 和 `json_object`，使用该接口时请配置 `STRUCTURED_OUTPUT_METHOD=json_mode`。普通 `/chat` 不发送 response_format，因此对话成功并不代表 json_schema 可用；不能将 Responses API 的 json_schema 能力套用到 Chat Completions。参见 [DeepSeek 官方接口定义](https://api-docs.deepseek.com/api/create-chat-completion/) 和 [JSON Output 指南](https://api-docs.deepseek.com/guides/json_mode/)。

若提取返回 `structured_output_rejected`，错误将包含上游 HTTP 状态、SDK 错误类型及配置提示。其他上游 HTTP 错误也提供状态与错误类型；不会返回上游响应原文、密钥或 URL。应用不自动切换结构化方式，修改 `.env` 后需重启。

## curl 演示

在另一个终端、仓库根目录执行。Windows 使用 `curl.exe`，通过 UTF-8 JSON 文件避免 PowerShell 引号问题；其他系统将其改为 `curl`。

第一轮，观察 `meta`、连续 `token` 和最后的 `done`：

```powershell
curl.exe -N -H "Content-Type: application/json" --data-binary "@examples/chat-first.json" http://127.0.0.1:8000/chat
```

将 `meta` 中的 `conversation_id` 复制到 examples/chat-second.json，再发送第二轮：

```powershell
curl.exe -N -H "Content-Type: application/json" --data-binary "@examples/chat-second.json" http://127.0.0.1:8000/chat
```

第二轮问题是“我刚才说的订单号和处理期望是什么？”，应能接上首轮的 A123 和换货诉求。

售后提取：

```powershell
curl.exe -H "Content-Type: application/json" --data-binary "@examples/after-sales.json" http://127.0.0.1:8000/after-sales/extract
```

期望得到以下语义的 JSON（期望方案允许同义表述）：

```json
{"order_id":"A123","request_type":"换货","expected_solution":"换一个新的"}
```

## 接口与上下文

| 接口 | 请求 | 返回 |
| --- | --- | --- |
| `POST /chat` | `message`、可选 `conversation_id` | SSE |
| `POST /after-sales/extract` | `description` | 固定三字段 JSON |

SSE 示例：

```text
event: meta
data: {"conversation_id":"服务端生成的 UUID"}

event: token
data: {"text":"您好"}

event: done
data: {"conversation_id":"同一 UUID"}
```

`token` 表示上游文本增量，一个增量可能含多个模型 token；收到即发送。流内失败发送 `error`，数据为 `code`、`message`，不再发送 `done`。HTTP 流已经建立时状态码仍为 200，调用方必须检查终止事件；只接受 `done` 为本轮成功。SSE 数据经过 JSON 编码，文本中的换行不会破坏事件边界。

只有模型正常结束且响应发送成功的完整轮次进入历史。上游错误、空回复、截断、异常结束、客户端断连均丢弃本轮；可沿用 `meta` 返回的 ID 重试。同会话并发返回 409；未知或重启失效的 ID 返回 404；空白输入、非法 UUID 或多余请求字段返回 422。提取失败返回 502，响应不会泄露上游错误原文。

历史保存在单进程内存中，重启后清空；不同会话 ID 隔离。本章按本机单进程运行，没有用户身份鉴权、持久化或业务工具。会话 ID 是本阶段的访问凭据，不适合作为多用户生产鉴权。

输入预算为 `CONTEXT_WINDOW_TOKENS - MAX_OUTPUT_TOKENS - TOKEN_SAFETY_MARGIN`。使用 LangChain 近似计数，每字符按约一个 token 并计入消息开销；超过预算时从最早的完整轮次开始裁剪，保留 System Prompt 与当前输入。仍超限返回 413，不调用模型。提取时也计入 schema 的近似开销。配置窗口大小需匹配上游模型；近似计数不能保证与不同模型 tokenizer 完全一致。回复达到输出上限而被截断时不保存本轮。上游超时由 `MODEL_TIMEOUT_SECONDS` 配置，关闭自动重试。

## 验证与过程

```powershell
uv run --locked pytest -q
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked python scripts/check_eval_data.py
# 前端 SSE 协议测试（需要 Node.js 22+，无需 npm install）
node --test tests/web/stream.test.mjs
```

自动化测试在 HTTP / ASGI 与 OpenAI 协议边界模拟上游，不访问真实模型，覆盖流式可见性、多轮、隔离、预算、异常回滚、断流、并发及两种结构化模式。真实模型验收由用户自行完成。

- [正式 spec（Issue #1）](https://github.com/DCZJtAWhH/support-agent/issues/1)
- [实现计划](docs/plans/0001-pure-chat.md)
- [标注集与评估判据](evals/README.md)
- [逐阶段开发记录](docs/dev-notes/0001-pure-chat.md)
- [领域词汇](GLOSSARY.md)
