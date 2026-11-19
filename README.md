# 电商智能客服 · 单轮工具调用

FastAPI + SQLAlchemy + MySQL + LangChain，实现客服对话中的单轮 Function Calling、SSE 流式回复和售后信息提取。五个 `@tool` 业务工具直接接入现有聊天页；不含多轮 Agent Loop、向量检索或真实电商接口。

## 本地启动

需要 Python 3.11–3.13 和 uv。在仓库根目录执行：

```powershell
uv sync --locked
# copy .env.example rename .env
# 编辑 .env，填写 OPENAI_BASE_URL、OPENAI_MODEL、OPENAI_API_KEY。
# 复用本机已运行的 MySQL Docker 容器：仅创建本项目数据库与账号。
uv run --locked python scripts/setup_mysql.py --container mysql-container
uv run --locked python scripts/seed_db.py
uv run --locked uvicorn support_agent.app:create_app --factory --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

`.env` 按 UTF-8 读取并被 Git 忽略。已有 `.env` 时直接编辑，跳过复制。标准环境变量优先于 `.env`；修改配置后重启服务。接口文档位于 http://127.0.0.1:8000/docs 。

MySQL 连接地址必须通过 `DATABASE_URL` 环境变量或 `.env` 提供，代码没有默认连接地址；缺失时启动即报配置错误。PowerShell 中可设置 `$env:DATABASE_URL = 'mysql+aiomysql://用户名:密码@127.0.0.1:3306/数据库名?charset=utf8mb4'`，也可以在 `.env` 中添加同名配置。密码包含 `@`、`:` 等特殊字符时需使用 URL 百分号编码；配置对象以 SecretStr 隐藏连接字符串，避免普通日志显示凭据。

没有现成 MySQL 时，可运行 `docker compose up -d --wait mysql`，再执行 `uv run --locked python scripts/setup_mysql.py --container support-agent-ch02-mysql --port 3307` 和种子命令。Compose 将 MySQL 8.4.9 映射到本机 3307，避免占用已有 3306。仓库内账号密码均仅供本地演示。初始化脚本从容器环境读取 root 凭据但不输出，保留其他数据库；已有正式数据库可直接配置 `DATABASE_URL`，跳过该脚本。

数据库只有 `faq`、`conversations`、`messages`、`tickets` 四表。种子命令可重复执行，补充 FAQ 和一段演示会话/工单，不清空已有数据。SQLAlchemy 在应用启动时建立缺失表并补充 FAQ；本章使用建表初始化，后续修改已有表结构时需另行迁移。

启动后在浏览器打开 **http://127.0.0.1:8000/** 即可使用客服聊天页，无需安装前端依赖或单独启动前端服务。输入问题并发送，客服回复会逐字显示；继续追问时自动携带当前会话 ID。Enter 发送，Shift + Enter 换行，中文输入法确认候选词不会触发发送。

页面支持等待状态、失败提示和重试本轮；生成期间不会重复提交。客服气泡内显示工具名称与执行状态徽章。点击「新建对话」会中止当前请求并开始独立会话。页面刷新也会开始新会话，聊天内容和会话 ID 不写入浏览器存储；已保存的后端记录不会因服务重启而丢失。

直接在聊天页输入：

- `订单 1001 的物流到哪了`：选择 `query_logistics`，显示徽章，并依据随机演示物流回答。
- `退货政策是什么`：`query_faq` 用“退货”查表，依据签收后 7 天等样例政策回答。
- `邮费是多少`：`query_faq` 用“邮费”查表无结果。这是预期漏召回，样例库使用“运费”表述，禁止同义词扩展。
- `商品损坏了，请帮我创建人工工单`：实际写入待处理工单，返回工单号；仅报告损坏或 FAQ 无结果不会自动开单。

`query_order`、`query_product`、`query_logistics` 在工具内部随机生成数据并标记 `simulated=true`，不建对应表，也不接真实系统。`query_faq` 使用转义通配符的 SQL LIKE；关键词必须是本轮用户原文的连续片段。`create_ticket` 由服务端绑定会话及请求标识，每次请求最多创建一张工单。

调用顺序是“模型一次决策（最多两个工具）→ 依次执行 → ToolMessage 回灌 → 未绑定工具的模型流式回答”。没有工具申请时直接流式返回该次回答；回灌后即使模型意外再次申请工具，也报错终止，绝不进入新一轮调用。工具参数执行前使用 Pydantic 严格校验；未知工具、参数错误、超时和执行失败均返回结构化错误。默认单次工具超时 5 秒，超时和数据库瞬时故障额外重试一次，可通过 `TOOL_TIMEOUT_SECONDS`、`TOOL_MAX_RETRIES` 调整；模型请求本身不自动重试。

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

本章三个验收请求：

```powershell
curl.exe -N -H "Content-Type: application/json" --data-binary "@examples/chat-logistics.json" http://127.0.0.1:8000/chat
curl.exe -N -H "Content-Type: application/json" --data-binary "@examples/chat-faq.json" http://127.0.0.1:8000/chat
curl.exe -N -H "Content-Type: application/json" --data-binary "@examples/chat-faq-miss.json" http://127.0.0.1:8000/chat
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
| `POST /chat` | `message`、可选 `conversation_id`、`request_id` | SSE |
| `GET /conversations/{id}` | 会话 UUID | 会话、完整与失败消息、工具申请/结果及工单审计 |
| `POST /after-sales/extract` | `description` | 固定三字段 JSON |

SSE 示例：

```text
event: meta
data: {"conversation_id":"服务端生成的 UUID"}

event: tool_status
data: {"name":"query_logistics","tool_call_id":"call_123","state":"running"}

event: tool_status
data: {"name":"query_logistics","tool_call_id":"call_123","state":"succeeded","attempts":1,"reused":false}

event: token
data: {"text":"您好"}

event: done
data: {"conversation_id":"同一 UUID"}
```

`token` 表示上游文本增量，一个增量可能含多个模型 token；收到即发送。流内失败发送 `error`，数据为 `code`、`message`，不再发送 `done`。HTTP 流已经建立时状态码仍为 200，调用方必须检查终止事件；只接受 `done` 为本轮成功。SSE 数据经过 JSON 编码，文本中的换行不会破坏事件边界。

只有模型正常结束且响应发送成功的完整轮次进入后续模型上下文。上游错误、空回复、截断和客户端断连均保留失败审计及已经执行成功的工单，未完成回答不进入上下文。工具调用申请由 assistant 消息的 `tool_calls` 保存，结果以 role=tool、匹配的 `tool_call_id` 保存。`GET /conversations/{id}` 可以查看全部记录。

前端每次发送生成 `request_id`，点击“重试本轮”复用该标识，恢复已保存的工具结果和工单；即使首次 `meta` 未送达，也能找到原会话。已成功的同请求重放直接回放已保存结果，不再次调用模型。同一请求标识搭配不同问题或会话返回 409。接口调用方若不传 `request_id`，服务端会生成并放入 `meta`；要在未收到 meta 时也能可靠重试，调用方应自行生成并保存 UUID。同会话并发返回 409；未知会话返回 404；空白输入、非法 UUID 或多余字段返回 422。提取失败返回 502，响应不会泄露上游错误原文。

历史保存在 MySQL，不同会话 ID 隔离；conversations 的用户为 `demo-user`，新会话状态 active，建单后 waiting_human，工单为 pending。本章按本机单工作进程运行，内存占用锁不支持多工作进程协调，没有用户身份鉴权。会话 ID 是演示阶段的访问凭据，不适合作为多用户生产鉴权。关闭页面不会撤销已建工单；服务崩溃遗留的 running 请求可凭原 request_id 恢复。

输入预算为 `CONTEXT_WINDOW_TOKENS - MAX_OUTPUT_TOKENS - TOKEN_SAFETY_MARGIN`。使用 LangChain 近似计数，每字符按约一个 token，计入消息开销与工具 Schema；超限从最早完整轮次裁剪，保留 System Prompt 与当前输入，不拆散工具申请和结果。仍超限返回 413，不调用模型；回灌后再检查预算，当前工具结果仍超限则发送 error。售后提取也计入 Schema 的近似开销。配置窗口需匹配上游模型；近似计数不能保证与 tokenizer 完全一致。截断回复仅留失败审计。上游超时由 `MODEL_TIMEOUT_SECONDS` 配置。

## 验证与过程

```powershell
uv run --locked pytest -q
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked python scripts/check_eval_data.py
# 前端 SSE 协议测试（需要 Node.js 22+，无需 npm install）
node --test tests/web/stream.test.mjs
# 以下两项使用已启动服务及 .env 中的真实模型，会生成演示会话和工单：
uv run --locked python scripts/eval_tools.py
uv run --locked python scripts/check_ch02_browser.py
```

pytest 在 HTTP / ASGI 与 OpenAI 协议边界模拟上游，数据库使用独立 SQLite 测试库；覆盖持久化、工具错误/重试、单轮调用、工单幂等、流式可见性、隔离、预算、断流、并发及两种提取模式。真实模型的标注样例在 `evals/tools.jsonl`；评估脚本校验工具选择与 FAQ 命中，回答语义按每条 criteria 人工复核。浏览器脚本使用 Playwright 验证真实聊天页面并保存桌面/手机截图；没有浏览器缓存时先运行 `uv run --locked playwright install chromium`。

- [正式 spec（Issue #1）](https://github.com/DCZJtAWhH/support-agent/issues/1)
- [实现计划](docs/plans/0001-pure-chat.md)
- [标注集与评估判据](evals/README.md)
- [逐阶段开发记录](docs/dev-notes/0001-pure-chat.md)
- [第 02 章开发与验收记录](dev-notes/ch02.md)
- [第 02 章标注样例](evals/tools.jsonl)
- [领域词汇](GLOSSARY.md)
