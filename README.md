# 电商智能客服 · 混合检索与证据回答

FastAPI + SQLAlchemy + MySQL + LangChain，实现客服工具、SSE 回复和售后信息提取。第四章加入 Milvus 原生 BM25 + BGE-M3 dense 混合检索、RRF、bge-reranker-v2-m3、引用与证据自评、低置信度问题池及四策略评估。项目为可运行 demo；本章不做指代消解、多轮改写或真实电商接口。

## 本地启动

需要 Python 3.11–3.13 和 uv。在仓库根目录执行：

```powershell
uv sync --locked
# copy .env.example rename .env
# 编辑 .env，填写聊天模型三项配置，以及 SILICONFLOW_API_KEY。
# 复用本机已运行的 MySQL Docker 容器：仅创建本项目数据库与账号。
uv run --locked python scripts/setup_mysql.py --container mysql-container
uv run --locked python scripts/seed_db.py
docker compose -f compose.milvus.yaml up -d --wait
uv run --locked python scripts/knowledge.py build
# 从第三章升级时，旧行已是 done，需向新混合集合重放全部原文：
uv run --locked python scripts/knowledge.py sync --rebuild
uv run --locked uvicorn support_agent.app:create_app --factory --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

`.env` 按 UTF-8 读取并被 Git 忽略。已有 `.env` 时直接编辑，跳过复制。标准环境变量优先于 `.env`；修改配置后重启服务。接口文档位于 http://127.0.0.1:8000/docs 。

MySQL 连接地址必须通过 `DATABASE_URL` 环境变量或 `.env` 提供，代码没有默认连接地址；缺失时启动即报配置错误。PowerShell 中可设置 `$env:DATABASE_URL = 'mysql+aiomysql://用户名:密码@127.0.0.1:3306/数据库名?charset=utf8mb4'`，也可以在 `.env` 中添加同名配置。密码包含 `@`、`:` 等特殊字符时需使用 URL 百分号编码；配置对象以 SecretStr 隐藏连接字符串，避免普通日志显示凭据。

没有现成 MySQL 时，可运行 `docker compose up -d --wait mysql`，再执行 `uv run --locked python scripts/setup_mysql.py --container support-agent-ch02-mysql --port 3307` 和种子命令。Compose 将 MySQL 8.4.9 映射到本机 3307，避免占用已有 3306。仓库内账号密码均仅供本地演示。初始化脚本从容器环境读取 root 凭据但不输出，保留其他数据库；已有正式数据库可直接配置 `DATABASE_URL`，跳过该脚本。

数据库保留原有表，知识表继续使用 `knowledge_chunks`、`qa_extraction_staging`、`knowledge_extraction_batches`。第四章新增 `low_confidence_questions`，初始化可重跑，不清空数据。旧 `faq` 仅在离线建库时作为种子来源，在线查 Milvus 并回查 MySQL 权威原文。

启动后在浏览器打开 **http://127.0.0.1:8000/** 即可使用客服聊天页，无需安装前端依赖或单独启动前端服务。输入问题并发送，客服回复会逐字显示；继续追问时自动携带当前会话 ID。Enter 发送，Shift + Enter 换行，中文输入法确认候选词不会触发发送。

页面支持等待状态、失败提示和重试本轮；生成期间不会重复提交。客服气泡显示工具徽章、可点击的引用和一次性满意度反馈。点击引用可查看原文快照、章节路径并跳回当前原文；👍/👎 选中点亮、显示“已反馈”并锁定。反馈仅保存在 localStorage，含请求/会话标识、选择与时间，不调用反馈后端。页面刷新会开始新会话，聊天正文不保存到浏览器；后端记录会持久保存。

直接在聊天页输入：

- `订单 1001 的物流到哪了`：选择 `query_logistics`，显示徽章，并依据随机演示物流回答。
- `退货政策是什么`：语义召回退货政策，依据签收后 7 天等条件回答。
- `邮费是多少`：即使库中用“运费”表述，也能召回满99元免运费、未满8元、偏远地区人工确认的说明。
- `商品损坏了，请帮我创建人工工单`：实际写入待处理工单，返回工单号；仅报告损坏或 FAQ 无结果不会自动开单。

`query_order`、`query_product`、`query_logistics` 随机生成并标记 `simulated=true`。`query_faq` 的 `keyword` 仍校验本轮原文片段及1～32字，但检索使用完整本轮问题，避免型号和否定条件被截掉。返回 `{keyword, category, matches}`，matches 保留 question/answer/category 并增加 chunk_id、章节、来源链接、原文摘要和相关分。历史问法可合并在同一块，同义词扩展只发生在查询侧。`create_ticket` 每次请求最多创建一张工单。

调用顺序是“模型一次决策（最多两个工具）→ 依次执行 → 回答”。知识工具结果进入结构化生成自评，程序先校验充分性与引用，再发送 SSE；空召回、低相关、自评不足或无效引用会拒答并幂等入池。模型无工具直接输出业务事实时程序补查 FAQ，固定寒暄可直接发送。普通业务工具仍使用流式回答，不再开启工具循环。工具参数严格校验；默认单次超时30秒，超时和数据库瞬时故障额外重试一次；模型本身不自动重试。

## 第三章知识库演示

Milvus 使用官方 standalone 的内嵌 etcd 与本地持久化方式，版本2.6.18，数据保存在 Docker 命名卷；仅监听本机19530和9091端口。`compose.milvus.yaml` 独立于 MySQL Compose，复用已有 MySQL 时无需启动第二个 MySQL。首次拉取镜像需要等待；`up -d --wait` 以健康检查完成为准。

```powershell
# 导入旧 FAQ 和 Markdown，先写 MySQL pending，再写 Milvus 并回填 done
uv run --locked python scripts/knowledge.py build
uv run --locked python scripts/knowledge.py query "邮费是多少"
uv run --locked python scripts/knowledge.py stats

# 可重复演示中断：先重新标记 pending，再故意在 Milvus 写后、MySQL 回填前退出
uv run --locked python scripts/knowledge.py sync --rebuild --interrupt-after-upsert
# 上一条退出码2为预期；不要再次带 --rebuild，直接恢复
uv run --locked python scripts/knowledge.py stats
uv run --locked python scripts/knowledge.py sync
uv run --locked python scripts/knowledge.py sync
# 第一次补齐，第二次 synchronized=0

# 手动挖掘：分批抽取→暂存→整体去重发布→补齐向量
uv run --locked python scripts/knowledge.py mine
uv run --locked python scripts/knowledge.py staging

# 独立终端常驻，北京时间每天02:00执行；--run-now 启动时先执行一次
uv run --locked python scripts/knowledge.py schedule --run-now
```

默认文档目录 `examples/knowledge/docs`，可用 `--docs 你的目录` 指定。Markdown 支持 ATX/Setext 标题层级；默认正文块最多900字，重叠约100字并保留完整句，表格按完整行切分且每块复制表头。超长完整句或单行无法容纳时明确失败，原文不静默截断。章节标题作为 questions、上级路径作为 category；`<!-- content_type: faq -->` 声明商品FAQ，其章节标题应为真实问法；`<!-- key_clause -->` 显式标记关键条款。四类元数据只保存在 MySQL，不加入嵌入文本。仅 category/questions/answer 被拼接向量化。

同一目录重复导入按来源和块序号幂等；修改的块转 pending，移除的文档或尾块失效并删除对应向量。`build --no-sync` 可只落原文；重跑 `sync` 会补齐任何来源的 pending。以 MySQL id 为 Milvus 主键 upsert，回查时校验摘要、生效状态及 done，更新期间可能暂时少召回。Milvus 额外保存 category 以执行检索前过滤，完整来源与章节仍以 MySQL 为准。demo 按单个建库/抽取任务进程运行，不要并发执行多个写任务。

历史导出默认位于 `examples/knowledge/history`，可用 `--history 你的目录` 指定。每行是包含 `conversation_id`、`reviewed:true` 和 `messages:[{role,content}]` 的 JSON；可选 state，只有 completed 消息参与。仓库例子均为虚构演示，不代表真实商家政策。现有库中的模型回复不自动作为已审核历史。成功批次有摘要检查点，重复扫描不重复抽取；抽取 Prompt 改动会形成新批次。所有候选先暂存，再与本轮和既有知识整体去重，同义问法合并，答案冲突保留暂存；可用 `mine --retry-conflicts` 重新核对。离线近邻与LLM只服务知识治理，在线仍为 dense 单路。

`KNOWLEDGE_TOP_K` / `KNOWLEDGE_MIN_SCORE` 仅供第三章 dense 基线。当前在线固定每路50、RRF k=60、融合最多100、精排10；`RERANK_MIN_SCORE` 默认0.05，仅作用于精排分。初始0.4曾误拒答邮费同义问法，0.05来自开发集调试，需用真实业务留出集再次校准；生成充分性门控继续执行。BGE-M3 固定1024维，不能在服务故障时替换模型或改用本地 BM25。

```powershell
# 真实检索标注集 + 故意中断恢复；会调用硅基流动
uv run --locked python scripts/eval_knowledge.py --recovery
# 同时验证已启动聊天服务的真实回答
uv run --locked python scripts/eval_knowledge.py --chat-url http://127.0.0.1:8000 --output dev-notes/ch03-chat-eval-results.json
```

知识抽取的语义判据见 `evals/knowledge-mining.jsonl`，检索判据见 `evals/knowledge.jsonl`；模型生成内容需结合实际输出复核。

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
| `POST /chat` | `message`、可选 `conversation_id`、`request_id`、`category` | SSE，知识回答含 sources 来源快照 |
| `GET /conversations/{id}` | 会话 UUID | 会话、完整与失败消息、工具申请/结果及工单审计 |
| `GET /knowledge/chunks/{id}` | 知识块主键 | 权威原文、章节路径、内容摘要 |
| `GET /knowledge/chunks/{id}/original` | 可选 `version` 内容摘要 | 所在来源文档，支持 `#chunk-{id}` 定位，版本变化提示 |
| `POST /after-sales/extract` | `description` | 固定三字段 JSON |
| `POST /knowledge/upload` | multipart 的 `file` | 201：文件名、来源、知识块主键与数量；503：原文已保存但索引待重试 |

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

`token` 表示可向用户展示的文本片段，知识答案先通过门控再发送，页面仍逐字展示。`sources` 事件携带本轮编号与来源原文快照；失败发送 `error`，不再发送 `done`。HTTP 流建立后仍为200，必须检查终止事件；只有 `done` 表示成功。SSE 数据经过 JSON 编码，文本换行不会破坏事件边界。

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
- [当前工具标注样例](evals/tools.jsonl)
- [第 03 章实现计划](docs/plans/0002-dense-knowledge-demo.md)
- [第 03 章逐阶段开发记录](dev-notes/ch03.md)
- [领域词汇](GLOSSARY.md)

## 第四章演示与评估

Milvus 服务与 SDK 固定2.6.18。`text` 字段使用内置 `chinese` analyzer，BM25 函数在服务端生成 sparse；dense/BM25 每路50，通过 `hybrid_search` + `RRFRanker(k=60)` 融合，再由硅基流动固定 `BAAI/bge-reranker-v2-m3` 取10块。原文一块一份，查询侧归一与扩展；prompt 将最相关两块放在首尾。

新集合配置是 `MILVUS_HYBRID_COLLECTION=knowledge_hybrid_v1`，原 `MILVUS_COLLECTION` 仅保留为旧配置，不删除旧集合。第一次升级必须 `sync --rebuild`。之后普通 `sync` 增量重放即可；只修改集合名而不重建会丢失旧 done 块的索引。

```powershell
# 建库并向独立混合集合重建
uv run --locked python scripts/knowledge.py build --no-sync
uv run --locked python scripts/knowledge.py sync --rebuild
# 具型号问法：BM25 应在首位命中 BX-500 原文；品类取章节上级路径
uv run --locked python scripts/knowledge.py query "BX-500容量是多少" --strategy bm25 --category "型号手册（演示）"
uv run --locked python scripts/knowledge.py query "邮费怎么算" --strategy hybrid_rerank
# 启动聊天页，在另一个终端发送知识与拒答演示
uv run --locked uvicorn support_agent.app:create_app --factory --app-dir src --host 127.0.0.1 --port 8000 --workers 1
curl.exe -N -H "Content-Type: application/json" --data-binary "@examples/chat-model.json" http://127.0.0.1:8000/chat
curl.exe -N -H "Content-Type: application/json" --data-binary "@examples/chat-unknown.json" http://127.0.0.1:8000/chat
curl.exe -N -H "Content-Type: application/json" --data-binary "@examples/chat-insufficient.json" http://127.0.0.1:8000/chat
# 查看低置信度问题：原话、会话、请求、入口、原因、UTC 时间；同请求不重复入池
uv run --locked python scripts/knowledge.py low-confidence
# 四策略真实评估：20题，含逐事实 Faithfulness 与 query 类型/难度分桶，会调用真实模型
uv run --locked python scripts/eval_retrieval.py --output dev-notes/ch04-eval-final-results.json
# 浏览器真实交互验收
uv run --locked python scripts/check_ch04_browser.py --base-url http://127.0.0.1:8000
# 真实未知/证据不足拒答与入池、会话关联及同请求重放（使用同一 .env 的数据库）
uv run --locked python scripts/check_ch04_acceptance.py --base-url http://127.0.0.1:8000
```

报告见 [最终四策略报告](dev-notes/ch04-eval-final-results.md)，逐事实评委依据、原文、回答与排名在同名 JSON / `.rows.jsonl`。20题是演示回归集，阈值也参考了其中的开发反馈，不能视为独立留出集或证明全面优于 dense。Faithfulness 使用同一提供方的独立评委调用，结果需人工抽查；拒答无事实时显示 N/A，并同时报告拒答正确率和可回答题的作答率，防止全拒答获得虚高分数。

纯粹复核指标可使用 `--replay dev-notes/ch04-eval-final-results.rows.jsonl --output dev-notes/ch04-replay.json`，报告会标记 replay。数据、Prompt 与评估口径见 [评估说明](evals/README.md)，开发过程见 [dev-notes/ch04.md](dev-notes/ch04.md)，实施计划见 [0003-hybrid-retrieval.md](docs/plans/0003-hybrid-retrieval.md)。

## 从聊天页上传自己的知识

启动服务后，点击输入框上方的 **“上传知识”**，选择或拖入文件，再点击 **“开始导入”**。支持多文件逐个导入，每个文件都会显示上传、导入、成功或失败状态；成功后返回聊天即可提问，并能点击回答引用查看上传的来源原文。

- 格式：UTF-8 编码的 `.md` / `.markdown` / `.txt`，每个文件不超过1 MiB、最多200个知识块。PDF、Word 和图片暂未接入。
- Markdown 按标题生成章节路径与品类；TXT 可直接写正文。窗口内提供格式示例。标题需符合当前数据库长度（品类255字、章节路径512字），单个不可拆分的完整句或表格行仍受900字切块上限约束；不符合时会提示修正，原有知识不会被错误文件覆盖。
- 上传来源为 `upload:<文件名>`，同名上传会更新之前从网页上传的知识；重复上传复用知识主键，不重复建块。CLI 导入的来源独立保留，网页上传不会替换或删除 CLI 文档。
- “清空列表”只清除窗口内记录，不删除已入库知识。聊天记录与当前会话不受上传影响。
- 索引失败时，页面明确提示“原文已保存，索引尚未完成”，保留文件供 **“重试导入”**；成功的其他文件不会重复上传。也可运行 `scripts/knowledge.py sync` 恢复 pending。网页上传按单工作进程串行写入，请沿用既有单写入任务约束，不同时运行 CLI 写入任务。

升级后先 `uv sync --locked` 安装新增的 `python-multipart`，重启服务。也可通过接口导入：

```powershell
curl.exe -F "file=@examples/knowledge/docs/models.md;type=text/markdown" http://127.0.0.1:8004/knowledge/upload
# 桌面/手机的真实多文件上传、错误提示、失败重试及导入后问答引用验收
uv run --locked python scripts/check_knowledge_upload_browser.py --base-url http://127.0.0.1:8004
```

上传验收会写入名称明确标注为“演示”的知识文件；首次503重试UI场景使用浏览器故障模拟，其余导入、编码拒绝与问答引用均经过真实服务。结果保存到 `dev-notes/ch04-upload-browser-results.json`，实施计划见 `docs/plans/0004-knowledge-upload.md`。

## 在界面查看 RAG 评估

点击聊天输入框上方 **“RAG 评估”**，会在新标签页打开 `/evaluations`，保留当前聊天和草稿。页面读取已保存的真实报告，提供：

- 概况：评估时间、题目数、报告知识块数与当前知识块数、真实运行或回放标识。
- 四策略对比：Recall@1/5/10/50、MRR、Faithfulness、拒答正确率及可回答题作答率；并列最优高亮、变化量相对纯向量，支持问题类型与难度联合筛选。
- 逐题证据：标注答案要点、预期来源/拒答、四策略实际回答、召回排名与原文快照、逐事实支持判断及原因。没有事实的拒答显示 Faithfulness N/A。
- 版本关系：知识库或标注集摘要变化时明确标记历史报告。上传知识不会自动生成该知识的标注题。

点击 **“重新评估”** 会在后台运行现有真实 CLI，完整评估当前 `evals/retrieval.jsonl` 的20题×4策略，页面显示实际完成结果数。浏览页面、切换筛选不会调用模型。评估通常需要数分钟；关闭标签页后继续运行，重新打开可恢复进度；同一单进程服务只允许一个任务。请保持知识库和标注集不变；期间有变更时不会发布混合版本的结果。待同步的知识需先完成同步，才能开始。

成功结果保存在 `dev-notes/rag-evaluations/<运行UUID>/report.json`（及同名 Markdown、rows JSONL），验证完整和版本一致后才更新 `latest.json`；旧章节报告不会覆盖。失败继续展示上一份报告，并支持重试；服务退出会停止自身子进程、记录中断。维持 `--workers 1` 的本地demo运行方式，不与外部CLI评估/知识写入任务并行操作。

| 接口 | 用途 |
| --- | --- |
| `GET /evaluations` | 评估页面 |
| `GET /api/evaluations?query_type=model&difficulty=hard` | 报告、联合分桶、逐题证据及快照关系 |
| `GET /api/evaluations/status` | 轻量进度轮询 |
| `POST /api/evaluations/run` | 启动完整评估，202接受、409已有任务或知识待同步 |

```powershell
# 只验证已有报告和界面，不重新调用模型
uv run --locked python scripts/check_evaluation_browser.py --base-url http://127.0.0.1:8004
# 同时点击入口，完成一次真实重新评估、刷新恢复及进度验收
uv run --locked python scripts/check_evaluation_browser.py --base-url http://127.0.0.1:8004 --run-evaluation
```

实施计划为 `docs/plans/0005-rag-evaluation-ui.md`，过程继续记录在 `dev-notes/ch04.md`。
