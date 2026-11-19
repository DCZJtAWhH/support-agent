# 纯对话阶段开发记录

## 2026-11-19 · 阶段 1：需求接收与事实核查

- 用户关键原话：
  - “第一步先跑通纯对话。”
  - “Python + FastAPI + LangChain。”
  - “地址、模型名、密钥全在 .env。”
  - “工具调用、Agent 循环，先跑通纯对话。”
  - “一律先用 Context7 MCP 查最新官方文档和接口定义再动手。”
  - “不许收尾时一次性补记。”
  - 显式调用 `$grill-with-docs`，要求结合 grilling 和 domain-modeling。
- 关键产出：
  - 读取仓库 AGENTS.md、docs/agents/domain.md 和上述技能。仓库尚无 GLOSSARY.md、ADR、应用代码或依赖清单。
  - 按 grilling 技能要求委派只读环境核查：main 分支、初始工作区干净；Python 3.13.7 / 3.11 和 uv 0.11.23 可用；没有 .env 或相关模型配置环境变量；没有输出密钥。
  - Context7 已核查 LangChain 的 OpenAI 兼容端点和结构化输出方式、FastAPI SSE 接口。ChatOpenAI 需要兼容 Chat Completions 的端点；json_schema 和 json_mode 是否可用取决于上游，不能仅凭更换模型名保证能力一致。
  - 文档来源：https://docs.langchain.com/oss/python/concepts/providers-and-models 、https://docs.langchain.com/oss/python/langchain/models 、https://fastapi.tiangolo.com/tutorial/server-sent-events/ 。
  - spec / plan 尚未生成；等待第一轮设计决策。需求评审尚未完成，尚未实现业务代码或运行测试。
- 用户拒绝或纠偏：本阶段尚无。
- 翻车与返工：尚无实现返工；Context7 的历史裁剪检索返回 Agent 中间件示例，不适用于本章，后续需定位纯消息裁剪的接口文档，不引入 Agent。
- 下一步：澄清会话归属、售后字段语义、客服边界、流式验收口径和首个真实上游；决策确定时即时更新领域词汇与设计文档。

## 2026-11-19 · 阶段 2：第一轮设计决策落盘

- 用户关键原话：Q1–Q5 均回答“使用推荐”；Q6 回答“模型、base URL 以及密钥，都是用 .env 文件配置。”
- 关键产出：
  - 新建根 GLOSSARY.md，记录客服对话、对话轮次、售后描述、订单号、诉求类型与期望方案的已确认语义。
  - 新建 docs/plans/0001-pure-chat.md，区分已确定设计、第二轮待确认建议与拟执行阶段；尚未实现代码。
  - 读取 docs/agents/issue-tracker.md；通过 gh 查询本仓库开放 Issues，结果为空。正式 spec 将按仓库约定记录于 GitHub Issues，当前只有设计计划草案。
  - 读取 TDD 技能；拟定公开 HTTP / 上游协议边界测试，等待用户确认测试边界。Prompt 与字段语义采用用户要求的标注样例评估。
  - Context7 补查 with_structured_output 的 json_schema / json_mode 和纯消息裁剪函数。来源：https://docs.langchain.com/oss/python/integrations/chat/openai 、https://docs.langchain.com/oss/python/langchain/models 、https://docs.langchain.com/oss/python/langgraph/add-memory 。仅参考消息工具函数，不引入 LangGraph。
  - 评审结论：首轮设计已收敛；结构化输出兼容方式、歧义处理、断流历史及验收边界仍需确认。
- 用户拒绝或纠偏：用户将第六项重新明确为 .env 配置要求；后续不再要求其在聊天中提供模型名、地址或密钥。
- 翻车与返工：无实现返工。Context7 对 ChatOpenAI 的默认结构化方法描述存在版本风险；实现必须显式传 method，并核对最终锁定版本的接口，避免依赖默认行为。
- 下一步：提出第二轮剩余边界及测试范围，设计确认后再执行实现计划。

## 2026-11-19 · 阶段 3：设计收敛与实现授权

- 用户关键原话：Q7–Q9 均“使用推荐”；Q10：“整体上是没问题，但是不用真实验收，真实验收我会自己来做，你不需要帮我。”
- 关键产出：docs/plans/0001-pure-chat.md 更新为已确认计划；明确 HTTP 与上游协议测试边界，采用垂直切片 TDD；结构化方法显式配置、不使用工具调用；仅成功轮次入历史；歧义字段保留空值或未知。
- 评审结论：设计分支已收敛，开始实现。当前取舍均可在本章规模内低成本调整，不为它们额外创建 ADR。
- 用户拒绝或纠偏：用户取消了由助手执行真实模型验收的工作。后续不得调用真实上游；仅交付供用户自行运行的命令和评估样例。
- 翻车与返工：无业务实现返工；更新此前计划中由助手执行真实评估的表述。
- 下一步：正式 spec、Context7 接口核查、工程环境与第一条流式测试。

## 2026-11-19 · 阶段 4：工程准备与首个流式切片

- 用户关键原话：继续遵守“Python + FastAPI + LangChain”“逐 token 推送”及“真实验收我会自己来做”。本阶段没有新增用户指令。
- 关键产出：
  - 正式 spec：https://github.com/DCZJtAWhH/support-agent/issues/1 。
  - pyproject.toml、uv.lock、.gitignore；uv sync 成功。锁定 FastAPI 0.165.0、LangChain Core / OpenAI 1.8.3、OpenAI SDK 3.55.1、Pydantic 2.14.3、HTTPX 0.28.1，使用 Python 3.13.7。
  - Context7 核查 PromptTemplate、ChatOpenAI、StreamingResponse、Pydantic Settings、HTTPX 测试传输、pytest 及 uv；随后检查已安装接口定义，确认 with_structured_output 默认已为 json_schema，仍按设计显式传参。
  - 第一条测试先因应用不存在失败；实现最小 POST /chat 后通过（1 passed）。模拟端点确认 /v1/chat/completions、模型名、Bearer 密钥、stream=true、无 tools 参数及 SSE meta/token/done 顺序。
  - 第二条测试已证明多轮失败（服务端没有复用会话 ID）；开始补充内存会话。
- 评审结论：首个端到端协议切片通过；尚不能作为全部功能完成。测试没有访问真实上游。
- 用户拒绝或纠偏：无新增，继续由用户负责真实验收。
- 翻车与返工：uv 缓存跨盘无法硬链接，自动退回复制，安装成功；文档默认值与已安装版本不同，已查源码消除歧义，不依赖默认值。
- 下一步：多轮与会话隔离、历史预算裁剪。

## 2026-11-19 · 阶段 5：多轮与预算控制

- 用户关键原话：Q1、Q5“使用推荐”；继续采用内存会话和近似预算，不增加持久化。
- 关键产出：src/support_agent/context.py 按完整轮次裁剪；config.py 暴露上下文窗口、输出上限和安全余量；第二轮复用 ID 并携带首轮消息，新会话隔离；系统提示与当前输入仍超预算时返回 413，不请求上游。
- 验证与评审结论：先证实第二轮 ID 不复用、预算未生效两个失败，再逐切片实现；3 项测试通过。预算计数使用 LangChain count_tokens_approximately，chars_per_token=1.0，对中文比默认每四字符一个 token 更保守，但仍是近似。
- 用户拒绝或纠偏：无新增。
- 翻车与返工：新增失败场景测试发现 length 截断被当作成功（符合尚未实现该边界的预期）；下一切片显式核对结束原因，避免保存半轮。
- 下一步：异常回滚、并发和客户端断流。

## 2026-11-19 · 阶段 6：流式生命周期与失败回滚

- 用户关键原话：Q9“使用推荐”，约定失败或断流丢弃半轮、同会话并发拒绝。
- 关键产出：src/support_agent/streaming.py 持续监听断流，发送成功后提交历史，并清理上游流；模型失败、空回复、缺少正常结束标记、length 截断或工具调用结束均产生 SSE error，不产生 done、不保存本轮。
- 验证与评审：上游限流等错误不回传上游原文。原始 ASGI 测试在上游仍阻塞时已经观察到 token，证明并非缓存整段后输出；同会话并发先失败（200），实现后为 409。
- 用户拒绝或纠偏：无新增。
- 翻车与返工：
  - ASGI 2.4 断流测试超时；Context7 与已安装 Starlette 源码确认新分支等待 send 失败，不主动监听 receive，导致上游静默期不能及时取消。补充统一监听，保留指定 FastAPI 技术栈。
  - 初版统一监听将 HTTPX 在正常结束后产生的 disconnect 误判为取消，导致多轮回归失败。改为最终 ASGI body 发送成功后立即提交历史，断流先发生则取消流。已重新运行覆盖正常结束与中途断流的测试。
- 下一步：完成回归并实现售后结构化输出切片。

## 2026-11-19 · 阶段 7：售后结构化输出

- 用户关键原话：Q2、Q7、Q8 均“使用推荐”；固定三字段、json_schema / json_mode 可配、歧义不擅选。
- 关键产出：src/support_agent/schemas.py 定义严格字段约束；prompts.py 添加售后提取 PromptTemplate；POST /after-sales/extract 使用 with_structured_output。两种方式的模拟请求验证通过，均不发送 tools。
- 验证与评审结论：先验证接口不存在（404），再实现后通过两种方式的成功测试。新增非 JSON、缺字段、非法分类、数字订单号、额外字段等错误样例，要求 502 且不回传上游内容；上游拒绝 response_format 时只发一次请求，不切换方式。
- 用户拒绝或纠偏：无新增，保持不执行真实模型验收。
- 翻车与返工：
  - include_raw=True 不能捕获 OpenAI SDK 在解析阶段提前抛出的全部异常；初版错误被归为泛化 upstream_error。已区分 Pydantic / 解析 / 截断异常，保留清晰错误类别。
  - 查已锁定 LangChain 源码发现某些旧模型名会触发自动改用 function_calling。新增测试证实该路径会尝试工具请求；在构建 structured runnable 时将明确的自动覆盖警告转为错误，阻断降级。保留用户的指定方法，用户可自行明确选择 json_mode。
  - 流式生命周期修正后，包含两种 ASGI 断流情况的 11 项对话测试已全部通过。
- 下一步：回归边界、配置与使用说明、标注评估资产及最终评审。

## 2026-11-19 · 阶段 8：配置、输入边界与评估资产

- 用户关键原话：“模型、base URL 以及密钥，都是用 .env 文件配置”；“真实验收我会自己来做”。
- 关键产出：
  - .env.example、README.md 及 examples/*.json；包含单进程启动、Windows curl 文件请求、两轮对话、售后提取、SSE 终止事件与错误约定。
  - evals/after_sales.jsonl 共 12 条标注及 evals/README.md 的对话评审判据；scripts/check_eval_data.py 离线校验标签能由固定输出契约表达，已通过 12 条。未访问模型，未声称 Prompt 语义通过。
  - HTTP 输入拒绝空白、多余字段、非字符串和非法 UUID；未知会话不访问上游。UTF-8 临时 .env 测试确认地址、模型、密钥由文件加载并进入 OpenAI 请求，使用保留域名模拟，未读取用户密钥。
- 评审结论：公开接口边界补全；结构化方法和版本兼容保护均具备回归用例。文档明确近似 token 计数与单进程内存的适用范围。
- 用户拒绝或纠偏：无新增，真实语义验收仍由用户自行负责。
- 翻车与返工：新增“模型完成但最后一个响应 body 仍阻塞”场景发现会话提前解锁（并发返回 200）；已将解锁统一放到响应清理阶段，修复后返回 409。Ruff 首轮发现导入顺序及行长问题，正在统一格式后完整回归。
- 下一步：完整测试、静态检查、需求逐项评审及交付记录。

## 2026-11-19 · 阶段 9：完整回归与兼容性评审

- 用户关键原话：继续满足“GPT、Claude、DeepSeek、Ollama 都能换着接”，应用侧保持统一 OpenAI 协议；不替换用户指定技术。
- 关键产出：
  - 第一轮完整 pytest：29 passed，61.45 秒。Ruff lint、格式检查及 12 条标注离线校验通过。
  - API 文档显式声明 POST /chat 的 text/event-stream；OpenAI SDK 作为直接使用的依赖写入清单并重新锁定。
  - Context7 和锁定源码确认 ChatOpenAI 会把 max_tokens 改写为 max_completion_tokens。新增 OUTPUT_TOKEN_PARAMETER 配置，分别用标准字段或 extra_body 明确发送所选参数；请求仅含一个输出上限字段。两个专项测试通过。
- 评审结论：指定技术、无工具调用、失败回滚、预算和字段契约均满足设计；真实上游需具备所配置的 OpenAI 协议能力。标注校验只证明数据契约，不报告模型效果。未发现需要改变技术选型的矛盾。
- 用户拒绝或纠偏：无新增，未调用真实模型。
- 翻车与返工：新增旧参数模式测试先因 max_tokens 不存在失败，补充显式配置后通过；该兼容修正不改变模型接入协议。最后补充项已进入最终回归。
- 下一步：完成最终测试与交付记录。

## 2026-11-19 · 阶段 10：交付

- 用户关键原话：完结交付“功能演示命令、测试结果、dev-notes 路径”；真实验收由用户自行完成。
- 关键产出：
  - 功能与启动/演示命令：README.md；请求文件 examples/chat-first.json、examples/chat-second.json、examples/after-sales.json；配置模板 .env.example。
  - 正式 spec：https://github.com/DCZJtAWhH/support-agent/issues/1 ；计划 docs/plans/0001-pure-chat.md；领域词汇 GLOSSARY.md。
  - 最终 uv run --locked pytest -q --tb=short：31 passed，62.42 秒。
  - Ruff lint：All checks passed；格式：14 files already formatted。
  - 离线标注校验：12 labeled examples validated offline；未评估模型语义。
  - uv lock --check 通过；git diff --check 通过；确认 .env / .env.local 被 Git 忽略。
- 评审结论：本章实现和已约定自动化验证完成；历史在单进程内存中、token 计数为近似值、上游需提供所选协议能力等边界已写明。没有执行真实模型或真实商家业务操作。改动保留在工作区，未提交或推送代码。
- 用户拒绝或纠偏：继续尊重由用户自行真实验收的边界，无其他新增要求。
- 翻车与返工：阶段 6、7、8、9 已分别记录断流竞态、解析错误归类、提前解锁和 token 参数兼容修复；最终回归无失败。

## 2026-11-19 · 阶段 11：用户报告提取失败，建立诊断回路

- 用户关键原话：返回 upstream_error / “模型调用失败，请检查 .env 中的地址、模型、密钥与上游能力。”，但“/chat 接口没问题”。
- 关键产出：读取 diagnosing-bugs 技能；不调用真实上游。仅检查 .env 中非敏感项，当前为 deepseek-flash、STRUCTURED_OUTPUT_METHOD=json_schema、OUTPUT_TOKEN_PARAMETER=max_completion_tokens。保留用户配置，不擅自切换方式。
- 诊断回路：uv run --locked pytest -q tests/test_extraction.py::test_unsupported_structured_method_never_falls_back --tb=short；模拟上游拒绝 response_format，断言应明确提示结构化方式被拒绝。已运行并失败：实际 upstream_error，期望 structured_output_rejected。该回路复现的是确定存在的错误信息丢失问题，不冒称已复现用户真实上游的全部原因。
- 用户拒绝或纠偏：仍遵守此前“真实验收我会自己来做”，未发起真实模型请求。
- 翻车与返工：首次交付的通用错误隐藏了上游状态和错误类别，无法区分基础连接问题与结构化能力问题；需要补充安全诊断。
- 下一步：对照官方接口分析候选原因，修复错误分类并添加回归测试；真实错误需由用户重试后返回明确分类确认。

## 2026-11-19 · 阶段 12：确认配置与 Chat Completions 能力不符并修正

- 用户关键原话：继续排查“/chat 接口没问题”但售后提取失败；没有新增授权真实验收。
- 关键产出：
  - 向用户列出三个可证伪候选：结构化方式不支持、schema 约束被拒绝、非流式 SDK 解析失败。
  - Context7 查询 DeepSeek 官方 Chat Completions 接口定义：response_format.type 只枚举 text / json_object；JSON Output 指南使用 deepseek-flash 和 json_object。Responses API 的 json_schema 是另一个端点能力，不能混用。
  - 仅通过本地解析确认 .env 配置的主机为官方 api.deepseek.com，未输出 URL、查询参数或密钥，未发起模型请求。已将本机 .env 的 STRUCTURED_OUTPUT_METHOD 从 json_schema 明确改为 json_mode；其余配置保留。
  - API 错误分类补充 structured_output_rejected、upstream_status、SDK error_type 和可执行的配置提示；不自动降级、不回传上游响应原文。README 与 .env.example 记录 DeepSeek 的正确方式。
- 验证与评审：最小诊断回路先红后绿（1 passed），证实错误信息丢失已修正；补充模拟 DeepSeek 能力边界的差异回归与非结构化错误的脱敏回归。
- 用户拒绝或纠偏：保持用户自行真实验收；本次只做官方文档核实、本地配置修正及模拟回归，不能声称真实调用已成功。
- 翻车与返工：原先通用错误导致用户看到连接配置提示，但实际配置与所用端点的结构化能力不匹配。首次实现已经提供 json_mode，却未在 DeepSeek 示例中明确说明，应补充供应商端点级别说明。
- 下一步：运行完整回归，交付重启后复试命令及测试结果。

## 2026-11-19 · 阶段 13：本次修复验证与交付

- 用户关键原话：售后提取返回通用 upstream_error，但 /chat 正常。
- 关键产出与评审：确认 DeepSeek Chat Completions 的模式配置不符，已明确设置本机 json_mode；提供端点级文档提示和更具体的脱敏诊断。模拟差异和脱敏专项测试 5 passed；完整回归 36 passed（73.37 秒）；Ruff lint、格式检查及 12 条标注离线校验通过。无临时 DEBUG 插桩或遗留临时诊断脚本。
- 用户拒绝或纠偏：未执行真实上游请求，用户需要重启服务后自行重试提取接口。真实成功结果尚未验证。
- 翻车与返工：修复了首次实现的错误诊断不足，避免再次将结构化参数拒绝笼统解释为地址或密钥问题；其他 401 / 429 / 500 错误保留准确的状态与类别，且不泄露响应原文。
