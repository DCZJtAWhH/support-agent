# create_app 方法逐项说明

对应源码：[src/support_agent/app.py](../../src/support_agent/app.py)，方法位于第 26–140 行。下文按源码顺序解释各项配置、执行步骤和接口行为；行号对应本文编写时的实现。

`create_app` 是应用工厂：负责组装并返回一个 FastAPI 应用。它把配置、模型客户端、数据库、聊天服务、静态资源和 HTTP 接口连接起来，具体的聊天编排和数据库读写则交给其他模块。

## 1 执行时机

理解这个方法，首先要区分四个阶段：定义内部函数时，并不会立即执行它们的函数体。

| 阶段 | 执行内容 |
| --- | --- |
| 调用 `create_app()` | 获取配置，定义生命周期函数，创建 FastAPI 实例，挂载静态资源，注册路由，返回应用。 |
| 应用启动 | FastAPI 执行 `lifespan` 中 `yield` 之前的代码，创建客户端和服务，初始化数据库。 |
| 处理请求 | 根据 URL 执行 `chat_page`、`extract`、`chat` 或 `conversation_history`。 |
| 应用退出 | 从 `yield` 后继续执行，关闭数据库连接池，再退出 HTTP 客户端的上下文，释放连接。 |

因此，`return app` 并不表示模型和数据库已经初始化；这些资源要等生命周期启动后才可使用。[FastAPI 生命周期文档](https://fastapi.tiangolo.com/advanced/events/)说明了 `yield` 前后分别执行启动与关闭逻辑的机制。

## 2 方法参数与配置

对应第 26–33 行：

```python
def create_app(
    settings: Settings | None = None, upstream_client: httpx.AsyncClient | None = None
) -> FastAPI:
    config = settings or Settings()
```

| 项目 | 作用 |
| --- | --- |
| `settings` | 可选的配置对象。传入时直接使用，便于测试或按需构建应用；未传入时创建 `Settings()`。 |
| `Settings()` | 根据 [config.py](../../src/support_agent/config.py) 从环境变量、UTF-8 编码的 `.env` 等来源加载并校验配置。环境变量优先于 `.env`。 |
| `upstream_client` | 可选的异步 HTTP 客户端，供模型访问上游。测试可以注入带模拟传输的客户端，避免真实模型请求。 |
| `-> FastAPI` | 声明返回值类型为 FastAPI 应用实例。 |
| `config` | 保存本次应用使用的配置。内部函数通过闭包访问它，因此每个应用实例可以持有自己的配置。 |

`Settings()` 在调用工厂时执行。缺少必填配置，或上下文预算不合法，会在这里报错；数据库连接是否可用则要到启动初始化时才会检验。

外部传入的 `upstream_client` 由调用方关闭，当前方法只负责关闭自己创建的客户端。

## 3 生命周期与共享服务

### 3.1 生命周期入口

对应第 35–39 行：

```python
@asynccontextmanager
async def lifespan(app):
    with httpx.Client() as sync_client:
        async with httpx.AsyncClient() as owned_client:
```

| 项目 | 作用 |
| --- | --- |
| `@asynccontextmanager` | 将包含 `yield` 的异步生成器函数包装为异步上下文管理器，供 FastAPI 管理启动和退出。 |
| `lifespan(app)` | 接收当前应用实例，在其生命周期内创建和释放共享资源。 |
| `httpx.Client()` | 创建同步 HTTP 客户端，传给 ChatOpenAI 的同步调用通道。当前接口主要使用异步通道，但这里同时明确提供两类客户端。 |
| `httpx.AsyncClient()` | 创建应用自己管理的异步 HTTP 客户端，用于未注入外部客户端时访问模型。 |
| `with` 和 `async with` | 让客户端在上下文退出时自动关闭，不需要在每个请求后手动关闭。 |

当前写法即使注入了外部异步客户端，也仍会创建并最终关闭 `owned_client`，只是模型不会选择它作为异步传输。

### 3.2 ChatOpenAI 的每个参数

对应第 40–60 行。`ChatOpenAI(...)` 创建的是访问模型 API 的包装对象，并不会在构造时生成一段回答；真正的模型调用发生在后续的 `ainvoke` 或聊天服务的流式调用中。

| 参数 | 当前取值 | 作用 |
| --- | --- | --- |
| `model` | `config.openai_model` | 指定请求的模型名。 |
| `base_url` | `config.openai_base_url` | 指定模型 API 的基础地址，可指向提供兼容协议的上游。仅更换地址并不能保证其支持全部接口能力。 |
| `api_key` | `config.openai_api_key.get_secret_value()` | 从 `SecretStr` 取出真实密钥，交给 SDK 用于认证。`SecretStr` 的作用是避免普通展示时直接暴露密钥。 |
| `http_client` | `sync_client` | 指定同步请求使用的 HTTP 客户端。 |
| `http_async_client` | `upstream_client or owned_client` | 优先使用调用方注入的异步客户端，否则使用应用自建的客户端。 |
| `timeout` | `config.model_timeout_seconds` | 设置 SDK 请求超时，默认 60 秒。这是 HTTP 请求超时配置，不是对整段聊天流程设置一个统一的 60 秒截止时间。 |
| `max_tokens` | 根据输出参数配置选择上限或 `None` | 设置模型输出 token 上限；与下一项配合，控制最终请求使用哪一个字段名。默认上限为 1024。 |
| `extra_body` | 根据输出参数配置选择字典或 `None` | 将额外字段合并进上游 JSON 请求体，本项目用它显式发送旧式 `max_tokens` 字段。 |
| `max_retries` | `0` | 关闭模型 SDK 的自动重试。工具执行的重试由聊天服务另行管理，不受这个参数直接控制。 |
| `stream_usage` | `False` | 不额外请求流式 token 用量统计，减少对上游兼容能力的要求；这不会关闭回答内容的流式传输。 |
| `use_responses_api` | `False` | 明确使用 Chat Completions API，避免自动改用 Responses API。 |

上述通用参数可参考 [ChatOpenAI 官方接口说明](https://reference.langchain.com/python/langchain-openai/ChatOpenAI)。下述字段改写行为还对应仓库安装版本的实现及 [输出参数测试](../../tests/test_chat.py)。

### 3.3 为什么输出上限分成两个分支

当前 LangChain 版本会把普通 `max_tokens` 参数改写成上游请求的 `max_completion_tokens`。但部分兼容上游只接受 `max_tokens`，所以这里通过 `output_token_parameter` 显式选择：

| `output_token_parameter` | 传给 ChatOpenAI 的参数 | 最终发送的输出上限字段 |
| --- | --- | --- |
| `max_completion_tokens` | `max_tokens=config.max_output_tokens`，`extra_body=None` | `max_completion_tokens` |
| `max_tokens` | `max_tokens=None`，`extra_body={"max_tokens": config.max_output_tokens}` | `max_tokens` |

这里的 `None` 表示不设置该普通参数，不表示向上游发送一个值为 `null` 的输出上限字段。两个分支的目的都是只发送所选的一个字段。

### 3.4 数据库与服务组装

对应第 61–68 行：

| 代码 | 作用 |
| --- | --- |
| `database = Database(config.database_url.get_secret_value())` | 从数据库地址创建异步 SQLAlchemy 引擎和会话工厂。生产使用 MySQL，测试可配置 SQLite。 |
| `await database.initialize(seed=True)` | 创建缺失的数据表，并补充尚不存在的 FAQ 演示数据；不覆盖已存在的 FAQ，也不等同于执行数据库结构迁移。 |
| `app.state.repository = ChatRepository(database)` | 创建仓储对象，封装会话、消息、FAQ 和人工工单等持久化操作。 |
| `app.state.chat = ChatService(app.state.model, app.state.repository, config)` | 把模型、仓储和配置交给聊天服务，由它管理单轮工具决策、回复生成、持久化与恢复。 |
| `yield` | 标记启动完成。生命周期在这里暂停，应用可以开始处理请求。 |
| `finally: await database.close()` | 退出时释放数据库连接池；若初始化过程在进入 `try` 后失败，也会执行清理。 |

`app.state` 是应用实例上的共享存储位置。这里放入三个对象：`model`、`repository` 和 `chat`，供后续路由复用；它们不是每次请求重新创建的。

正常退出时，释放顺序为：数据库连接池 → 自建异步 HTTP 客户端 → 同步 HTTP 客户端。外部注入的异步客户端不在这个关闭链条中。

相关实现：[database.py](../../src/support_agent/database.py)、[repositories.py](../../src/support_agent/repositories.py)、[chat.py](../../src/support_agent/chat.py)。

## 4 应用实例与静态资源

对应第 70–72 行：

| 代码 | 作用 |
| --- | --- |
| `FastAPI(title="电商智能客服", lifespan=lifespan)` | 创建应用，设置 API 文档中的标题，并注册刚才定义的生命周期管理函数。 |
| `Path(__file__).with_name("web")` | 将当前 `app.py` 文件路径的末尾替换为 `web`，定位同目录下的前端资源目录。资源路径基于模块位置，不依赖命令行启动目录。 |
| `StaticFiles(directory=web_directory)` | 创建静态文件服务，用于读取该目录中的 CSS、JavaScript 等资源。 |
| `app.mount("/static", ..., name="static")` | 将静态文件服务挂到 `/static` URL 前缀。`name="static"` 是路由名称，可用于反向生成 URL，不是磁盘目录名。 |

例如，`web` 目录下的某个资源文件可以通过 `/static/文件名` 访问。首页 `/` 则由下一段单独返回。

## 5 首页接口

对应第 74–77 行：

```python
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def chat_page():
    return FileResponse(web_directory / "index.html", headers={"Cache-Control": "no-cache"})
```

| 项目 | 作用 |
| --- | --- |
| `@app.get("/")` | 注册网站根路径的 GET 接口。装饰器在创建应用时执行，函数体在访问首页时执行。 |
| `response_class=FileResponse` | 声明该路由使用文件响应。 |
| `include_in_schema=False` | 将首页从 OpenAPI 接口文档中隐藏；不会禁止浏览器访问它。 |
| `web_directory / "index.html"` | 使用 `Path` 的 `/` 运算符拼接首页文件路径。 |
| `Cache-Control: no-cache` | 允许缓存，但要求复用缓存前向服务器验证，便于前端页面更新后生效；不等同于 `no-store`。 |

首页、静态资源和 API 由同一个应用提供，页面可以使用同源请求访问聊天接口。

## 6 售后信息提取接口

对应第 79–121 行。这段代码完成一次独立的售后描述提取，返回普通 JSON；不读取聊天历史，也不在这里创建人工工单或写入会话。

### 6.1 路由与输入输出约束

```python
@app.post("/after-sales/extract", response_model=AfterSalesRequest)
async def extract(body: ExtractionRequest):
```

| 项目 | 作用 |
| --- | --- |
| `POST /after-sales/extract` | 接收一次售后信息提取请求。 |
| `body: ExtractionRequest` | 使用 Pydantic 校验 JSON 请求体，要求提供 `description`，并拒绝多余字段。描述会去掉首尾空白，长度必须为 1–16384 个字符。 |
| `response_model=AfterSalesRequest` | 声明并校验返回结构，同时生成相应的 OpenAPI 文档。模型调用阶段也会使用这个类型解析结果。 |

固定的返回字段定义在 [schemas.py](../../src/support_agent/schemas.py)：

| 字段 | 含义与约束 |
| --- | --- |
| `order_id` | 唯一明确的订单号，类型为字符串或 `null`；缺失或多订单无法唯一确定时为 `null`。 |
| `request_type` | 诉求类型，只能是退款、退货退款、换货、维修、其他或未知。 |
| `expected_solution` | 顾客明确表达的期望方案，类型为字符串或 `null`；不代表商家已批准。 |

三个字段都必须出现，不允许额外字段；允许 `null` 不等于允许省略字段。

### 6.2 生成 Schema 并预留预算

对应第 82–83 行：

```python
schema = json.dumps(AfterSalesRequest.model_json_schema(), ensure_ascii=False)
schema_cost = len(schema) if config.structured_output_method == "json_schema" else 0
```

`model_json_schema()` 根据输出模型生成字段约束字典，`json.dumps` 将其转为 JSON 文本；`ensure_ascii=False` 保留中文描述，避免转换成 Unicode 转义序列。

Schema 随后会写进系统提示。在 `json_schema` 模式下，约束还会作为结构化参数随请求发送，所以代码额外扣除 `len(schema)` 作为这部分的预算估算；`json_mode` 没有额外的结构化 Schema 参数，因此此处扣除 0。两种模式都仍然包含系统提示中的 Schema。

这里按字符数预留成本，属于近似估算，不是模型实际 tokenizer 的精确计数。

### 6.3 组装消息并检查输入预算

对应第 84–92 行：

| `fit_messages` 的实参 | 作用 |
| --- | --- |
| `SystemMessage(AFTER_SALES_EXTRACTION.format(schema=schema))` | 将 Schema 填入售后提取提示模板，明确字段、歧义处理和禁止编造信息等规则，作为系统消息。 |
| `[]` | 历史消息为空，只分析本次描述。 |
| `HumanMessage(body.description)` | 将顾客提交的描述作为用户消息。 |
| `config.input_token_budget - schema_cost` | 给消息正文使用的剩余输入预算。 |

`input_token_budget` 的计算式是：

```text
输入预算 = 上下文窗口 - 最大输出 token 数 - 安全余量
默认值   = 8192 - 1024 - 256 = 6912
```

[fit_messages](../../src/support_agent/context.py) 会保留系统提示和当前输入，并按近似 token 数检查预算。通常它也能裁剪旧轮次，但这里没有历史，所以两条必要消息超过预算时，只能抛出 `ContextBudgetExceeded`。

`except ContextBudgetExceeded` 将该异常转为 HTTP 413，并通过 `detail=str(exc)` 返回原因。`raise ... from exc` 保留异常因果链，便于服务端定位问题；这一分支不会发起模型请求。

### 6.4 选择结构化输出方式

对应第 94–103 行：

```python
options = {"strict": True} if config.structured_output_method == "json_schema" else {}
```

| 方式 | 上游约束与本地校验 |
| --- | --- |
| `json_schema` | 通过 `response_format` 携带 JSON Schema，并设置 `strict=True`，要求支持该能力的上游遵守字段约束；返回后仍由 Pydantic 校验。 |
| `json_mode` | 使用 `response_format.type=json_object` 请求 JSON 对象；字段规则主要由提示词表达，再由 Pydantic 校验。这里不传 `strict` 参数。 |

`with_structured_output(...)` 在共享模型基础上创建本次提取使用的调用包装器：

| 参数 | 作用 |
| --- | --- |
| `AfterSalesRequest` | 指定期望的字段结构以及结果解析类型。 |
| `method=config.structured_output_method` | 明确选择配置的结构化方式，不依赖库的默认方式。 |
| `include_raw=True` | 同时保留原始模型消息、解析结果和解析错误，以便检查内容和结束原因。 |
| `**options` | 展开字典参数，仅在 `json_schema` 模式传入 `strict=True`。 |

官方文档说明了 [`include_raw` 返回原始消息和解析结果](https://docs.langchain.com/oss/python/langchain/models)的用途。当前接口对两种配置都要求上游支持相应的 `response_format` 能力。

### 6.5 阻止静默切换为工具调用

对应第 95–108 行：

| 代码 | 作用 |
| --- | --- |
| `warnings.catch_warnings()` | 在这个作用域内临时调整警告处理方式，退出后恢复。 |
| `warnings.filterwarnings("error", message=".*Overriding.*", category=UserWarning)` | 把匹配 `Overriding` 的 `UserWarning` 转为异常，拦截 LangChain 自动覆盖结构化方式的行为。 |
| 内层 `except UserWarning` | 将被提升为异常的警告转换为 `ModelResponseError`，给出明确的配置提示。 |

某些模型名称会触发 LangChain 警告，并尝试把 `json_schema` 改为 `function_calling`。此处让包装器构建直接失败，要求显式选择兼容方式。它也不会自动从 `json_schema` 重试成 `json_mode`。

这段限制作用于售后提取接口；聊天服务本身有独立的工具调用流程。

### 6.6 调用模型并验收结果

对应第 109–116 行：

```python
result = await structured.ainvoke(messages)
```

`ainvoke` 异步发起一次模型调用，等待完整结果；该接口不会逐段向前端发送回复。由于设置了 `include_raw=True`，返回字典包含：

| 键 | 内容 |
| --- | --- |
| `raw` | 原始模型消息，包含响应元数据。 |
| `parsed` | 解析得到的 `AfterSalesRequest` 对象，解析失败时可能为 `None`。 |
| `parsing_error` | 解析异常，解析成功时为 `None`。 |

随后按短路逻辑检查三个条件，任意一个成立都判定失败：

1. `parsing_error is not None`：返回内容没有通过解析或字段约束校验。
2. `parsed is None`：没有得到可返回的结构化对象。
3. `raw.response_metadata.get("finish_reason") != "stop"`：上游未明确报告正常结束；包括被长度限制截断、以其他原因结束或缺少标记。

失败时抛出 `ModelResponseError`。全部通过才返回 `result["parsed"]`，由 FastAPI 转为 JSON 响应。

### 6.7 统一模型错误响应

对应第 117–121 行。外层 `except Exception` 捕获包装器构建、上游调用、解析和结果验收中的异常，统一转换为 HTTP 502：

```python
detail=model_error(exc, structured_output_method=config.structured_output_method)
```

[model_error](../../src/support_agent/errors.py) 将异常映射为中文提示和错误码，例如 `upstream_timeout`、`structured_output_rejected`、`invalid_model_response` 或 `upstream_error`。传入当前结构化方式，可以在上游拒绝 `response_format` 时提供对应的配置提示。

返回体的外层是 FastAPI 的 `detail` 字段，例如：

```json
{
  "detail": {
    "code": "invalid_model_response",
    "message": "模型未返回符合售后字段约束的完整 JSON，请检查上游能力。"
  }
}
```

这里不会直接向调用方回传上游响应原文。前面的预算错误位于独立的 `try` 中，因此它保持 HTTP 413，不会被这个分支转成 502。

## 7 聊天接口

对应第 123–130 行：

```python
@app.post(
    "/chat",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat(body: ChatRequest):
    return await app.state.chat.respond(body)
```

| 项目 | 作用 |
| --- | --- |
| `POST /chat` | 注册客服聊天入口。 |
| `response_class=StreamingResponse` | 声明路由使用流式响应。实际响应对象由聊天服务返回。 |
| `responses={200: ...}` | 在 OpenAPI 文档中标明成功响应的媒体类型为 `text/event-stream`；该声明本身不会生成 SSE 事件。 |
| `body: ChatRequest` | 接收 `message`，以及可选的 `conversation_id` 和 `request_id`。消息需要通过文本约束，两个标识需要通过 UUID 校验，多余字段会被拒绝。 |
| `await app.state.chat.respond(body)` | 将请求委托给启动时创建的聊天服务，取得流式响应对象。 |

`conversation_id` 用于复用客服对话；`request_id` 用于识别同一轮请求，使重试能够恢复已保存的工具检查点。缺省标识由聊天服务生成。

聊天服务负责会话与请求校验、上下文预算、一次工具决策、最多两个工具的顺序执行、持久化和最终回复。它返回 [ChatStreamResponse](../../src/support_agent/streaming.py)，这是 `StreamingResponse` 的子类，负责 SSE 响应头、断流监听和发送后的清理。

流中会出现 `meta`、`token`、`tool_status`、`done` 或 `error` 等事件。`await respond(...)` 主要完成前置处理并返回响应对象，后续内容随着响应发送逐步生成。流开始之后的模型错误通过 SSE `error` 事件告知前端，不能简单理解为所有失败都会改成 HTTP 502。

人工工单及工具执行记录可能在最终回复失败前已经持久化。失败轮次保留审计，但不会作为后续完整对话上下文；这符合 [工具副作用持久化 ADR](../adr/0001-persist-tool-effects-across-chat-retries.md)。

## 8 会话历史接口

对应第 132–138 行：

```python
@app.get("/conversations/{conversation_id}")
async def conversation_history(conversation_id: UUID):
    result = await app.state.repository.read_conversation(str(conversation_id))
    if result is None:
        raise HTTPException(404, detail="会话不存在。")
    return result
```

| 项目 | 作用 |
| --- | --- |
| `{conversation_id}` | 从 URL 中提取会话标识。 |
| `conversation_id: UUID` | 让 FastAPI 校验并解析 UUID；格式不合法时通常直接返回 HTTP 422，不会执行函数体。 |
| `str(conversation_id)` | 转为仓储接口使用的字符串形式。 |
| `read_conversation(...)` | 查询会话基础信息、消息审计和人工工单列表。 |
| `result is None` | 表示数据库中没有对应会话，转为 HTTP 404。 |
| `return result` | 返回可序列化字典，包含 `id`、`user_id`、`status`、`created_at`、`messages` 和 `tickets`。 |

这里的 `messages` 包含失败轮次及工具关联信息，因此它是持久化审计视图，与后续送给模型的完整对话历史范围不同。当前接口尚未接入用户身份认证和会话归属校验，使用的是演示会话读取逻辑。

## 9 返回应用

第 140 行的 `return app` 将已经注册生命周期、静态资源和路由的应用交给服务器。仓库使用以下方式启动：

```powershell
uv run --locked uvicorn support_agent.app:create_app --factory --app-dir src --host 127.0.0.1 --port 8000 --workers 1
```

`--factory` 表示服务器需要调用 `create_app` 获取应用，`--app-dir src` 指定模块查找目录。`--workers 1` 与当前聊天服务在内存中记录正在执行的会话和请求相匹配；这些并发占用记录不跨进程共享。

## 10 接口速查

| 入口 | 输出 | 主要职责 |
| --- | --- | --- |
| `GET /` | HTML 文件 | 返回客服聊天页。 |
| `/static/...` | 静态文件 | 提供前端资源。 |
| `POST /after-sales/extract` | JSON | 从本次售后描述中提取三个固定字段。 |
| `POST /chat` | SSE 事件流 | 执行客服对话、工具操作及回复。 |
| `GET /conversations/{conversation_id}` | JSON | 查询会话、消息审计和人工工单。 |

常见失败的分工是：请求格式或字段校验失败由 FastAPI 返回 422；未知会话返回 404；聊天请求冲突由聊天服务返回 409；输入预算不足返回 413；售后提取的模型相关异常返回 502；聊天流开始后的异常通过 SSE `error` 事件返回。
