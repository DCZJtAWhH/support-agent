# 第六章实施计划

1. 结构化问题理解：同一 LLM 调用完成历史指代消解和标准问法归一；明确问题原样保留；分类独立输出 intent/confidence，含其他兜底及边界示例。复用聊天上游，分流默认使用该上游已提供的 deepseek-v4-pro，不启用小模型降级。
2. 检索：核心意图才生成严格 queries 数组；保留完整问题，多路召回后按知识块去重，强制检索政策；不修改入库布局。
3. Workflow：resolve → classify → route；退款退货/售后统一进入订单选择/核实 → 检索政策 → 证据闸 → 主力 Agent。缺订单时用原生 interrupt 保存状态，原请求点选后 Command 恢复；订单号不由 LLM 提取或猜测，仅接受明确订单号、用户选项及同会话明确订单。
4. HTTP/持久化：校验点选归属、过期轮次、请求重放；固定原因的退款申请独立显式提交，按原请求幂等保存，提交不代表退款完成。
5. 前端：保留当前聊天布局，流内订单卡片、理解结果和执行轨迹；点选自动回填并续跑；退款申请为固定类目下拉的简单表单。
6. 验证：后端按确认的三个边界逐片 red/green；标注集检验真实模型多轮切换、其他兜底、完整问题透传与扩写；浏览器在桌面/手机执行缺订单退款、选择续跑、退款提交和后续物流。

## 实施约束

- 无新增运行组件，沿用现有模型、数据库、检索和 checkpointer。现有订单/物流上游为演示数据，选择器和查询共享同一演示订单源，页面与回复明确标注。
- 各阶段及时追加 dev-notes/ch06.md，记录原话、产物、纠偏与返工。
- 所修改 src 文件逐个函数检查中文注释。

## 接口与版本查证

实现阶段通过 Context7 查证下列接口，并结合本地安装版本及实际调用验证：

| 依赖 | 本地版本 | 本章使用的接口 / 文档 |
| --- | --- | --- |
| LangGraph | 1.2.24 | [interrupt / Command 恢复](https://docs.langchain.com/oss/python/langgraph/interrupts)，沿用 astream custom 与 SQLite checkpointer；未采用文档新增的 stream_events v3 |
| LangChain Core / OpenAI | 1.8.3 | [模型结构化输出](https://docs.langchain.com/oss/python/langchain/models)，bind response_format + astream；json_schema/json_mode 按既有配置明确选择 |
| FastAPI | 0.165.0 | [Pydantic 请求体](https://fastapi.tiangolo.com/tutorial/body/)，固定表单枚举及 HTTPException |
| SQLAlchemy | 2.0.58 | Context7 2.0.55 同系列文档，async_sessionmaker.begin、select、唯一约束；[异步接口](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html) |
| Pydantic | 2.14.3 | Context7 2.14.1 同系列文档，严格字段、extra=forbid、model_validate_json |
| Playwright | 1.65.2 | Context7 1.64.1 异步浏览器示例；本地 inspect 核对 expect_response 和 select_option(label=...)，真实 Chromium 验证 |
| PyMilvus | 2.6.18 | 最终验证阶段补查[固定版本 SDK](https://github.com/milvus-io/pymilvus/tree/v2.6.18)并 inspect hybrid_search；沿用既有向量适配器，无新增 Milvus API |
| HTTPX | 0.28.1 | 最终验证阶段补查[固定版本文档](https://github.com/encode/httpx/blob/0.28.1/docs/quickstart.md)，核对沿用的 json POST、超时、raise_for_status |

Context7 的 Milvus 综合文档指向 v3.0.x，未直接采用其中的新接口；以本地2.6.18 SDK定义和既有适配器为准。HTTPX 的 Context7 查证晚于脚本初稿，此处如实记录；其实际接口与项目原有调用方式一致。
