# 第七章：当前会话上下文管理

范围：当前会话三层滑窗及追加式摘要、完整 checkpoint、只读会话 API 和原页面侧栏；不增加长期记忆或历史语义检索。

1. TDD：配置预算及上下文公开接口。统一中文 token 估算；动态倒推历史额度，70%/30%，启动检查。演示配置独立核验 5650/3954/1695。
2. TDD：会话仓储及摘要后台任务。边界为 `id <= summary_upto_msg_id` 摘要，`summary_upto_msg_id < id < layer1_from_msg_id` 层 2，`id >= layer1_from_msg_id` 层 1；边界只能前移。新工具结果仅写 checkpoint，保留旧消息审计和业务副作用幂等保护。SQL 统一放根目录 `ch7.sql`。
3. TDD：LangGraph messages 使用 add_messages，节点仅增量返回；模型投影单独保存。背景梗概和证据合成当前用户消息之后的一条非 system 消息。history_ctx 每轮输出，model_ctx 每次主力模型调用输出，写 log/app.log。
4. TDD：只读 `/api/conversations`、`/api/conversations/{id}/messages`，固定现有演示用户范围。沿用页面视觉和技术栈增加侧栏、回载和续聊；静默处理侧栏请求失败。
5. 验收：测试套件、默认和演示配置连续 20+ 轮、异步不阻塞、订单回忆、Prompt 标注样例、浏览器桌面/移动/失败降级。保存实际结果和功能演示命令。按用户后续纠偏，最终验收全部使用离线模型 HTTP 替身，不再进行真实模型测试；静态标注检查不冒充语义质量评估。

实现注意：ADR 0001 的工具审计存储位置由本章显式要求调整为 checkpoint；原有 tickets/refund 的数据库幂等规则保留。旧 checkpoint 需要从已完成原文迁移，不能让旧裁剪历史继续充当完整历史。
