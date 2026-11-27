# 接口与流式协议

应用地址以 `http://127.0.0.1:8010` 为例，字段约束以运行时 `/docs`、`/openapi.json` 和源码 Schema 为准。当前没有用户身份鉴权，会话目录固定属于 `demo-user`。

## 客服对话

`POST /chat` 返回 `text/event-stream`。首次请求示例：

```json
{"message":"查询订单1001物流轨迹","request_id":"c03ecbf1-1639-4519-8f56-ea41df57e766"}
```

| 字段 | 含义 |
| --- | --- |
| `message` | 必填，去除首尾空白后 1–16384 字符；仍受 token 输入预算限制。 |
| `conversation_id` | 可选 UUID，续聊时使用服务端返回的会话标识。 |
| `request_id` | 可选 UUID；建议调用方发送前生成并保存，重试时复用。 |
| `category` | 可选知识分类，1–255 字符。 |
| `selected_order_id` | 订单选择暂停后回填选中的订单号。 |
| `ticket_confirmation` | 工单暂停后的严格布尔确认值，不能传字符串。 |
| `ticket_tool_call_id` | 工单暂停时返回的工具调用标识。 |

不接受额外字段。订单点选、工单确认或取消时，必须复用原 `message`、`conversation_id`、`request_id` 以及原品类；工单内容取服务端 checkpoint，客户端不能另传描述。暂停请求的 SSE 也可能正常以 `done` 结束，业务是否完成需结合暂停事件判断。

### SSE 事件

```text
event: meta
data: {"conversation_id":"会话UUID","request_id":"请求UUID"}

event: token
data: {"text":"您好"}

event: done
data: {"conversation_id":"会话UUID","request_id":"请求UUID"}
```

| 事件 | 用途 |
| --- | --- |
| `meta` | 返回本次会话与请求标识。 |
| `understanding` | 展示当前问题理解和意图。 |
| `workflow` / `retrieval_queries` | 展示工作流节点及检索查询。 |
| `tool_status` | 工具运行、成功或失败状态；包含调用标识等执行信息。 |
| `sources` | 本轮引用来源与原文快照。 |
| `token` | 可向用户显示的文本片段；知识答案先经过门控。 |
| `actions` / `refund_form` | 后续动作与退款申请表单。 |
| `order_required` / `ticket_required` | 等待订单点选或工单确认。 |
| `done` | 本次流正常结束。 |
| `error` | 本次流失败，不再发送 `done`。 |

SSE 数据按 JSON 编码；解析器应按事件边界处理，不能假设网络分块就是事件或 token。HTTP 流建立后的错误仍可能是 HTTP 200，应检查终止事件；没有正常终止的断流不能当成成功。

### 重试与状态

完成请求的重复提交回放已保存结果；失败请求凭原 `request_id` 恢复已执行的工具效果。首次 `meta` 未送达时，只有调用方已保存的请求标识才能可靠恢复。

同标识输入冲突或同会话并发返回 409；未知会话返回 404；Schema 无效返回 422；输入预算不足可返回 413。上游流中故障通过 `error` 返回。已确认工单不会因断流撤销，失败回答不进入后续完整上下文。

## 业务动作与售后提取

| 接口 | 请求与行为 |
| --- | --- |
| `POST /actions/ticket` | `{conversation_id, request_id}`；用于已完成投诉轮次的页面建单确认，内容取该轮服务端记录。 |
| `POST /actions/refund` | `{conversation_id, request_id, reason}`；订单取原轮次已核实快照，只登记待核实申请。 |
| `POST /after-sales/extract` | `{description}`；返回 `order_id`、`request_type`、`expected_solution`。 |

退款原因只允许“不想要了”“商品质量问题”“收到商品破损”“商品与描述不符”“发错货或少件”“其他”。同请求同输入幂等，改变已提交原因返回 409。

售后提取的订单号缺失或无法唯一确定时为 `null`；诉求类型为退款、退货退款、换货、维修、其他、未知；未表达具体期望方案时为 `null`。提取结果不代表资格核实或批准，模型故障返回 502。

## 会话与知识

| 接口 | 用途 |
| --- | --- |
| `GET /api/conversations` | 按新在前列出演示用户会话。 |
| `GET /api/conversations/{id}/messages` | 回载用户与客服原文。 |
| `GET /conversations/{id}` | 完整持久化消息、工单／退款及工具调用审计。 |
| `POST /knowledge/upload` | multipart 的 `file`；完整导入返回 201，原文保存但索引未完成返回 503。 |
| `GET /knowledge/chunks/{id}` | 返回当前有效知识原文与摘要。 |
| `GET /knowledge/chunks/{id}/original?version=内容摘要` | 展示来源文档；可用 `#chunk-{id}` 定位，版本不符时提示已更新。 |

上传限制及失败恢复见[知识维护](knowledge.md)。引用快照代表回答时证据；原文接口显示当前有效内容，不能将两者混为同一版本。

## 反馈、审核与评估

| 接口 | 请求或用途 |
| --- | --- |
| `POST /api/feedback` | `{conversation_id, request_id, value}`，`value` 为 `up` 或 `down`。 |
| `POST /api/flywheel/process` | 手动整理待归并的问题池，可能调用真实模型。 |
| `GET /api/review-queue` | 可按 `status` 筛选 `pending`、`publishing`、`approved`、`rejected`。 |
| `GET /api/review-queue/{id}` | 查看缺口、用户原问和召回快照。 |
| `POST /api/review-queue/{id}/decision` | `{decision, approved_answer}`；决定为 `approved` / `rejected`，通过必须填写核准答案。 |
| `GET /api/costs` | 按意图汇总真实 token 与未知用量计数。 |
| `GET /api/evaluations` | 报告与逐题证据，支持 `query_type`、`difficulty` 联合筛选。 |
| `GET /api/evaluations/status` | 轻量任务进度。 |
| `POST /api/evaluations/run` | 启动完整真实评估，返回 202；已有任务或知识待同步时返回 409。 |
| `GET /api/evaluations/runs` | 评估指标与版本历史。 |

审核发布的索引同步失败返回 503，应使用原核准答案重试，不能替换为另一份答案继续同一发布。
