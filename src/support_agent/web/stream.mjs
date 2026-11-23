/** 按 SSE 行规则解析事件，保留跨网络分片的行，并兼容 LF、CRLF 与 CR。 */
export function createEventParser(onEvent) {
  let line = "";
  let event = "message";
  let data = [];
  let skipLF = false;

  /** 处理一行；空行提交完整事件，注释和不认识的字段按 SSE 规则忽略。 */
  function acceptLine() {
    if (line === "") {
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
      event = "message";
      data = [];
    } else if (!line.startsWith(":")) {
      const colon = line.indexOf(":");
      const field = colon < 0 ? line : line.slice(0, colon);
      let value = colon < 0 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      if (field === "data") data.push(value);
    }
    line = "";
  }

  /** 接收解码后的文本；流末尾没有空行的事件不提交，避免把断流误判为完成。 */
  function push(text) {
    for (const character of text) {
      if (skipLF && character === "\n") {
        skipLF = false;
        continue;
      }
      skipLF = character === "\r";
      if (character === "\n" || character === "\r") acceptLine();
      else line += character;
    }
  }

  return { push };
}

/** 消费 POST 返回的 SSE；只有匹配会话且含非空回复的 done 才算成功。 */
export async function readChatStream(response, onEvent) {
  if (!response.ok) {
    const error = new Error("请求未成功，请稍后重试。");
    error.status = response.status;
    if (response.status === 404) error.message = "对话已失效，请新建对话后重新发送。";
    else if (response.status === 409) error.message = "上一条回复仍在处理中，请稍后重试。";
    else if (response.status === 413) error.message = "这条消息过长，请缩短后重新发送。";
    else if (response.status === 422) error.message = "消息不符合要求，请检查后重新发送。";
    throw error;
  }
  if (!response.headers.get("content-type")?.includes("text/event-stream") || !response.body) {
    throw new Error("服务未返回对话内容，请稍后重试。");
  }
  let conversationId = null;
  let done = false;
  let reply = "";
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const reader = response.body.getReader();

  /** 校验事件顺序和字段，流内 error 立即中止，未知事件留给后续协议扩展。 */
  function acceptEvent(event, data) {
    if (!["meta", "tool_status", "sources", "actions", "token", "done", "error",
      "understanding", "order_required", "refund_form", "workflow", "retrieval_queries"].includes(event)) return;
    if (done || !data || typeof data !== "object") throw new Error("回复格式异常，请重试本轮。");
    if (event === "error") throw new Error(data.message || "本轮回复失败，请重试。");
    if (event === "meta") {
      if (conversationId || typeof data.conversation_id !== "string" || !data.conversation_id) {
        throw new Error("会话信息异常，请新建对话。");
      }
      conversationId = data.conversation_id;
    } else if (!conversationId) {
      throw new Error("回复缺少会话信息，请重试。");
    } else if (event === "understanding") {
      if (typeof data.question !== "string" || typeof data.intent !== "string" ||
          !Number.isFinite(data.confidence) || data.confidence < 0 || data.confidence > 1) {
        throw new Error("问题理解格式异常，请重试本轮。");
      }
    } else if (event === "order_required") {
      if (!Array.isArray(data.orders) || !data.orders.length || data.orders.some(function invalidOrder(order) {
        // 卡片只显示服务端订单快照，订单编号不能由页面内容推断。
        return typeof order.order_id !== "string" || typeof order.product_name !== "string" ||
          typeof order.status !== "string" || !Number.isFinite(order.total);
      })) throw new Error("订单列表异常，请重试本轮。");
    } else if (event === "refund_form") {
      if (typeof data.order_id !== "string" || !Array.isArray(data.reasons) ||
          !data.reasons.length || data.reasons.some(function invalidReason(reason) {
            // 原因只作为下拉选项显示，提交仍由后端固定枚举校验。
            return typeof reason !== "string" || !reason;
          })) throw new Error("退款申请格式异常，请重试本轮。");
    } else if (event === "workflow" || event === "retrieval_queries") {
      if (event === "workflow" && typeof data.node !== "string") throw new Error("处理状态异常。");
      if (event === "retrieval_queries" && (!Array.isArray(data.queries) ||
          data.queries.some(function invalidQuery(query) {
            // 检索查询仅用于显示本轮可观察轨迹。
            return typeof query !== "string";
          }))) throw new Error("检索查询格式异常。");
    } else if (event === "tool_status") {
      if (typeof data.name !== "string" || typeof data.tool_call_id !== "string" ||
          !["running", "succeeded", "failed"].includes(data.state)) {
        throw new Error("工具状态格式异常，请重试本轮。");
      }
    } else if (event === "sources") {
      if (!Array.isArray(data.sources) || data.sources.some(function invalidSource(source) {
        // 来源只接受服务端快照结构，链接由客户端根据整数主键构造。
        return !Number.isSafeInteger(source.chunk_id) || source.chunk_id < 1 ||
          !Number.isSafeInteger(source.citation) || source.citation < 1 ||
          typeof source.section_path !== "string" || typeof source.text !== "string";
      })) throw new Error("引用来源格式异常，请重试本轮。");
    } else if (event === "actions") {
      if (!Array.isArray(data.actions) || data.actions.some(function invalidAction(action) {
        // 仅接收两个无副作用建议，动作执行必须来自独立点击。
        return !["handoff", "create_ticket"].includes(action);
      })) throw new Error("可选操作格式异常，请重试本轮。");
    } else if (event === "token") {
      if (typeof data.text !== "string") throw new Error("回复格式异常，请重试本轮。");
      reply += data.text;
    } else {
      if (data.conversation_id !== conversationId || !reply.trim()) {
        throw new Error("回复未完整结束，请重试本轮。");
      }
      done = true;
    }
    onEvent(event, data);
  }

  const parser = createEventParser(acceptEvent);
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      parser.push(decoder.decode(chunk.value, { stream: true }));
    }
    parser.push(decoder.decode());
    if (!done) throw new Error("连接已中断，本轮未完成，请重试。");
    return conversationId;
  } finally {
    try { await reader.cancel(); } finally { reader.releaseLock(); }
  }
}
