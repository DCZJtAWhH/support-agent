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
    if (!["meta", "token", "done", "error"].includes(event)) return;
    if (done || !data || typeof data !== "object") throw new Error("回复格式异常，请重试本轮。");
    if (event === "error") throw new Error(data.message || "本轮回复失败，请重试。");
    if (event === "meta") {
      if (conversationId || typeof data.conversation_id !== "string" || !data.conversation_id) {
        throw new Error("会话信息异常，请新建对话。");
      }
      conversationId = data.conversation_id;
    } else if (!conversationId) {
      throw new Error("回复缺少会话信息，请重试。");
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
