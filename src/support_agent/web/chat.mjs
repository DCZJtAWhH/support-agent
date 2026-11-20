import { readChatStream } from "./stream.mjs";
import { setupKnowledgeUpload } from "./knowledge-upload.mjs";

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const messages = document.querySelector("#messages");
const scroller = document.querySelector("#conversation-scroll");
const status = document.querySelector("#status");
const notice = document.querySelector("#notice");
const retry = document.querySelector("#retry");
const welcome = document.querySelector("#welcome");
const newConversation = document.querySelector("#new-conversation");
let conversationId = null;
let activeRequest = null;
let failedTurn = null;

/** 显示本轮检索时的来源原文快照，避免知识更新后引用悄悄变化。 */
function openSource(source) {
  let dialog = document.querySelector("#source-dialog");
  if (!dialog) {
    dialog = document.createElement("dialog");
    dialog.id = "source-dialog";
    dialog.className = "source-dialog";
    dialog.setAttribute("aria-labelledby", "source-title");
    document.body.append(dialog);
  }
  dialog.replaceChildren();
  const header = document.createElement("div");
  header.className = "source-header";
  const title = document.createElement("h2");
  title.id = "source-title";
  title.textContent = `引用 [${source.citation}]`;
  const close = document.createElement("button");
  close.type = "button";
  close.textContent = "关闭 ×";
  close.addEventListener("click", function closeSource() {
    // 原生 dialog 恢复焦点到打开它的引用按钮。
    dialog.close();
  });
  header.append(title, close);
  const path = document.createElement("p");
  path.className = "source-path";
  path.textContent = source.section_path;
  const origin = document.createElement("p");
  origin.className = "source-origin";
  origin.textContent = source.source_ref;
  const text = document.createElement("pre");
  text.className = "source-text";
  text.textContent = source.answer || source.text;
  const link = document.createElement("a");
  link.textContent = "查看原文与所在章节 ↗";
  link.href = `/knowledge/chunks/${source.chunk_id}/original?version=${encodeURIComponent(source.content_hash)}#chunk-${source.chunk_id}`;
  link.target = "_blank";
  link.rel = "noopener";
  const note = document.createElement("p");
  note.className = "source-origin";
  note.textContent = "上方为回答时引用的原文；打开原文可查看当前版本。";
  dialog.append(header, path, origin, text, link, note);
  dialog.showModal();
}

/** 仅将存在来源映射的编号替换为按钮，其他内容始终以纯文本渲染。 */
function renderCitations(assistant, sources) {
  const text = assistant.reply.textContent;
  assistant.reply.replaceChildren();
  let offset = 0;
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    assistant.reply.append(document.createTextNode(text.slice(offset, match.index)));
    const source = sources.find(function findCitation(item) {
      // 编号和来源由本轮快照绑定，跨回答互不混用。
      return item.citation === Number(match[1]);
    });
    if (source) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "citation";
      button.textContent = match[0];
      button.setAttribute("aria-label", `查看引用 ${source.citation}：${source.section_path}`);
      button.addEventListener("click", function showCitation() {
        // 通过闭包保留当前回答的来源版本。
        openSource(source);
      });
      assistant.reply.append(button);
    } else assistant.reply.append(document.createTextNode(match[0]));
    offset = match.index + match[0].length;
  }
  assistant.reply.append(document.createTextNode(text.slice(offset)));
}

/** 为完整回答提供一次性满意度反馈；按请求标识仅在当前浏览器本地采集。 */
function addFeedback(assistant, requestId) {
  const key = `support-agent:feedback:${requestId}`;
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(key)); } catch { /* 存储不可用时仍允许本次页面采集。 */ }
  let choice = ["up", "down"].includes(saved?.value) ? saved.value : null;
  const group = document.createElement("div");
  group.className = "answer-feedback";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "这段回答对您有帮助吗");
  const state = document.createElement("span");
  state.setAttribute("role", "status");
  const buttons = [];

  /** 一次选定后锁定两项，重试同一回答也恢复已采集状态。 */
  function updateFeedback() {
    for (const button of buttons) {
      button.disabled = Boolean(choice);
      button.classList.toggle("selected", button.dataset.value === choice);
      button.setAttribute("aria-pressed", String(button.dataset.value === choice));
    }
    state.textContent = choice ? "已反馈" : "回答有帮助吗";
  }

  for (const [value, icon, label] of [["up", "👍", "满意"], ["down", "👎", "不满意"]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.value = value;
    button.textContent = icon;
    button.setAttribute("aria-label", label);
    button.addEventListener("click", function chooseFeedback() {
      // 本次选择不可撤销；后续数据飞轮可从这些本地记录导出。
      if (choice) return;
      choice = value;
      const record = { request_id: requestId, conversation_id: conversationId, value,
        created_at: new Date().toISOString() };
      try { localStorage.setItem(key, JSON.stringify(record)); } catch { /* 本轮内存锁定仍有效。 */ }
      updateFeedback();
    });
    buttons.push(button);
    group.append(button);
  }
  group.append(state);
  assistant.bubble.append(group);
  updateFeedback();
}

/** 只在顾客靠近底部时跟随新文字，阅读上文时不强制跳回底部。 */
function followConversation(force = false) {
  if (force || scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 120) {
    scroller.scrollTop = scroller.scrollHeight;
  }
}

/** 根据输入和本轮状态更新按钮；请求期间允许先编辑下一条消息。 */
function updateControls() {
  send.disabled = Boolean(activeRequest) || !input.value.trim();
  send.textContent = activeRequest ? "正在回复…" : "发送消息 ↗";
  retry.disabled = Boolean(activeRequest);
  retry.hidden = failedTurn?.expired || !failedTurn;
}

/** 用纯文本创建消息气泡，防止顾客或模型内容被当成 HTML 执行。 */
function addMessage(role, text = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "我" : "聊";
  const content = document.createElement("div");
  content.className = "message-content";
  const name = document.createElement("p");
  name.className = "message-name";
  name.textContent = role === "user" ? "您" : "智能客服";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const tools = document.createElement("div");
  tools.className = "tool-trace";
  tools.hidden = true;
  tools.setAttribute("aria-label", "本轮工具轨迹");
  const reply = document.createElement("span");
  reply.className = "reply";
  reply.textContent = text;
  bubble.append(tools, reply);
  content.append(name, bubble);
  article.append(avatar, content);
  messages.append(article);
  followConversation(true);
  return { article, content, bubble, reply, tools };
}

/** 在客服气泡中按调用标识更新工具徽章，保留本轮已完成或失败的轨迹。 */
function updateToolBadge(assistant, data) {
  const labels = { query_order: "查订单", query_product: "查商品", query_logistics: "查物流", query_faq: "查常见问题", create_ticket: "登记人工工单" };
  const states = { running: "执行中", succeeded: "已完成", failed: "失败" };
  let badge = Array.from(assistant.tools.children).find(function matchesCall(item) {
    // 按工具调用标识复用徽章，同名工具的不同申请仍分别显示。
    return item.dataset.callId === data.tool_call_id;
  });
  if (!badge) {
    badge = document.createElement("span");
    badge.dataset.callId = data.tool_call_id;
    assistant.tools.append(badge);
  }
  badge.className = `tool-badge ${data.state}`;
  badge.textContent = `${labels[data.name] || data.name} · ${states[data.state]}${data.reused ? " · 已复用" : ""}`;
  badge.title = data.name;
  const code = document.createElement("code");
  code.textContent = data.name;
  badge.append(code);
  assistant.tools.hidden = false;
  status.textContent = data.state === "running" ? `正在${labels[data.name] || data.name}…` : "正在根据查询结果组织回答…";
  followConversation();
}

/** 将上游文本增量拆为字符依次显示；队列排空后才开放下一轮发送。 */
function createTypewriter(bubble) {
  const queue = [];
  let position = 0;
  let timer = null;
  let finish = null;
  const rendered = document.createTextNode("");
  bubble.append(rendered);

  /** 每次追加少量字符；长回复加快速度，保持动画并缩短等待时间。 */
  function tick() {
    const count = queue.length - position > 150 ? 3 : 1;
    rendered.appendData(queue.slice(position, position + count).join(""));
    position = Math.min(position + count, queue.length);
    followConversation();
    if (position < queue.length) timer = setTimeout(tick, 16);
    else {
      timer = null;
      if (finish) finish();
    }
  }

  /** 接收任意大小的增量，空闲时启动下一次渲染。 */
  function append(text) {
    queue.push(...Array.from(text));
    if (timer === null && position < queue.length) timer = setTimeout(tick, 16);
  }

  /** 等待已收到的文字全部显示后再确认本轮完成。 */
  function drain() {
    if (position === queue.length) return Promise.resolve();
    // 保存排空回调，供最后一个字符显示后调用。
    return new Promise(function waitForQueue(resolve) {
      finish = resolve;
    });
  }

  /** 错误或新建会话时停止计时，保留已收到的文本并释放等待者。 */
  function stop() {
    clearTimeout(timer);
    timer = null;
    rendered.appendData(queue.slice(position).join(""));
    position = queue.length;
    if (finish) finish();
  }

  return { append, drain, stop };
}

/** 发送一轮对话或重试失败轮次，沿用 meta 给出的会话 ID 以衔接服务端上下文。 */
async function sendMessage(message, previous = null) {
  if (activeRequest || !message.trim()) return;
  notice.hidden = true;
  failedTurn = null;
  welcome.hidden = true;
  if (!previous) {
    addMessage("user", message);
    input.value = "";
  } else previous.article.remove();
  const assistant = addMessage("assistant");
  assistant.bubble.classList.add("streaming");
  const writer = createTypewriter(assistant.reply);
  const requestId = previous?.requestId || crypto.randomUUID();
  let sources = [];
  const request = { controller: new AbortController(), writer };
  activeRequest = request;
  updateControls();
  status.textContent = "正在等待客服回复…";
  document.querySelector("#conversation-title").textContent = message;
  try {
    const body = { message, request_id: requestId };
    if (conversationId) body.conversation_id = conversationId;
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: request.controller.signal,
    });
    // 保存会话信息，将文本交给逐字渲染器。
    await readChatStream(response, function handleEvent(event, data) {
      if (activeRequest !== request) return;
      if (event === "meta") {
        conversationId = data.conversation_id;
        document.querySelector("#session-label").textContent = "当前对话";
      }
      if (event === "token") {
        status.textContent = "客服正在回复…";
        writer.append(data.text);
      }
      if (event === "tool_status") updateToolBadge(assistant, data);
      if (event === "sources") sources = data.sources;
    });
    await writer.drain();
    if (activeRequest !== request) return;
    renderCitations(assistant, sources);
    addFeedback(assistant, requestId);
    status.textContent = "您可以继续追问，前面的内容会接着聊";
  } catch (error) {
    if (activeRequest !== request) return;
    writer.stop();
    assistant.article.classList.add("failed");
    if (!assistant.reply.textContent) assistant.reply.textContent = "本轮未能收到完整回复。";
    const state = document.createElement("p");
    state.className = "message-state";
    state.textContent = "本轮未完成";
    assistant.content.append(state);
    failedTurn = { message, requestId, ...assistant, expired: error.status === 404 };
    document.querySelector("#notice-text").textContent = error instanceof TypeError
      ? "无法连接客服服务，请检查网络后重试。"
      : error.message || "本轮回复失败，请稍后重试。";
    notice.hidden = false;
    status.textContent = error.status === 404 ? "请点击「新建对话」重新开始" : "可以重试本轮，或修改问题后发送";
  } finally {
    assistant.bubble.classList.remove("streaming");
    if (activeRequest === request) {
      activeRequest = null;
      updateControls();
      followConversation();
      input.focus({ preventScroll: true });
    }
  }
}

/** 提交当前输入；空白输入与生成期间的重复提交不会创建消息。 */
function handleSubmit(event) {
  event.preventDefault();
  void sendMessage(input.value.trim());
}

/** Enter 发送，Shift+Enter 换行；中文输入法确认候选词时不提交。 */
function handleKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (input.value.trim()) form.requestSubmit();
  }
}

/** 通过新请求重试原问题，复用原顾客气泡并替换不完整的客服回复。 */
function retryFailedTurn() {
  if (failedTurn && !failedTurn.expired) void sendMessage(failedTurn.message, failedTurn);
}

/** 中止当前请求并清空当前页面状态，之后发送不再带旧会话 ID。 */
function resetConversation() {
  const request = activeRequest;
  activeRequest = null;
  request?.controller.abort();
  request?.writer.stop();
  conversationId = null;
  failedTurn = null;
  messages.replaceChildren();
  input.value = "";
  welcome.hidden = false;
  notice.hidden = true;
  status.textContent = "说说您遇到的问题吧";
  document.querySelector("#conversation-title").textContent = "购物与售后咨询";
  document.querySelector("#session-label").textContent = "新对话";
  scroller.scrollTop = 0;
  updateControls();
  input.focus({ preventScroll: true });
}

/** 使用示例问题发起真实客服请求。 */
function handleSuggestion(event) {
  const button = event.target.closest("[data-prompt]");
  if (button) void sendMessage(button.dataset.prompt);
}

form.addEventListener("submit", handleSubmit);
input.addEventListener("input", updateControls);
input.addEventListener("keydown", handleKeydown);
retry.addEventListener("click", retryFailedTurn);
newConversation.addEventListener("click", resetConversation);
welcome.addEventListener("click", handleSuggestion);
updateControls();
setupKnowledgeUpload();
