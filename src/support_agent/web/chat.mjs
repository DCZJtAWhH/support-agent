import { readChatStream } from "./stream.mjs";

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
  const bubble = document.createElement("p");
  bubble.className = "bubble";
  bubble.textContent = text;
  content.append(name, bubble);
  article.append(avatar, content);
  messages.append(article);
  followConversation(true);
  return { article, content, bubble };
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
  const writer = createTypewriter(assistant.bubble);
  const request = { controller: new AbortController(), writer };
  activeRequest = request;
  updateControls();
  status.textContent = "正在等待客服回复…";
  document.querySelector("#conversation-title").textContent = message;
  try {
    const body = { message };
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
    });
    await writer.drain();
    if (activeRequest !== request) return;
    status.textContent = "您可以继续追问，前面的内容会接着聊";
  } catch (error) {
    if (activeRequest !== request) return;
    writer.stop();
    assistant.article.classList.add("failed");
    if (!assistant.bubble.textContent) assistant.bubble.textContent = "本轮未能收到完整回复。";
    const state = document.createElement("p");
    state.className = "message-state";
    state.textContent = "本轮未完成";
    assistant.content.append(state);
    failedTurn = { message, ...assistant, expired: error.status === 404 };
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
