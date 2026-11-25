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
const conversationList = document.querySelector("#conversation-list");
let conversationId = null;
let activeRequest = null;
let failedTurn = null;
let loadingConversation = false;
let selectionRevision = 0;
let listRevision = 0;

/** 刷新当前用户的会话目录；请求失败保持现状，不向聊天区显示错误。 */
async function refreshConversations() {
  const revision = ++listRevision;
  try {
    const response = await fetch("/api/conversations");
    if (!response.ok) return;
    const rows = await response.json();
    if (revision !== listRevision || !Array.isArray(rows)) return;
    const fragment = document.createDocumentFragment();
    for (const row of rows) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.conversationId = row.id;
      button.className = "conversation-item";
      const preview = document.createElement("span");
      preview.textContent = row.preview || "新对话";
      button.append(preview);
      if (row.has_summary) {
        const badge = document.createElement("small");
        badge.className = "summary-badge";
        badge.textContent = "已摘要";
        button.append(badge);
      }
      fragment.append(button);
    }
    conversationList.replaceChildren(fragment);
    updateControls();
  } catch {
    // 侧栏是辅助入口，网络失败不能影响发送或正在显示的回复。
  }
}

/** 有效回载后切换会话并恢复已完成轮次的反馈入口；迟到历史不会覆盖新对话。 */
async function selectConversation(id) {
  if (activeRequest) return;
  const revision = ++selectionRevision;
  loadingConversation = true;
  updateControls();
  try {
    const response = await fetch(`/api/conversations/${encodeURIComponent(id)}/messages`);
    if (!response.ok) return;
    const history = await response.json();
    if (revision !== selectionRevision || !Array.isArray(history)) return;
    resetConversation();
    conversationId = id;
    welcome.hidden = history.length > 0;
    for (const row of history) {
      if (!['user', 'assistant'].includes(row.role)) continue;
      const item = addMessage(row.role, row.content);
      if (row.state === "failed") {
        item.article.classList.add("failed");
        const note = document.createElement("p");
        note.className = "message-state";
        note.textContent = "本轮未完成";
        item.content.append(note);
      }
      if (row.role === "assistant" && row.state === "completed" && row.position === 1000000) {
        addFeedback(item, row.request_id);
      }
    }
    document.querySelector("#conversation-title").textContent = history.find(function firstQuestion(row) {
      // 标题仅作首问预览，气泡仍显示完整原文。
      return row.role === "user";
    })?.content || "购物与售后咨询";
    document.querySelector("#session-label").textContent = "当前对话";
    status.textContent = "您可以继续追问，前面的内容会接着聊";
    setSidebarOpen(false);
    updateControls();
    followConversation(true);
  } catch {
    // 回载失败保留当前聊天，不显示全局错误或创建新请求。
  } finally {
    if (revision === selectionRevision) {
      loadingConversation = false;
      updateControls();
    }
  }
}

/** 控制移动端历史抽屉及遮罩，保持展开状态可由辅助技术读取。 */
function setSidebarOpen(open) {
  document.body.classList.toggle("sidebar-open", open);
  document.querySelector("#conversation-toggle").setAttribute("aria-expanded", String(open));
  document.querySelector("#sidebar-backdrop").hidden = !open;
}

/** 通过事件委托切换目录内的会话。 */
function handleConversationClick(event) {
  const button = event.target.closest("[data-conversation-id]");
  if (button) void selectConversation(button.dataset.conversationId);
}

/** 在移动端打开或关闭历史目录。 */
function toggleSidebar() {
  setSidebarOpen(!document.body.classList.contains("sidebar-open"));
}

/** 按 Escape 收起历史目录并将键盘焦点返回打开按钮。 */
function handleSidebarEscape(event) {
  if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) {
    setSidebarOpen(false);
    document.querySelector("#conversation-toggle").focus();
  }
}

/** 在原客服气泡内显示点选订单，保留原请求标识以恢复暂停的子流程。 */
function addOrderPicker(assistant, orders, message, requestId, sourceConversationId) {
  const group = document.createElement("div");
  group.className = "order-picker";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "选择要处理的订单");
  const heading = document.createElement("strong");
  heading.textContent = "选择一笔订单";
  group.append(heading);
  for (const order of orders) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "order-card";
    const title = document.createElement("strong");
    title.textContent = order.product_name;
    const detail = document.createElement("span");
    detail.textContent = `订单 ${order.order_id} · ${order.status} · ¥${order.total}`;
    const action = document.createElement("span");
    action.className = "order-card-action";
    action.textContent = "选择此订单 →";
    card.append(title, detail, action);
    card.addEventListener("click", function selectOrder() {
      // 点选只回填服务端列表中的编号；继续同一请求，避免重新猜测意图。
      if (activeRequest || conversationId !== sourceConversationId) return;
      for (const button of group.querySelectorAll("button")) button.disabled = true;
      addMessage("user", `选择订单 ${order.order_id} · ${order.product_name}`);
      void sendMessage(message, { ...assistant, requestId }, order.order_id);
    });
    group.append(card);
  }
  assistant.bubble.append(group);
}

/** 展示服务端工单预览；只回传确认和原调用标识，不接受页面修改工单内容。 */
function addTicketPreview(assistant, preview, message, requestId, sourceConversationId) {
  const card = document.createElement("section");
  card.className = "ticket-preview";
  card.setAttribute("aria-label", "工单预览");
  const heading = document.createElement("strong");
  heading.textContent = "请核对工单";
  const type = document.createElement("p");
  type.className = "ticket-preview-type";
  type.textContent = `工单类型 · ${preview.ticket_type}`;
  const description = document.createElement("p");
  description.className = "ticket-preview-description";
  description.textContent = preview.description;
  const note = document.createElement("p");
  note.className = "ticket-preview-note";
  note.textContent = "确认后登记为待处理工单，由人工继续跟进。";
  const actions = document.createElement("div");
  actions.className = "ticket-preview-actions";
  for (const [confirmed, label] of [[true, "确认提交"], [false, "取消"]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = confirmed ? "ticket-confirm" : "ticket-cancel";
    button.addEventListener("click", function resumeTicket() {
      // 继续原请求，由后端从 checkpoint 取工单参数；取消同样需进入引擎留下拒绝审计。
      if (activeRequest || conversationId !== sourceConversationId) return;
      for (const control of actions.querySelectorAll("button")) control.disabled = true;
      void sendMessage(message, { ...assistant, requestId }, null, confirmed, preview.tool_call_id);
    });
    actions.append(button);
  }
  card.append(heading, type, description, note, actions);
  assistant.bubble.append(card);
}

/** 用户提交时才采集固定原因，并将申请凭据显示在原聊天气泡中。 */
function addRefundForm(assistant, data, requestId, sourceConversationId) {
  const form = document.createElement("form");
  form.className = "refund-form";
  const heading = document.createElement("strong");
  heading.textContent = `退款申请 · 订单 ${data.order_id}`;
  const label = document.createElement("label");
  label.htmlFor = `refund-reason-${requestId}`;
  label.textContent = "退款原因";
  const select = document.createElement("select");
  select.id = label.htmlFor;
  select.required = true;
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "请选择退款原因";
  empty.disabled = true;
  empty.selected = true;
  select.append(empty);
  for (const reason of data.reasons) {
    const option = document.createElement("option");
    option.value = reason;
    option.textContent = reason;
    select.append(option);
  }
  const submit = document.createElement("button");
  submit.type = "submit";
  submit.textContent = "提交退款申请";
  const result = document.createElement("p");
  result.className = "refund-result";
  result.setAttribute("role", "status");
  result.textContent = "演示申请将保存为待核实，不会直接执行退款。";
  form.append(heading, label, select, submit, result);
  form.addEventListener("submit", async function submitRefund(event) {
    // 原因无需在对话中追问，订单由后端使用本轮已核实的快照。
    event.preventDefault();
    if (activeRequest || submit.disabled || conversationId !== sourceConversationId) return;
    submit.disabled = true;
    select.disabled = true;
    submit.textContent = "正在提交…";
    try {
      const response = await fetch("/actions/refund", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: sourceConversationId, request_id: requestId,
          reason: select.value }),
      });
      const saved = await response.json();
      if (!response.ok) throw new Error(saved.detail || "提交失败，请重试。");
      if (conversationId !== sourceConversationId) return;
      submit.textContent = "已提交申请";
      result.textContent = `申请 ${saved.application_no} 已登记，等待核实。`;
    } catch (error) {
      submit.disabled = false;
      select.disabled = false;
      submit.textContent = "提交退款申请";
      result.textContent = error.message || "提交失败，请重试。";
    }
    followConversation();
  });
  assistant.bubble.append(form);
}

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

/** 为完整回答采集反馈；后端确认后才锁定，使用稳定轮次标识防止错轮入池。 */
function addFeedback(assistant, requestId) {
  const key = `support-agent:feedback:${requestId}`;
  const feedbackConversationId = conversationId;
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(key)); } catch { /* 存储不可用时仍允许本次页面采集。 */ }
  let choice = saved?.server_confirmed && ["up", "down"].includes(saved?.value) ? saved.value : null;
  let sending = false;
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
      button.disabled = Boolean(choice) || sending;
      button.classList.toggle("selected", button.dataset.value === choice);
      button.setAttribute("aria-pressed", String(button.dataset.value === choice));
    }
    state.textContent = sending ? "正在提交反馈…" : choice ? "已反馈" : "回答有帮助吗";
  }

  for (const [value, icon, label] of [["up", "👍", "满意"], ["down", "👎", "不满意"]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.value = value;
    button.textContent = icon;
    button.setAttribute("aria-label", label);
    button.addEventListener("click", async function chooseFeedback() {
      // 失败保留重试入口；异步期间切换会话也不改变反馈归属。
      if (choice || sending) return;
      sending = true;
      updateFeedback();
      try {
        const response = await fetch("/api/feedback", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: requestId, conversation_id: feedbackConversationId, value }),
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(typeof error.detail === "string" ? error.detail : "反馈提交失败，请重试。");
        }
        choice = value;
        const record = { request_id: requestId, conversation_id: feedbackConversationId, value,
          server_confirmed: true, created_at: new Date().toISOString() };
        try { localStorage.setItem(key, JSON.stringify(record)); } catch { /* 后端已经保存。 */ }
      } catch (error) {
        state.textContent = error.message;
      } finally {
        sending = false;
        if (choice) updateFeedback();
        else for (const option of buttons) option.disabled = false;
      }
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
  input.disabled = loadingConversation;
  send.disabled = Boolean(activeRequest) || loadingConversation || !input.value.trim();
  send.textContent = activeRequest ? "正在回复…" : "发送消息 ↗";
  retry.disabled = Boolean(activeRequest);
  retry.hidden = failedTurn?.expired || !failedTurn;
  for (const button of conversationList.querySelectorAll("button")) {
    button.disabled = Boolean(activeRequest);
    if (button.dataset.conversationId === conversationId) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }
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

/** 将后端建议呈现为独立按钮，只有明确点击才执行对应动作。 */
function addActions(assistant, actions, requestId, sourceConversationId) {
  if (!actions.length) return;
  const group = document.createElement("div");
  group.className = "answer-actions";
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "可选的后续处理");
  const result = document.createElement("p");
  result.className = "action-result";
  result.setAttribute("role", "status");
  for (const action of new Set(actions)) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.action = action;
    button.textContent = action === "handoff" ? "转人工" : "建工单";
    button.addEventListener("click", async function confirmAction() {
      // 绑定原会话和原请求；新建会话后迟到的响应不能写入新的聊天页。
      if (activeRequest || conversationId !== sourceConversationId) {
        result.textContent = "请等待当前回复完成后再操作。";
        return;
      }
      if (action === "handoff") {
        button.disabled = true;
        addMessage("assistant", "已转接人工客服");
        const greeting = addMessage("assistant", "您好，我是客服小猫，请问有什么可以帮您的");
        greeting.article.querySelector(".message-name").textContent = "客服小猫";
        return;
      }
      if (!window.confirm("确认将本轮问题登记为待处理工单？")) return;
      button.disabled = true;
      button.textContent = "正在建单…";
      result.textContent = "";
      try {
        const response = await fetch("/actions/ticket", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ conversation_id: sourceConversationId, request_id: requestId }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "建单失败，请重试。");
        if (conversationId !== sourceConversationId) return;
        button.textContent = "已建工单";
        result.textContent = `已登记工单 ${data.ticket_no}，等待人工跟进。`;
      } catch (error) {
        button.disabled = false;
        button.textContent = "建工单";
        result.textContent = error.message || "建单失败，请重试。";
      }
    });
    group.append(button);
  }
  assistant.bubble.append(group, result);
}

/** 在客服气泡中按调用标识更新工具徽章，保留本轮已完成或失败的轨迹。 */
function updateToolBadge(assistant, data) {
  const labels = { query_order: "查订单", query_product: "查商品", query_logistics: "查物流", query_faq: "查常见问题", create_ticket: "登记人工工单" };
  const states = { running: "执行中", waiting: "等待确认", succeeded: "已完成", failed: "失败" };
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
async function sendMessage(message, previous = null, selectedOrderId = null,
  ticketConfirmation = null, ticketCallId = null) {
  if (activeRequest || loadingConversation || !message.trim()) return;
  for (const card of document.querySelectorAll(".order-picker button, .ticket-preview button")) card.disabled = true;
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
  let actions = [];
  let orders = null;
  let refundForm = null;
  let ticketPreview = null;
  let sourceConversationId = conversationId;
  const request = { controller: new AbortController(), writer };
  activeRequest = request;
  updateControls();
  status.textContent = "正在等待客服回复…";
  document.querySelector("#conversation-title").textContent = message;
  try {
    const body = { message, request_id: requestId };
    if (selectedOrderId) body.selected_order_id = selectedOrderId;
    if (ticketConfirmation !== null) {
      body.ticket_confirmation = ticketConfirmation;
      body.ticket_tool_call_id = ticketCallId;
    }
    if (conversationId) body.conversation_id = conversationId;
    const response = await requestChat(body, request.controller.signal);
    // 保存会话信息，将文本交给逐字渲染器。
    await readChatStream(response, function handleEvent(event, data) {
      if (activeRequest !== request) return;
      if (event === "meta") {
        conversationId = data.conversation_id;
        sourceConversationId = data.conversation_id;
        document.querySelector("#session-label").textContent = "当前对话";
      }
      if (event === "token") {
        status.textContent = "客服正在回复…";
        writer.append(data.text);
      }
      if (event === "tool_status") updateToolBadge(assistant, data);
      if (event === "sources") sources = data.sources;
      if (event === "actions") actions = data.actions;
      if (event === "order_required") orders = data.orders;
      if (event === "refund_form") refundForm = data;
      if (event === "ticket_required") {
        ticketPreview = data;
        updateToolBadge(assistant, { name: "create_ticket", tool_call_id: data.tool_call_id, state: "waiting" });
      }
      if (event === "understanding") {
        let note = assistant.bubble.querySelector(".question-understanding");
        if (!note) {
          note = document.createElement("p");
          note.className = "question-understanding";
          assistant.bubble.prepend(note);
        }
        note.textContent = `理解为：${data.question} · ${data.intent}`;
      }
      if (event === "workflow") {
        const labels = { resolve: "正在理解您的问题…", select_order: "正在核对所选订单…",
          load_order: "正在查询订单…", retrieve: "正在检索相关政策…", confidence: "正在核对政策依据…" };
        if (labels[data.node]) status.textContent = labels[data.node];
      }
    });
    await writer.drain();
    if (activeRequest !== request) return;
    renderCitations(assistant, sources);
    addActions(assistant, actions, requestId, sourceConversationId);
    if (orders) addOrderPicker(assistant, orders, message, requestId, sourceConversationId);
    else if (ticketPreview) addTicketPreview(assistant, ticketPreview, message, requestId, sourceConversationId);
    else addFeedback(assistant, requestId);
    if (refundForm) addRefundForm(assistant, refundForm, requestId, sourceConversationId);
    status.textContent = ticketPreview ? "请确认提交或取消这张工单" :
      orders ? "点选订单后继续处理，也可以发送其他问题" : "您可以继续追问，前面的内容会接着聊";
  } catch (error) {
    if (activeRequest !== request) return;
    writer.stop();
    assistant.article.classList.add("failed");
    if (!assistant.reply.textContent) assistant.reply.textContent = "本轮未能收到完整回复。";
    const state = document.createElement("p");
    state.className = "message-state";
    state.textContent = "本轮未完成";
    assistant.content.append(state);
    failedTurn = { message, requestId, selectedOrderId, ticketConfirmation, ticketCallId,
      ...assistant, expired: error.status === 404 };
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
      void refreshConversations();
      followConversation();
      input.focus({ preventScroll: true });
    }
  }
}

/** 连续发言或点选紧随流结束时，仅对服务端提交收尾的短暂忙状态自动重试。 */
async function requestChat(body, signal) {
  for (let attempt = 0; ; attempt += 1) {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body), signal,
    });
    if (response.status !== 409 || attempt >= 4) return response;
    const failure = await response.clone().json();
    if (typeof failure.detail !== "string" || !failure.detail.includes("正在执行")) return response;
    await new Promise(function waitForCommit(resolve) {
      // 保持原会话、请求与所选订单，只等待上次响应完成持久化。
      setTimeout(resolve, 250);
    });
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
  if (failedTurn && !failedTurn.expired) void sendMessage(failedTurn.message, failedTurn,
    failedTurn.selectedOrderId, failedTurn.ticketConfirmation, failedTurn.ticketCallId);
}

/** 开始独立的新会话草稿，首次发送时创建；旧会话目录及服务端原文保持可访问。 */
function resetConversation() {
  selectionRevision += 1;
  loadingConversation = false;
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
  setSidebarOpen(false);
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
document.querySelector("#mobile-new").addEventListener("click", resetConversation);
conversationList.addEventListener("click", handleConversationClick);
document.querySelector("#conversation-toggle").addEventListener("click", toggleSidebar);
document.querySelector("#sidebar-backdrop").addEventListener("click", toggleSidebar);
document.addEventListener("keydown", handleSidebarEscape);
welcome.addEventListener("click", handleSuggestion);
updateControls();
setupKnowledgeUpload();
void refreshConversations();
