const status = document.querySelector("#review-status");
const filter = document.querySelector("#review-filter");
const labels = { pending: "待审核", approved: "已通过", rejected: "已驳回", publishing: "入库待重试" };
const entries = { workflow_gate: "检索证据不足", retrieval: "检索证据不足", generation: "生成自评不足", feedback: "用户反馈未解决" };

/** 用纯文本构建节点，避免用户原话和知识片段成为 HTML。 */
function el(tag, text, className = "") {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

/** 统一读取后端结果，失败时保留可读错误供当前操作提示。 */
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "操作失败，请稍后重试。");
  return result;
}

/** 展示一条原始问题及该轮召回快照，空数组与未走检索明确区分。 */
function origin(row) {
  const item = el("li", undefined, "origin");
  const entryNames = (row.entry_points || [row.entry_point]).map(function entryName(entry) {
    // 同一轮命中多个入口时展示所有来源，计数仍只表示一个用户问题。
    return entries[entry] || entry;
  }).join(" / ");
  item.append(el("p", row.question), el("p", `${entryNames} · ${row.created_at} · ${row.request_id}`, "meta"));
  if (row.retrieval_snapshot === null) item.append(el("p", "该轮没有检索记录。", "meta"));
  else if (!row.retrieval_snapshot.length) item.append(el("p", "该轮已检索，未召回片段。", "meta"));
  else for (const [index, match] of row.retrieval_snapshot.entries()) {
    const details = el("details", undefined, "snapshot");
    details.append(el("summary", `Top ${index + 1} · 得分 ${Number(match.score).toFixed(4)} · ${match.section_path || match.question || "知识片段"}`));
    details.append(el("pre", match.text || match.answer || ""));
    item.append(details);
  }
  return item;
}

/** 构建可展开审核项；示例答案只供参考，核准答案必须由审核人填写。 */
function reviewRow(row) {
  const details = el("details", undefined, "review-row");
  details.dataset.reviewId = row.id;
  const summary = el("summary");
  summary.append(el("strong", row.normalized_question), el("span", `${row.occurrence_count} 次 · ${labels[row.status] || row.status}　⌄`));
  details.append(summary, el("p", `示例答案（待核实）：${row.example_answer}`, "sample"));
  const origins = el("ul", undefined, "origins");
  let loaded = false;
  /** 首次展开才读取所有归并原话，失败允许重新展开重试。 */
  async function loadOrigins() {
    if (!details.open || loaded) return;
    loaded = true;
    try {
      const data = await api(`/api/review-queue/${row.id}`);
      origins.replaceChildren(...data.questions.map(origin));
    } catch (error) { loaded = false; origins.replaceChildren(el("li", error.message, "empty")); }
  }
  details.addEventListener("toggle", loadOrigins);
  details.append(el("h3", "归并原话与当轮召回"), origins);
  if (["pending", "publishing"].includes(row.status)) {
    const label = el("label", "核准答案");
    const answer = el("textarea");
    answer.id = `answer-${row.id}`;
    answer.maxLength = 6000;
    answer.value = row.approved_answer || "";
    answer.placeholder = "请核实政策与事实后，填写可正式发布的答案。";
    label.htmlFor = answer.id;
    const actions = el("div", undefined, "review-actions");
    const approve = el("button", "通过并写入知识库", "primary");
    const reject = el("button", "驳回", "secondary");
    approve.type = reject.type = "button";
    /** 保存人工决定；写索引失败时保留当前输入，重试使用相同知识来源。 */
    async function decide(decision) {
      if (decision === "approved" && !answer.value.trim()) { answer.focus(); status.textContent = "请先填写核准答案。"; return; }
      approve.disabled = reject.disabled = true;
      status.textContent = decision === "approved" ? "正在写入知识库并同步索引…" : "正在保存审核结果…";
      try {
        await api(`/api/review-queue/${row.id}/decision`, { decision, approved_answer: answer.value.trim() });
        await load();
        status.textContent = decision === "approved" ? "已通过，知识已入库并可检索。" : "已驳回该问题。";
      } catch (error) { status.textContent = error.message; }
      finally { approve.disabled = reject.disabled = false; }
    }
    /** 提交审核通过决定。 */
    function accept() { return decide("approved"); }
    /** 提交驳回决定。 */
    function decline() { return decide("rejected"); }
    approve.addEventListener("click", accept);
    reject.addEventListener("click", decline);
    actions.append(approve, reject);
    details.append(label, answer, actions);
  } else if (row.approved_answer) details.append(el("h3", "核准答案"), el("p", row.approved_answer));
  return details;
}

/** 按总 token 画用量条；真实用量缺失不会伪装为零费用。 */
function renderCosts(rows) {
  const target = document.querySelector("#cost-list");
  target.replaceChildren();
  const maximum = Math.max(1, ...rows.map(totalTokens));
  for (const row of rows) {
    const item = el("div", undefined, "cost-row");
    const meter = el("meter");
    meter.max = maximum; meter.value = row.total_tokens;
    meter.setAttribute("aria-label", `${row.intent} ${row.total_tokens} tokens`);
    const value = el("div", `${row.total_tokens.toLocaleString()} tokens`);
    value.append(el("small", `输入 ${row.input_tokens} / 输出 ${row.output_tokens} · ${row.request_count} 轮 · 缺用量 ${row.missing_usage_count}`));
    item.append(el("strong", row.intent), meter, value);
    target.append(item);
  }
  if (!rows.length) target.append(el("p", "暂无请求用量，发送一条客服消息后刷新。", "empty"));
}

/** 提取总 token，供比较条统一比例。 */
function totalTokens(row) { return row.total_tokens; }

/** 在固定 0–1 坐标中画历史指标，空值断线，保留表格作为可访问数据。 */
function renderRuns(rows) {
  const chart = document.querySelector("#trend-chart");
  const tableTarget = document.querySelector("#run-list");
  chart.replaceChildren(); tableTarget.replaceChildren();
  if (!rows.length) { chart.append(el("p", "尚无已完成评估。运行两轮后可以比较趋势。", "empty")); return; }
  const keys = [["recall@5", "Recall@5", "#1b6255"], ["mrr", "MRR", "#4076a2"], ["faithfulness", "Faithfulness", "#a67428"]];
  const legend = el("div", undefined, "trend-legend");
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", "0 0 800 240"); svg.classList.add("trend-svg");
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "历史评估折线，纵轴0到1；具体数值见下表。");
  for (const level of [0, 0.5, 1]) {
    const y = 205 - level * 180;
    const line = document.createElementNS(namespace, "line");
    line.setAttribute("x1", "36"); line.setAttribute("x2", "780"); line.setAttribute("y1", y); line.setAttribute("y2", y); line.setAttribute("stroke", "#e0e7e3");
    const text = document.createElementNS(namespace, "text"); text.setAttribute("x", "4"); text.setAttribute("y", y + 4); text.textContent = level;
    svg.append(line, text);
  }
  for (const [key, name, color] of keys) {
    const label = el("span", name); const marker = el("i"); marker.style.background = color; label.prepend(marker); legend.append(label);
    let path = "", connected = false;
    rows.forEach(function plot(row, index) {
      // 空分母留白，避免将 N/A 画成 0 分。
      const value = row.metrics.hybrid_rerank?.[key];
      if (value === null || value === undefined) { connected = false; return; }
      const x = 40 + index * 730 / Math.max(1, rows.length - 1), y = 205 - value * 180;
      path += `${connected ? "L" : "M"}${x},${y} `; connected = true;
      const dot = document.createElementNS(namespace, "circle");
      dot.setAttribute("cx", x); dot.setAttribute("cy", y); dot.setAttribute("r", "4"); dot.setAttribute("fill", color);
      const title = document.createElementNS(namespace, "title"); title.textContent = `${row.created_at} ${name}: ${value.toFixed(4)}`; dot.append(title); svg.append(dot);
    });
    const line = document.createElementNS(namespace, "path"); line.setAttribute("d", path); line.setAttribute("stroke", color); line.setAttribute("fill", "none"); line.setAttribute("stroke-width", "2"); svg.append(line);
  }
  chart.append(legend, svg, el("p", "从左到右：较早 → 最近。红色数值表示较上一轮下滑。", "table-note"));
  const table = el("table"), head = el("thead"), header = el("tr");
  for (const title of ["时间 / 触发方式", "题数", "Recall@5", "MRR", "Faithfulness"]) header.append(el("th", title));
  head.append(header); const body = el("tbody");
  rows.forEach(function runRow(row, index) {
    // 保持所有轮次时间升序，以便和折线逐点对照。
    const tr = el("tr"); tr.append(el("td", `${row.created_at} · ${row.trigger_type}`), el("td", row.dataset_count));
    for (const [key] of keys) {
      const value = row.metrics.hybrid_rerank?.[key], previous = rows[index - 1]?.metrics.hybrid_rerank?.[key];
      const cell = el("td", value === null || value === undefined ? "N/A" : value.toFixed(4));
      if (value != null && previous != null && value < previous) cell.className = "decline";
      tr.append(cell);
    }
    body.append(tr);
  });
  table.append(head, body); tableTarget.append(table);
}

/** 同步刷新队列与统计，局部失败明确提示，避免展示假成功。 */
async function load() {
  status.textContent = "正在刷新…";
  try {
    const [queue, costs, runs] = await Promise.all([api(`/api/review-queue?status=${filter.value}`), api("/api/costs"), api("/api/evaluations/runs")]);
    document.querySelector("#queue-count").textContent = `${queue.length} 个知识缺口 · 展开查看原话、证据和审核操作`;
    document.querySelector("#review-list").replaceChildren(...(queue.length ? queue.map(reviewRow) : [el("p", "当前状态没有问题。新问题将自动整理，也可点击“处理新问题”。", "empty")]));
    renderCosts(costs); renderRuns(runs); status.textContent = "数据已更新。";
  } catch (error) { status.textContent = error.message; }
}

/** 手动触发同一持久问题池的整理流程，失败的条目留待重试。 */
async function processPool() {
  const button = document.querySelector("#process-pool"); button.disabled = true; status.textContent = "正在标准化和查重…";
  try { await api("/api/flywheel/process", {}); await load(); }
  catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; }
}
document.querySelector("#refresh").addEventListener("click", load);
document.querySelector("#process-pool").addEventListener("click", processPool);
filter.addEventListener("change", load);
load();
