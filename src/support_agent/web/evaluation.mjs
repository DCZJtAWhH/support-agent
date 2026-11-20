const STRATEGIES = [
  ["dense", "纯向量"], ["bm25", "纯 BM25"], ["hybrid", "混合检索"], ["hybrid_rerank", "混合加重排"],
];
const TYPES = { direct: "直接问法", colloquial: "口语", synonym: "同义词", conditions: "条件判断",
  model: "具体型号", filter: "品类过滤", multi_fact: "跨块事实", unknown: "未知问题",
  safety: "安全边界", ambiguous: "歧义问法" };
const LEVELS = { easy: "简单", medium: "中等", hard: "困难" };
let data = null;
let polling = null;
let controller = null;
let currentJob = null;
const typeSelect = document.querySelector("#query-type");
const levelSelect = document.querySelector("#difficulty");
const kSelect = document.querySelector("#recall-k");
const caseSelect = document.querySelector("#case-select");
const search = document.querySelector("#case-search");
const runButton = document.querySelector("#run-evaluation");

/** 创建纯文本节点，报告中的问题、答案、原文和文件名均不作为HTML执行。 */
function element(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null && text !== undefined) node.textContent = text;
  return node;
}

/** 更新固定页面字段，保留缺失值的未知状态。 */
function setText(id, text) { document.getElementById(id).textContent = text; }

/** 仅有限数字显示四位小数，空分母不会被格式化成0或满分。 */
function metric(value) { return Number.isFinite(value) ? value.toFixed(4) : "N/A"; }

/** 显示可恢复的读取/启动错误，避免把失败当作空报告。 */
function showError(message) {
  const box = document.querySelector("#page-error");
  box.hidden = !message;
  box.textContent = message || "";
}

/** 请求同源评估接口，错误消息仅采用服务端公开字段。 */
async function request(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  let result;
  try { result = await response.json(); } catch { throw new Error("评估服务未返回有效数据，请稍后刷新。"); }
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "评估请求未完成，请稍后重试。");
  return result;
}

/** 更新筛选选项并保持用户已有选择，选项始终来自真实报告。 */
function setOptions(select, values, labels, emptyLabel) {
  const selected = select.value;
  select.replaceChildren(new Option(emptyLabel, ""));
  for (const value of values) select.add(new Option(labels[value] || value, value));
  select.value = values.includes(selected) ? selected : "";
}

/** 展示报告快照与当前知识/标注的关系，不把旧成绩显示为当前状态。 */
function renderOverview() {
  const report = data.report;
  const metadata = report?.metadata || {};
  setText("report-cases", report ? `${report.case_count} 题` : "—");
  setText("report-rows", report ? `${report.row_count} 条策略结果` : "尚未生成评估报告");
  setText("report-corpus", Number.isFinite(metadata.corpus_size) ? `${metadata.corpus_size} 块` : "—");
  setText("current-corpus", `当前知识库：${data.corpus.size} 块${data.corpus.pending_count ? `，${data.corpus.pending_count} 块待同步` : ""}`);
  setText("dataset-cases", `${data.dataset.case_count} 题`);
  const date = metadata.created_at ? new Date(metadata.created_at) : null;
  setText("report-time", date && Number.isFinite(date.getTime()) ? date.toLocaleString("zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }) : "—");
  setText("report-mode", !report ? "尚未生成" : report.mode === "live" ? "真实模型运行" : "回放报告");
  const freshness = document.querySelector("#freshness");
  freshness.classList.toggle("stale", data.stale || data.dataset_changed);
  if (!report) freshness.textContent = "暂无已完成报告。点击“重新评估”可运行当前标注集。";
  else if (data.stale || data.dataset_changed) freshness.textContent = `${[
    data.stale ? "知识库已更新" : "", data.dataset_changed ? "评估集已更新" : "",
  ].filter(Boolean).join("，")}。当前展示历史报告，重新评估后才会更新成绩。`;
  else freshness.textContent = "报告与当前知识库、标注集一致。成绩只覆盖已有标注题，请结合逐题依据复核。";
  const modelInfo = [];
  for (const [key, label] of [["model", "生成模型"], ["embedding", "嵌入模型"], ["reranker", "重排模型"]]) {
    if (metadata[key]) modelInfo.push(`${label}：${metadata[key]}`);
  }
  setText("model-info", modelInfo.join("；"));
  setOptions(typeSelect, data.filters.query_types, TYPES, "全部类型");
  const levels = [];
  for (const key of ["easy", "medium", "hard", ...data.filters.difficulties]) {
    if (data.filters.difficulties.includes(key) && !levels.includes(key)) levels.push(key);
  }
  setOptions(levelSelect, levels, LEVELS, "全部难度");
}

/** 根据后台聚合值绘制对比表，最优值允许并列，变化量始终相对纯向量。 */
function renderTable() {
  if (!data) return;
  const body = document.querySelector("#strategy-rows");
  body.replaceChildren();
  const keys = [`recall@${kSelect.value}`, "mrr", "faithfulness", "refusal_accuracy", "answer_rate_on_answerable"];
  setText("recall-heading", `Recall@${kSelect.value}`);
  setText("selection-count", `${data.case_count} 道题 · ${data.rows.length} 条策略结果${data.case_count ? "" : " · 当前筛选下没有题目"}`);
  for (const [strategy, label] of STRATEGIES) {
    const values = data.strategies[strategy];
    const row = element("tr");
    row.dataset.strategy = strategy;
    const name = element("td", "", label);
    if (strategy === "dense") name.append(element("br"), element("span", "baseline", "对照基线"));
    row.append(name, element("td", "", values.count));
    for (const key of keys) {
      const value = values[key];
      const baseline = data.strategies.dense[key];
      let best = -1;
      for (const [other] of STRATEGIES) {
        const candidate = data.strategies[other][key];
        if (Number.isFinite(candidate)) best = Math.max(best, candidate);
      }
      const cell = element("td", Number.isFinite(value) && value > 0 && Math.abs(value - best) < 1e-9 ? "best" : "");
      cell.dataset.metric = key;
      cell.append(element("span", "metric-value", metric(value)));
      if (Number.isFinite(value)) {
        const bar = element("span", "metric-bar");
        bar.setAttribute("aria-hidden", "true");
        const fill = element("i");
        fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
        bar.append(fill);
        cell.append(bar);
      }
      if (strategy !== "dense" && Number.isFinite(value) && Number.isFinite(baseline)) {
        const difference = value - baseline;
        const label = Math.abs(difference) < 1e-9 ? "与基线持平" : `Δ ${difference > 0 ? "+" : ""}${difference.toFixed(4)}`;
        cell.append(element("small", `metric-delta ${difference > 1e-9 ? "up" : difference < -1e-9 ? "down" : ""}`, label));
      }
      row.append(cell);
    }
    body.append(row);
  }
}

/** 打开评估时的原文快照，原文跳转仅由合法知识主键构造。 */
function openSource(source) {
  setText("source-path", source.section_path || source.question || "来源知识块");
  setText("source-origin", source.source_ref || "未记录来源名称");
  setText("source-text", source.answer || source.text || "报告没有保存原文。");
  const link = document.querySelector("#source-link");
  const valid = Number.isSafeInteger(source.chunk_id) && source.chunk_id > 0;
  link.hidden = !valid;
  if (valid) link.href = `/knowledge/chunks/${source.chunk_id}/original?version=${encodeURIComponent(source.content_hash || "")}#chunk-${source.chunk_id}`;
  document.querySelector("#evaluation-source").showModal();
}

/** 将真实回答编号映射到该策略本次保存的来源，缺失映射时保留普通文本。 */
function answerContent(row) {
  const box = element("div", "case-answer");
  const text = row.answer || "本次没有生成回答。";
  let offset = 0;
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    box.append(document.createTextNode(text.slice(offset, match.index)));
    const source = (row.sources || []).find(function matchingSource(item) {
      // 相同编号只在当前策略回答内查找，不混用其他策略的来源。
      return item.citation === Number(match[1]);
    });
    if (source) {
      const button = element("button", "citation", match[0]);
      button.type = "button";
      button.setAttribute("aria-label", `查看引用 ${match[1]}`);
      button.addEventListener("click", function showCitation() {
        // 闭包持有报告的原文快照，后续知识更新不修改这段证据。
        openSource(source);
      });
      box.append(button);
    } else box.append(document.createTextNode(match[0]));
    offset = match.index + match[0].length;
  }
  box.append(document.createTextNode(text.slice(offset)));
  return box;
}

/** 展示实际召回顺序、标注命中及策略内部相关分，不把分数当成跨策略概率。 */
function rankings(row) {
  const details = element("details");
  const matches = row.matches || [];
  details.append(element("summary", "", `召回排序与来源 · ${matches.length} 块`));
  details.append(element("p", "muted", "分数仅在同一策略内比较；点击可查看评估时的原文。"));
  const list = element("ol", "rank-list");
  for (const [index, label] of (row.ranked || []).entries()) {
    const source = matches[index];
    const item = element("li");
    const text = `${index + 1}. ${label}${row.relevant.includes(label) ? " · 标注来源" : ""}`;
    if (source) {
      const button = element("button", "", text);
      button.type = "button";
      button.append(element("small", "", `相关分 ${metric(source.score)}`));
      button.addEventListener("click", function showRankedSource() {
        // 原文与排名逐项对应，避免使用当前数据库内容替代旧快照。
        openSource(source);
      });
      item.append(button);
    } else item.textContent = text;
    list.append(item);
  }
  if (!list.children.length) list.append(element("li", "muted", "未召回来源。"));
  details.append(list);
  return details;
}

/** 展开逐事实支持判断，拒答没有事实时明确显示N/A。 */
function claims(row) {
  const details = element("details");
  const values = row.claims || [];
  let supported = 0;
  for (const claim of values) if (claim.supported) supported++;
  details.append(element("summary", "", `Faithfulness ${values.length ? metric(supported / values.length) : "N/A"} · ${values.length} 条事实`));
  if (!values.length) details.append(element("p", "muted", "本次没有生成可评事实，不计作忠实度满分。"));
  else {
    const list = element("ol", "claim-list");
    for (const claim of values) {
      const item = element("li", claim.supported ? "" : "unsupported");
      item.append(element("strong", "", claim.supported ? "有依据 · " : "无依据 · "), document.createTextNode(claim.statement || "未记录事实文本"));
      item.append(element("small", "", claim.reason || "未记录判定原因"));
      list.append(item);
    }
    details.append(list);
  }
  return details;
}

/** 渲染选中题目的标注和四策略回答，事实、拒答和完整性保持各自含义。 */
function renderCase() {
  if (!data) return;
  const reference = document.querySelector("#case-reference");
  const cards = document.querySelector("#case-strategies");
  reference.replaceChildren();
  cards.replaceChildren();
  const rows = data.rows.filter(function selectedCase(row) {
    // 筛选后按题目标识聚合四种策略，未命中的组合显示空态。
    return row.id === caseSelect.value;
  });
  if (!rows.length) {
    reference.append(element("p", "empty", "没有匹配题目，可调整筛选条件或搜索内容。"));
    return;
  }
  const first = rows[0];
  reference.append(element("h3", "", first.question));
  reference.append(element("p", "", `${first.id} · ${TYPES[first.query_type] || first.query_type} · ${LEVELS[first.difficulty] || first.difficulty}`));
  reference.append(element("p", "", first.expected_refusal ? "标注结论：应明确拒答。" : `标注答案要点：${(first.expected_facts || []).join("、") || "未记录"}`));
  if (first.relevant.length) reference.append(element("p", "", `预期来源：\n${first.relevant.join("\n")}`));
  if (first.category) reference.append(element("p", "", `品类过滤：${first.category}`));
  for (const [strategy, name] of STRATEGIES) {
    const row = rows.find(function matchingStrategy(item) {
      // 固定对比顺序，不因数值或数据文件行顺序改变排列。
      return item.strategy === strategy;
    });
    if (!row) continue;
    const card = element("article", "case-card");
    card.dataset.strategy = strategy;
    const header = element("header");
    header.append(element("h3", "", name), element("span", `tag ${row.refused ? "refused" : ""}`, row.refused ? "已拒答" : "已回答"));
    card.append(header, answerContent(row));
    if (row.refusal_reason) card.append(element("p", "case-reason", row.refusal_reason));
    card.append(rankings(row), claims(row));
    cards.append(card);
  }
}

/** 搜索只改变逐题选项，不擅自改变后端已经聚合的指标口径。 */
function renderCaseOptions() {
  if (!data) return;
  const previous = caseSelect.value;
  const needle = search.value.trim().toLocaleLowerCase();
  const seen = new Set();
  caseSelect.replaceChildren();
  for (const row of data.rows) {
    if (seen.has(row.id) || (needle && !`${row.id} ${row.question}`.toLocaleLowerCase().includes(needle))) continue;
    seen.add(row.id);
    caseSelect.add(new Option(`${row.question} · ${row.id}`, row.id));
  }
  if (seen.has(previous)) caseSelect.value = previous;
  if (!seen.size) caseSelect.add(new Option("没有匹配题目", ""));
  renderCase();
}

/** 显示真实后台完成数，运行失败时保留已有成绩并恢复重试按钮。 */
function renderJob(job) {
  currentJob = job;
  const panel = document.querySelector("#job-panel");
  panel.hidden = !job;
  const running = job?.status === "running";
  runButton.disabled = running || !data || data.corpus.pending_count > 0 || !data.dataset.case_count;
  runButton.textContent = running ? "正在评估…" : "重新评估";
  runButton.setAttribute("aria-busy", String(running));
  if (!job) return;
  panel.className = `job-panel ${job.status}`;
  setText("job-title", { running: "评估进行中", completed: "评估已完成", failed: "评估未完成", interrupted: "评估已中断" }[job.status] || "评估状态");
  setText("job-count", `${job.completed} / ${job.total} 条结果`);
  setText("job-message", job.message);
  const progress = document.querySelector("#job-progress");
  progress.max = Math.max(1, job.total);
  progress.value = job.completed;
  if (running && polling === null) polling = setTimeout(pollJob, 1500);
}

/** 读取报告时取消已过期的筛选请求，避免较慢响应覆盖用户的新选择。 */
async function loadReport() {
  controller?.abort();
  const active = new AbortController();
  controller = active;
  showError("");
  try {
    const query = new URLSearchParams({ query_type: typeSelect.value, difficulty: levelSelect.value });
    const result = await request(`/api/evaluations?${query}`, { signal: active.signal });
    if (controller !== active) return;
    data = result;
    renderOverview();
    renderTable();
    renderCaseOptions();
    renderJob(result.job);
  } catch (error) {
    if (error.name !== "AbortError" && controller === active) {
      showError(error.message);
      setText("selection-count", "读取失败，仍保留上一次成功读取的结果。");
    }
  }
}

/** 只轮询轻量进度，完成后读取新报告；网络暂时不可用时继续恢复进度。 */
async function pollJob() {
  polling = null;
  try {
    const result = await request("/api/evaluations/status");
    const wasRunning = currentJob?.status === "running";
    renderJob(result.job);
    if ((result.job?.status === "completed" && data?.report?.id !== result.job.id) ||
        (wasRunning && ["failed", "interrupted"].includes(result.job?.status))) await loadReport();
  } catch {
    setText("job-message", "暂时无法获取进度，正在重试；后台任务会继续执行。");
    polling = setTimeout(pollJob, 3000);
  }
}

/** 明确点击后才启动真实评估，不因查看页面或切换筛选消耗模型调用。 */
async function startRun() {
  if (runButton.disabled || currentJob?.status === "running") return;
  runButton.disabled = true;
  showError("");
  try {
    const result = await request("/api/evaluations/run", { method: "POST" });
    renderJob(result.job);
  } catch (error) {
    showError(error.message);
    await loadReport();
    showError(error.message);
  }
}

/** 清除类型与难度限制，保留用户选择的召回深度。 */
function clearFilters() {
  typeSelect.value = "";
  levelSelect.value = "";
  void loadReport();
}

/** 关闭原文对话框并恢复触发按钮的焦点。 */
function closeSource() { document.querySelector("#evaluation-source").close(); }

/** 离开页面时仅停止本页轮询，服务端评估仍按明确启动的任务继续运行。 */
function stopPolling() { clearTimeout(polling); polling = null; controller?.abort(); }

/** 浏览器从往返缓存恢复页面时重新读取任务和版本，避免停留在旧进度。 */
function restorePage(event) { if (event.persisted) void loadReport(); }

typeSelect.addEventListener("change", loadReport);
levelSelect.addEventListener("change", loadReport);
kSelect.addEventListener("change", renderTable);
search.addEventListener("input", renderCaseOptions);
caseSelect.addEventListener("change", renderCase);
runButton.addEventListener("click", startRun);
document.querySelector("#clear-filters").addEventListener("click", clearFilters);
document.querySelector("#reload-report").addEventListener("click", loadReport);
document.querySelector("#close-source").addEventListener("click", closeSource);
window.addEventListener("pagehide", stopPolling);
window.addEventListener("pageshow", restorePage);
void loadReport();
