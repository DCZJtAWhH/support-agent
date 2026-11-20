/** 初始化独立知识上传入口，保留文件队列与重试状态，不把文件发送成聊天消息。 */
export function setupKnowledgeUpload() {
  const dialog = document.querySelector("#knowledge-upload-dialog");
  const input = document.querySelector("#knowledge-files");
  const dropzone = document.querySelector("#knowledge-dropzone");
  const list = document.querySelector("#knowledge-upload-list");
  const summary = document.querySelector("#knowledge-upload-summary");
  const start = document.querySelector("#knowledge-upload-start");
  const clear = document.querySelector("#knowledge-upload-clear");
  const close = document.querySelector("#knowledge-upload-close");
  const back = document.querySelector("#knowledge-upload-return");
  let items = [];
  let busy = false;

  /** 根据队列与导入状态更新操作按钮，成功文件不会随重试再次上传。 */
  function updateControls() {
    const pending = items.filter(function canImport(item) {
      // 只重试传输或服务失败的文件，不重试本地格式校验错误。
      return ["queued", "failed"].includes(item.state);
    });
    start.disabled = busy || pending.length === 0;
    start.textContent = busy ? "正在导入…" : pending.some(function isRetry(item) {
      // 失败文件保留原始 File 对象，用户不必重新选择。
      return item.state === "failed";
    }) ? "重试导入" : "开始导入";
    clear.disabled = busy || items.length === 0;
    input.disabled = busy;
    close.disabled = busy;
    back.disabled = busy;
    dropzone.classList.toggle("disabled", busy);
    dialog.setAttribute("aria-busy", String(busy));
  }

  /** 将每个文件的独立状态绘制为纯文本，文件名和服务提示均不作为 HTML 执行。 */
  function renderItem(item) {
    item.row.dataset.state = item.state;
    item.description.textContent = item.message;
    item.progress.hidden = !["uploading", "processing"].includes(item.state);
    if (item.state === "processing") item.progress.removeAttribute("value");
    else item.progress.value = item.percent || 0;
  }

  /** 加入或替换同名文件，先在浏览器检查格式和大小，再由服务器校验正文。 */
  function selectFiles(files) {
    if (busy) return;
    for (const file of Array.from(files)) {
      const existing = items.findIndex(function sameName(item) {
        // 同名只保留新选中的一次导入意图，避免队列内顺序覆盖造成困惑。
        return item.file.name === file.name;
      });
      if (existing >= 0) items.splice(existing, 1)[0].row.remove();
      const row = document.createElement("li");
      const name = document.createElement("strong");
      name.textContent = file.name;
      const description = document.createElement("span");
      const progress = document.createElement("progress");
      progress.max = 100;
      progress.setAttribute("aria-label", `${file.name} 导入进度`);
      row.append(name, description, progress);
      const item = { file, row, description, progress, state: "queued",
        message: `${Math.max(1, Math.ceil(file.size / 1024))} KB · 等待导入` };
      if (!/\.(md|markdown|txt)$/i.test(file.name)) {
        item.state = "invalid";
        item.message = "格式不支持，请选择 Markdown 或 TXT 文件。";
      } else if (file.size > 1024 * 1024) {
        item.state = "invalid";
        item.message = "文件超过 1 MiB，请拆分后选择。";
      } else if (!file.size) {
        item.state = "invalid";
        item.message = "文件为空，请添加知识正文。";
      }
      items.push(item);
      list.append(row);
      renderItem(item);
    }
    input.value = "";
    summary.textContent = "准备好后点击“开始导入”；导入成功的知识即可在聊天中查询。";
    updateControls();
  }

  /** 发送单个 multipart 文件，分别展示网络传输和服务器入库阶段。 */
  function uploadFile(item) {
    return new Promise(function sendUpload(resolve) {
      // 不手动设置 Content-Type，由浏览器生成正确的 multipart boundary。
      const xhr = new XMLHttpRequest();
      const body = new FormData();
      body.append("file", item.file);
      xhr.open("POST", "/knowledge/upload");
      xhr.responseType = "json";
      item.state = "uploading";
      item.message = "正在上传文件…";
      item.percent = 0;
      renderItem(item);

      /** 网络传输百分比仅表示文件上传，不提前宣称知识已经可用。 */
      function uploadProgress(event) {
        if (event.lengthComputable && item.state === "uploading") {
          item.percent = Math.round(event.loaded / event.total * 100);
          item.message = `正在上传文件 · ${item.percent}%`;
          renderItem(item);
        }
      }

      /** 文件发完后切到不定进度，等待后台完成解析和索引。 */
      function processing() {
        if (item.state !== "uploading") return;
        item.state = "processing";
        item.message = "文件已上传，正在导入知识…";
        renderItem(item);
      }

      /** 网络中断时保留文件以便重试，同名导入由服务器保证不重复建块。 */
      function networkFailure() {
        item.state = "failed";
        item.message = "无法确认导入结果，请检查网络后重试；重复上传不会重复添加知识。";
        renderItem(item);
        resolve();
      }

      /** 仅服务器明确报告索引就绪才显示成功，否则展示可读错误并保留重试入口。 */
      function completed() {
        const result = xhr.response;
        if (xhr.status === 201 && result?.status === "ready" && Number.isInteger(result.chunk_count)) {
          item.state = "ready";
          item.message = `已加入知识库 · ${result.chunk_count} 个知识块 · 可直接提问`;
        } else {
          item.state = [400, 413, 422].includes(xhr.status) ? "invalid" : "failed";
          const detail = result?.detail;
          item.message = typeof detail === "string" ? detail
            : detail?.message || "导入未完成，请稍后重试。";
        }
        renderItem(item);
        resolve();
      }

      xhr.upload.addEventListener("progress", uploadProgress);
      xhr.upload.addEventListener("load", processing);
      xhr.addEventListener("load", completed);
      xhr.addEventListener("error", networkFailure);
      xhr.addEventListener("abort", networkFailure);
      try { xhr.send(body); } catch { networkFailure(); }
    });
  }

  /** 顺序处理本次待导入队列，单个失败不阻断其他文件，结束后汇总实际结果。 */
  async function beginImport() {
    if (busy) return;
    busy = true;
    updateControls();
    try {
      for (const item of items) {
        if (!["queued", "failed"].includes(item.state)) continue;
        summary.textContent = `正在导入 ${item.file.name}，请保持页面打开。`;
        await uploadFile(item);
      }
    } finally {
      busy = false;
      const ready = items.filter(function isReady(item) {
        // 汇总成功与未成功项，避免部分失败时误显示全部成功。
        return item.state === "ready";
      }).length;
      summary.textContent = `${ready} 个文件已加入知识库${items.length > ready
        ? `，${items.length - ready} 个文件未完成，请查看上方提示。` : "，返回聊天即可提问。"}`;
      updateControls();
    }
  }

  /** 打开入口时保留本页面的导入记录，便于查看结果或重试。 */
  function openDialog() { dialog.showModal(); }

  /** 正在导入时防止关闭入口；完成或失败后可自由返回聊天。 */
  function closeDialog() { if (!busy) dialog.close(); }

  /** 读取浏览器选择的文件，和拖拽入口使用同一校验逻辑。 */
  function fileSelection() { selectFiles(input.files); }

  /** 清空仅影响界面列表，已入库知识保持可用。 */
  function clearList() {
    if (busy) return;
    items = [];
    list.replaceChildren();
    summary.textContent = "列表已清空，已导入的知识仍可查询。";
    updateControls();
  }

  /** 拖入文件时阻止浏览器跳转打开文件，并显示拖放反馈。 */
  function dragOver(event) {
    event.preventDefault();
    if (!busy) dropzone.classList.add("dragging");
  }

  /** 移出拖放区域时恢复正常样式。 */
  function dragLeave() { dropzone.classList.remove("dragging"); }

  /** 处理拖放文件，导入期间不修改正在执行的队列。 */
  function dropFiles(event) {
    event.preventDefault();
    dragLeave();
    selectFiles(event.dataTransfer.files);
  }

  /** Escape 遵守与关闭按钮一致的处理中状态，避免误以为导入已取消。 */
  function cancelDialog(event) { if (busy) event.preventDefault(); }

  document.querySelector("#knowledge-open").addEventListener("click", openDialog);
  close.addEventListener("click", closeDialog);
  back.addEventListener("click", closeDialog);
  clear.addEventListener("click", clearList);
  input.addEventListener("change", fileSelection);
  start.addEventListener("click", beginImport);
  dialog.addEventListener("cancel", cancelDialog);
  dropzone.addEventListener("dragover", dragOver);
  dropzone.addEventListener("dragleave", dragLeave);
  dropzone.addEventListener("drop", dropFiles);
}
