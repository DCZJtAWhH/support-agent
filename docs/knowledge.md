# 知识维护与审核

MySQL 是知识原文权威源，Milvus 是可重放的检索索引。正式知识按知识块保存分类、问法、答复、来源、章节路径和内容摘要；同一原文不为多种问法拆存副本。

## 文件导入

聊天页“上传知识”支持多文件逐个导入。也可以调用接口：

```powershell
curl.exe -F "file=@examples/knowledge/docs/models.md;type=text/markdown" http://127.0.0.1:8010/knowledge/upload
```

- 仅支持 UTF-8 的 `.md`、`.markdown`、`.txt`，每文件最多 1 MiB、200 个知识块。
- Markdown 按标题形成章节路径与分类；TXT 可直接写正文。PDF、Word 和图片未接入。
- 分类长度最多 255 字符，章节路径最多 512 字符；不可拆分的完整句或表格行也不能超过 900 字切块上限。
- 网页来源为 `upload:<文件名>`；同名上传更新同一来源，重复上传复用主键。CLI 来源独立，不会被网页同名文件覆盖。
- “清空列表”只清除上传窗口的显示记录，不删除已入库知识。

索引失败时原文已经保存，接口返回 503，页面保留“重试导入”。可以重试该文件，或用 CLI 同步 pending；不要将失败提示理解为原文完全没有保存。

## CLI 维护

所有命令在仓库根目录执行，使用现有 `.env`。命令可能调用付费模型或嵌入服务。

```powershell
# 导入演示 FAQ 和文档，只保存原文。
uv run --locked python scripts/knowledge.py build --no-sync
# 同步 pending；可重复执行以恢复中断。
uv run --locked python scripts/knowledge.py sync
# 首次建混合集合、切换集合或需要完整重建时执行。
uv run --locked python scripts/knowledge.py sync --rebuild
# 查询与状态。
uv run --locked python scripts/knowledge.py query "BX-500容量是多少" --strategy hybrid_rerank
uv run --locked python scripts/knowledge.py stats
uv run --locked python scripts/knowledge.py low-confidence
```

`build --docs <目录>` 指定自有文档目录，但同时会导入内置 FAQ。检索策略可为 `dense`、`bm25`、`hybrid`、`hybrid_rerank`，`--category` 可限制分类。仅修改集合名而不重建，不会自动补齐旧 done 块的索引。

先提交 pending 原文，再按同一 MySQL 主键向 Milvus upsert，成功后回填向量标识并置 done。进程中断后重跑 sync；即使向量写入已成功、状态回填失败，重放也不会产生另一份主键。回查时还核对原文摘要和生效状态，避免旧索引匹配新原文。

保持单写入任务：网页上传、CLI 建库／抽取／同步、审核发布不要并行修改同一知识库；评估期间也应停止知识写入。

## 检索、证据与引用

Milvus 使用中文 analyzer 和服务端 BM25，与 BGE-M3 dense 各召回 50 个候选，原生 RRF（`k=60`）融合，再由 `BAAI/bge-reranker-v2-m3` 精排取 10 块。模型上下文还受独立证据投影和预算约束。

普通知识查询使用完整问题；售后政策查询先核实订单，再扩写多路查询、去重和统一精排。订单缺失时先等待选择，不凭空猜测。

聊天工作流保留精排阈值前 Top10 结果，用于召回快照和证据门控。门控综合精排 Top1、有效证据数量和 Top1/Top2 分差，参数来自[校准配置](../config/evidence-confidence.json) 的 `weights`、`relevance_floor` 和 `threshold`。未通过时直接兜底并记录问题；通过后按 `relevance_floor` 过滤交给模型的证据。这个分数不是概率。

聊天生成还要通过充分性自评，证据不足时拒答并记录问题。引用编号由提示词约束，当前聊天流程没有独立的编号合法性校验；检索评估使用的 `GroundedAnswerer` 会显式检查引用是否缺失或越界。

聊天引用保存当时原文快照。来源链接显示当前有效原文，知识已更新或失效时明确提示版本差异；不能把更新后的内容当成旧回答的实际依据。

## 从可信历史对话抽取

```powershell
uv run --locked python scripts/knowledge.py mine --history examples/knowledge/history
uv run --locked python scripts/knowledge.py staging
```

候选先暂存，检查与去重后发布；同义问法可合并，冲突知识保留待核实。个人订单信息和未经审核的对话不能直接成为正式知识。`mine --retry-conflicts` 重新核对冲突候选；`schedule` 可启动北京时间每日 02:00 的独立抽取任务，仍须遵守单写入约束。

## 问题池与人工审核

未解决原问来自证据门控、生成和反馈三个入口，同一轮幂等计数，多个入口保留在 `entry_points`。检索后即保存阈值前 Top10 原文、分数和来源；`[]` 表示检索无结果，`null` 表示没有检索记录。反馈按原请求读取持久快照，不用最近轮次代替。

整理模型保留原话中的型号、数字、否定、主体和适用条件，只生成待核实示例答案。语义去重比较完整候选队列；型号、部件、地区或诉求不同，不能仅因文字相似就合并。

在 `/reviews` 查看原问与召回证据，填写人工核准答案后通过，或驳回。用户原话、召回文本与待审内容均是不可信数据，不能作为系统指令执行。

通过审核时先保存核准答案与 `publishing`，以稳定来源 `review:<审核ID>` 发布，再同步索引。同步失败时保留原答案重试；完成前不能宣称已经可检索。自动整理与手动 `/api/flywheel/process` 使用同一流水线，自动间隔设为 0 时仍可手动处理。
