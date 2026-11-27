# 评估与验证

自动化回归验证接口和执行契约，标注评估验证模型与检索表现；离线替身通过不能证明真实模型语义质量。标注集用途见 [evals/README.md](../evals/README.md)。

## 不调用真实模型的检查

在仓库根目录运行：

```powershell
uv run --locked pytest -q
node --test tests/web/stream.test.mjs
uv run --locked ruff check src scripts tests
uv run --locked ruff format --check src scripts tests
```

pytest 使用独立数据库与外部协议替身，覆盖会话幂等、工单确认、流中断、预算、检索、MCP、审核发布、模块边界及持久数据兼容恢复。前端协议测试需要 Node.js 22+。

pytest 中的浏览器用例使用 Playwright；尚未安装浏览器时运行 `uv run --locked playwright install chromium`。独立的阶段验收脚本已移除。

## 完整检索评估

在 `/evaluations` 查看报告、按问题类型及难度筛选、展开逐题原文与事实判断。浏览已有报告不调用模型；“重新评估”会对当前 `evals/retrieval.jsonl` 运行四策略完整真实评估，并调用生成及逐事实评委模型。

当前标注集为 20 道开发回归题，覆盖直接问法、口语、同义词、条件、型号、分类、跨块、歧义、未知和安全场景。上传知识不会自动增加对应标注题。

四种策略为 `dense`、`bm25`、`hybrid`、`hybrid_rerank`，共享权威语料和问题理解结果：

| 指标 | 含义与限制 |
| --- | --- |
| Recall@1/5/10/50、MRR | 仅对有 ground truth 的题计算；相关性身份由来源和章节组成，不绑定数据库自增 ID，同章节多块去重计分。 |
| Faithfulness | 评委按本轮召回原文判断回答的原子事实是否受支持；无事实的拒答不计作满分。 |
| 拒答正确率、作答率与样本量 | 同时报告，避免全拒答得到虚高分。 |

精排策略只保留 10 块，因此它的 Recall@50 仍只覆盖这 10 块。各策略原生分数不可互换，精排阈值不能套用于 dense、BM25 或 RRF 分。指标与阈值使用开发集校准，不能视为独立留出评估；同提供方的模型评委结果应人工抽查。

CLI 入口：

```powershell
# 真实模型评估，写入一份新报告。
uv run --locked python scripts/eval_retrieval.py --output data/reports/retrieval-report.json
# 用已保留的逐条结果重算指标，不调用真实模型。
uv run --locked python scripts/eval_retrieval.py --replay data/reports/retrieval-report.rows.jsonl --output data/reports/retrieval-replay.json
# 依据真实精排报告重新校准证据门控。
uv run --locked python scripts/calibrate_evidence.py --report data/reports/retrieval-report.json
```

网页与后台定时任务使用同一完整流水线，单进程只允许一个任务。`EVAL_INTERVAL_SECONDS=0` 关闭定时触发。评估期间不要修改知识或标注集，也不要并行启动外部 CLI 评估。

## 报告与恢复

网页运行写入 `data/rag-evaluations/<运行UUID>/report.json`、同名 Markdown 和 rows JSONL，并在 `process.log` 保存子进程输出。完整验证知识与标注版本后才更新 `latest.json`；`job.json` 保存可恢复的进度，服务退出会终止自身子进程，重启后将未完成任务标为中断。失败继续展示上一份已发布报告。

页面只读取 `latest.json` 指向的已发布报告。没有指针或目标报告缺失时，接口返回 `report: null` 和空明细，页面展示“暂无已完成报告”；不自动读取其他文件或预置报告。首次评估未完成、失败或中断时继续为空。已有已发布报告时，后续失败不覆盖它。知识或标注版本不一致时页面标记历史报告。MySQL 还保存完成评估的指标历史；报告文件用于逐题证据，不替代数据库指标记录。

报告属于本地运行数据，统一放在已被 Git 忽略的 `data/`：

```text
data/
├── rag-evaluations/
│   ├── latest.json             最新已发布报告的运行 ID
│   ├── job.json                最近任务的进度和状态
│   └── <运行UUID>/
│       ├── report.json         指标、逐题回答和证据，供评估页读取
│       ├── report.md           供人工阅读
│       ├── report.rows.jsonl   供离线回放
│       └── process.log         评估子进程输出
└── reports/                    CLI 评估报告及本地验证产物
```

直接执行 `eval_retrieval.py` 默认输出 `data/reports/retrieval-report.*`，不更新网页发布指针，必要时自行创建目录。未被指针引用的历史报告不会自动展示。

这些生成结果不随仓库分发，新克隆的项目没有报告。删除本地评估报告后页面为空，但不会自动删除 MySQL 中的指标历史。需要保留逐题证据和任务恢复状态时自行备份。

校准脚本必须通过 `--report` 显式指定真实报告。`config/evidence-confidence.json` 继续由 Git 管理，保存参数、来源路径和摘要；来源报告保留在本地或备份中，应用使用参数时不需要读取该来源文件。

## 人工语义复核

其他能力的模型语义复核使用 `evals/` 中保留的标注集，通过当前接口或页面逐条核查。问题整理需检查型号、条件和否定保留、归并落点及不可信输入处理；售后提取需逐字段核对，期望方案允许同义表达。

`scripts/` 仅保留完整检索评估、知识维护和证据校准入口，见[脚本索引](../scripts/README.md)。真实调用的费用保护见[配置说明](configuration.md#真实模型测试费用保护)。
