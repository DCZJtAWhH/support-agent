# 运行与维护脚本

本目录只保留应用评估与知识维护使用的三个入口。所有命令在仓库根目录执行，应用通过 `uv run poe dev` 启动，环境准备见 [README](../README.md)。

| 脚本 | 用途与外部影响 |
| --- | --- |
| `knowledge.py` | 知识导入、同步、检索、抽取和问题池查询；可能写库和调用付费服务。 |
| `calibrate_evidence.py` | 必须用 `--report` 指定完整真实报告，校准证据门控并写入指定配置。 |
| `eval_retrieval.py` | 当前四策略完整检索评估；应用后台也调用它。`--replay` 只复核既有明细。 |

```powershell
uv run --locked python scripts/knowledge.py stats
uv run --locked python scripts/eval_retrieval.py
uv run --locked python scripts/calibrate_evidence.py --report data/reports/retrieval-report.json
```

网页评估由应用写入 `data/rag-evaluations/`；直接执行检索评估默认写入 `data/reports/`，`--output` 可指定其他位置。目录按需创建，生成结果由 `.gitignore` 中的 `data/` 规则排除，不提交到版本库。校准参数默认写入受版本管理的 `config/evidence-confidence.json`。

自动化回归由 `tests/` 中的 pytest 和前端协议测试承担，见[评估文档](../docs/evaluation.md)。没有已发布报告时页面为空，不提供预置报告。已有累计费用账本仍保留，不因删除验证脚本而重置。
