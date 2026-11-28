# 标注集索引

这些 JSONL 文件是当前模型与检索的回归输入，包含预期字段、相关来源、答案要点或人工判据。运行步骤、指标和报告管理见[评估与验证](../docs/evaluation.md)。

| 文件 | 用途 |
| --- | --- |
| `after_sales.jsonl` | 售后字段提取，覆盖缺失、歧义、多订单、前导零和指令注入。订单号与诉求分类精确匹配，期望方案允许同义表达。 |
| `tools.jsonl` | 工具选择、FAQ 参数和知识命中。 |
| `mcp-tools.jsonl` | MCP 工具发现与选择、参数、工单确认等工具平台场景。 |
| `router.jsonl` | 问题理解、指代、意图分类与分流。 |
| `workflow.jsonl` | 工作流路径、工具、步数、引用与拒答等端到端判据。 |
| `knowledge.jsonl` | 知识语义召回、条件问题和无关问题拒答。 |
| `knowledge-mining.jsonl` | 可信历史对话抽取、同义合并、冲突暂存及无效来源排除。 |
| `retrieval.jsonl` | 四策略检索的开发回归集；包含问题类型、难度、相关章节、答案要点与预期拒答。 |
| `context-summaries.jsonl` | 上下文摘要应保留的事实、禁止内容和长度约束。 |
| `context-summary-responses.jsonl` | 摘要人工参考输出，供离线标注校验，不代表真实模型结果。 |
| `flywheel.jsonl` | 知识缺口标准化、必要条件保留、语义归并和不可信输入处理。 |

演示知识与历史对话位于 [`examples/knowledge/`](../examples/knowledge/)，不是实际商家政策。数据结构检查和 HTTP 替身测试不能代替真实 Prompt 语义评估；修改语料或标注后应重新评估并核对报告版本。
