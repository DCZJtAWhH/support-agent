# 用原生 BM25 混合检索替代单路在线召回

在线检索使用 Milvus 原生 BM25 + dense + RRF，以及 bge-reranker-v2-m3 精排，取代单路 dense 召回；[ADR 0002](0002-knowledge-authority-and-recovery.md) 的 MySQL 原文权威、pending 重放与摘要回查继续有效。BM25 函数必须在建集合时定义，因此使用独立的 `knowledge_hybrid_v1` 集合并显式重建，保留既有 dense 集合；切换索引期间未同步的原文仍不可用。

引用使用来源内容快照，保留 chunk 主键、内容摘要、章节路径和原文链接；当前 chunk 已变更时原文接口说明版本差异。问题池按轮次幂等，避免断流重试重复计数。

当前聊天工作流保留精排阈值前 Top10 快照，先按校准的多信号证据分门控，再过滤低于校准相关分下限的证据；门控或生成充分性自评不通过时拒答。聊天引用编号由提示词约束，独立检索评估的 `GroundedAnswerer` 另有显式引用校验。
