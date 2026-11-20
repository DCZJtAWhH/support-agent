# 第三章 dense 知识库 demo 实施计划

依据：用户本轮功能需求、首轮八项推荐的确认，以及随后“只是一个 demo，能正常跑通即可”的范围纠偏。技术选型与验收不变，其余日常实现选择由代理采用简单默认值，不再逐轮访谈。

1. 文档切分入口：标题层级、递归段落/完整句、句子重叠、表格完整行与表头复制；不可切的超长句/行报错。默认正文 900 字、重叠约 100 字。FAQ 文档章节标题使用真实问法；政策使用章节标题，上级路径为 category。关键条款以显式标记为准。
2. 知识持久化与同步入口：映射已建两表，增加来源键/摘要/生效标记等必要字段，提供可重跑迁移。先保存 MySQL pending，再按主键 upsert Milvus、回填 done；来源文档更新与删除同步失效。单进程运行，命令间不并发执行。
3. 语义检索入口：硅基流动 BAAI/bge-m3，1024 维；Milvus 2.6 / PyMilvus 2.6，COSINE Top-3；原文从 MySQL 回查，返回原有 question/answer/category，异常沿用工具错误封装，无关键词后备通道。
4. 抽取发布入口：扫描已审核 JSONL 导出，分批使用现有聊天模型结构化抽取，先暂存再对本轮与既有知识整体去重；语义近邻仅在离线筛选，LLM 判断等价/冲突，同义合并问法、冲突留存。独立进程北京时间每天 02:00 执行，也支持手动单次执行；失败批次可重跑。
5. 验证：按入口逐个 red→green。切分入口验证内容/表头/边界；持久化同步入口验证中断恢复和幂等；工具入口验证保持契约的邮费语义召回；抽取发布入口验证暂存、归并、冲突与失败重跑。按用户 demo 范围纠偏直接选定这些测试入口，不再新增确认步骤。模型 Prompt 用标注评估集与实际模型运行验证；最后运行真实 MySQL、Docker Milvus、硅基流动的演示并记录结果。

官方查证：Context7 已查询 SQLAlchemy 2.0（项目 2.0.58，与文档 2.0.55 同系列）、LangChain 1.x（项目 langchain-openai/core 1.8.3）、PyMilvus 2.6.18 与 Milvus Docker 部署文档。嵌入请求直接使用已有 httpx 0.28.1 调用官方 REST，不增加模型运行库。BGE-M3 为 1024 维、8192 token；不传仅 Qwen 支持的 dimensions。实现时核对已安装接口与提供方实际返回。

参考：
- https://api-docs.siliconflow.cn/docs/api/embeddings-post
- https://huggingface.co/BAAI/bge-m3
- https://milvus.io/docs/install_standalone-docker-compose.md
- https://github.com/milvus-io/pymilvus/tree/v2.6.18
- https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- https://docs.langchain.com/oss/python/langchain/models
