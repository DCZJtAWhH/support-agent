class KnowledgeSearch:
    def __init__(self, retriever=None):
        """绑定检索后端，公开完整问题和多路政策检索。"""
        self.retriever = retriever

    async def query_faq(self, keyword: str, *, category=None, prepared=False) -> list[dict]:
        """用完整问题检索；Workflow 已归一的问题跳过旧的同义词扩展。"""
        if self.retriever is None:
            raise RuntimeError("知识检索尚未配置，请先建库。")
        options = {"category": category} if category else {}
        if prepared:
            options["prepared"] = {"normalized": keyword, "retrieval_text": keyword}
            options["apply_threshold"] = False
        return await self.retriever.search(keyword, **options)

    async def query_policies(self, question, queries, *, category=None) -> list[dict]:
        """将即时扩写的全部查询交给检索侧合并，强制只使用政策或手册条款。"""
        if self.retriever is None:
            raise RuntimeError("政策检索尚未配置，请先建库。")
        return await self.retriever.search_many(
            question,
            queries,
            category=category,
            apply_threshold=False,
        )
