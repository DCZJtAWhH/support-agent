"""提供硅基流动 BGE-M3 dense 嵌入与 Milvus 索引访问。"""

import asyncio
import json
import math
from operator import itemgetter

import httpx


class BGEEmbeddings:
    def __init__(
        self,
        api_key: str,
        *,
        client=None,
        dimensions: int = 1024,
        timeout: float = 30,
    ):
        """配置固定模型与向量维度；注入客户端用于 HTTP 边界测试，不记录凭据。"""
        self.api_key, self.client = api_key, client
        self.dimensions, self.timeout = dimensions, timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量获取 dense 向量，校验数量、索引、维度与有限值，并恢复输入顺序。"""
        if not texts:
            return []
        if not self.api_key:
            raise ValueError("请配置 SILICONFLOW_API_KEY。")
        if any(not text.strip() for text in texts):
            raise ValueError("嵌入输入不能为空。")
        if self.client is None:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                return await self._request(client, texts)
        return await self._request(self.client, texts)

    async def _request(self, client, texts):
        """执行固定官方端点请求；提供方异常由工具层脱敏，不输出响应正文。"""
        response = await client.post(
            "https://api.siliconflow.cn/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "BAAI/bge-m3", "input": texts, "encoding_format": "float"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        if len(data) != len(texts) or sorted(r["index"] for r in data) != list(range(len(texts))):
            raise ValueError("嵌入响应数量或索引不匹配。")
        result = [r["embedding"] for r in sorted(data, key=itemgetter("index"))]
        if any(
            len(v) != self.dimensions
            or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in v)
            or not any(v)
            for v in result
        ):
            raise ValueError("嵌入响应维度或数值无效。")
        return result


class MilvusVectors:
    def __init__(
        self, uri: str = "http://127.0.0.1:19530", collection: str = "knowledge_hybrid_v1"
    ):
        """延迟连接本地 Milvus，避免未使用检索时阻塞应用启动。"""
        self.uri, self.collection, self.client = uri, collection, None
        self.lock = asyncio.Lock()

    async def _connect(self):
        """在线程中建立同步 SDK 连接，并创建 dense 与原生 BM25 混合集合。"""
        async with self.lock:
            if self.client is None:
                self.client = await asyncio.to_thread(self._open)
        return self.client

    def _open(self):
        """检查或初始化集合；已有同名不兼容集合明确报错，不自动删除。"""
        from pymilvus import DataType, Function, FunctionType, MilvusClient

        client = MilvusClient(uri=self.uri, timeout=10)
        try:
            if not client.has_collection(self.collection, timeout=10):
                schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
                schema.add_field("id", DataType.INT64, is_primary=True)
                schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
                schema.add_field(
                    "text",
                    DataType.VARCHAR,
                    max_length=65535,
                    enable_analyzer=True,
                    analyzer_params={"type": "chinese"},
                )
                schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
                schema.add_field("category", DataType.VARCHAR, max_length=1024)
                schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
                schema.add_function(
                    Function(
                        name="text_bm25",
                        function_type=FunctionType.BM25,
                        input_field_names=["text"],
                        output_field_names=["sparse"],
                    )
                )
                indexes = client.prepare_index_params()
                indexes.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
                indexes.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
                client.create_collection(
                    collection_name=self.collection,
                    schema=schema,
                    index_params=indexes,
                    consistency_level="Strong",
                    timeout=30,
                )
            schema = client.describe_collection(self.collection, timeout=10)
            vector = next(f for f in schema["fields"] if f["name"] == "vector")
            names = {f["name"] for f in schema["fields"]}
            if (
                int(vector["params"]["dim"]) != 1024
                or schema.get("auto_id")
                or not {"text", "sparse", "category", "content_hash"} <= names
                or not schema.get("functions")
            ):
                raise ValueError(
                    "集合不兼容，请配置新的 MILVUS_HYBRID_COLLECTION 并 sync --rebuild。"
                )
            client.load_collection(self.collection, timeout=30)
            return client
        except BaseException:
            client.close()
            raise

    async def upsert(self, rows: list[dict]):
        """按 MySQL 主键插入或覆盖向量，重复执行不产生新业务主键。"""
        client = await self._connect()
        await asyncio.to_thread(client.upsert, self.collection, rows, timeout=30)

    async def delete(self, ids: list[int]):
        """删除已失效知识的向量；重复删除允许重放。"""
        client = await self._connect()
        await asyncio.to_thread(client.delete, self.collection, ids=ids, timeout=30)

    async def search(self, vector: list[float], limit: int) -> list[tuple[int, float, str]]:
        """仅执行 COSINE dense Top-K，返回主键、相似度与原文摘要。"""
        client = await self._connect()
        results = await asyncio.to_thread(
            client.search,
            self.collection,
            data=[vector],
            limit=limit,
            output_fields=["content_hash"],
            search_params={"metric_type": "COSINE"},
            anns_field="vector",
            consistency_level="Strong",
            timeout=10,
        )
        return [
            (int(hit["id"]), float(hit["distance"]), hit["entity"].get("content_hash", ""))
            for hit in results[0]
        ]

    async def retrieve(self, vector, text: str, *, strategy="hybrid", category=None):
        """两路统一先按品类过滤，各召回50；混合策略交由 Milvus RRF 融合。"""
        from pymilvus import AnnSearchRequest, RRFRanker

        client = await self._connect()
        expression = (
            "" if category is None else "category == " + json.dumps(category, ensure_ascii=False)
        )
        requests = []
        if strategy != "bm25":
            requests.append(
                AnnSearchRequest(
                    [vector],
                    "vector",
                    {"metric_type": "COSINE"},
                    50,
                    expr=expression,
                )
            )
        if strategy != "dense":
            requests.append(
                AnnSearchRequest(
                    [text],
                    "sparse",
                    {"metric_type": "BM25"},
                    50,
                    expr=expression,
                )
            )
        options = dict(output_fields=["content_hash"], consistency_level="Strong", timeout=30)
        if len(requests) == 2:
            results = await asyncio.to_thread(
                client.hybrid_search,
                self.collection,
                reqs=requests,
                ranker=RRFRanker(k=60),
                limit=100,
                **options,
            )
        else:
            request = requests[0]
            results = await asyncio.to_thread(
                client.search,
                self.collection,
                data=request.data,
                anns_field=request.anns_field,
                search_params=request.param,
                filter=expression,
                limit=50,
                **options,
            )
        return [
            (int(h["id"]), float(h["distance"]), h["entity"].get("content_hash", ""))
            for h in results[0]
        ]

    async def close(self):
        """释放已打开的 SDK 连接，不触发延迟连接。"""
        if self.client is not None:
            await asyncio.to_thread(self.client.close)
            self.client = None


class KnowledgeRetriever:
    def __init__(self, store, embeddings, vectors, *, top_k: int = 3, min_score: float = 0.4):
        """绑定单路召回依赖，低相关阈值可用标注集校准。"""
        self.store, self.embeddings, self.vectors = store, embeddings, vectors
        self.top_k, self.min_score = top_k, min_score

    async def search(self, question: str) -> list[dict]:
        """向量召回后回查有效且版本一致的 MySQL 原文，保持旧结果字段。"""
        vector = (await self.embeddings.embed([question]))[0]
        hits = await self.vectors.search(vector, self.top_k)
        matches = []
        for key, score, content_hash in hits:
            row = await self.store.get(key)
            if (
                score >= self.min_score
                and row
                and row.is_active
                and row.vectorize_status == "done"
                and row.content_hash == content_hash
            ):
                matches.append(
                    {
                        "question": row.questions,
                        "answer": row.answer,
                        "category": row.category,
                    }
                )
        return matches


class BGEReranker:
    def __init__(self, api_key: str, *, client=None, timeout: float = 30):
        """绑定固定 bge-reranker-v2-m3 HTTP 服务，不在失败时替换模型或跳过精排。"""
        self.api_key, self.client, self.timeout = api_key, client, timeout

    async def rank(self, question: str, documents: list[str], top_n: int = 10):
        """返回原文索引与相关分；完整校验数量、唯一索引、范围及有限分数。"""
        if not documents:
            return []
        if not self.api_key:
            raise ValueError("请配置 SILICONFLOW_API_KEY。")
        if self.client is None:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                return await self._request(client, question, documents, top_n)
        return await self._request(self.client, question, documents, top_n)

    async def _request(self, client, question, documents, top_n):
        """请求官方精排端点；异常响应不带入提示或用户输出。"""
        count = min(top_n, len(documents))
        response = await client.post(
            "https://api.siliconflow.cn/v1/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": question,
                "documents": documents,
                "top_n": count,
                "return_documents": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json()["results"]
        indices = [r["index"] for r in rows]
        if (
            len(rows) != count
            or len(set(indices)) != count
            or any(type(i) is not int or not 0 <= i < len(documents) for i in indices)
            or any(
                not isinstance(r["relevance_score"], (int, float))
                or not math.isfinite(r["relevance_score"])
                or not 0 <= r["relevance_score"] <= 1
                for r in rows
            )
        ):
            raise ValueError("精排响应的索引、数量或分数无效。")
        return sorted(
            [(r["index"], float(r["relevance_score"])) for r in rows],
            key=itemgetter(1),
            reverse=True,
        )
