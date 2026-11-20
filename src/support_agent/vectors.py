"""提供硅基流动 BGE-M3 dense 嵌入与 Milvus 索引访问。"""

import asyncio
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
    def __init__(self, uri: str = "http://127.0.0.1:19530", collection: str = "knowledge"):
        """延迟连接本地 Milvus，避免未使用检索时阻塞应用启动。"""
        self.uri, self.collection, self.client = uri, collection, None
        self.lock = asyncio.Lock()

    async def _connect(self):
        """在线程中建立同步 SDK 连接，并创建自带整数主键的 1024 维集合。"""
        async with self.lock:
            if self.client is None:
                self.client = await asyncio.to_thread(self._open)
        return self.client

    def _open(self):
        """检查或初始化集合；已有同名不兼容集合明确报错，不自动删除。"""
        from pymilvus import MilvusClient

        client = MilvusClient(uri=self.uri, timeout=10)
        try:
            if not client.has_collection(self.collection, timeout=10):
                client.create_collection(
                    collection_name=self.collection,
                    dimension=1024,
                    metric_type="COSINE",
                    auto_id=False,
                    consistency_level="Strong",
                    timeout=30,
                )
            schema = client.describe_collection(self.collection, timeout=10)
            vector = next(f for f in schema["fields"] if f["name"] == "vector")
            if int(vector["params"]["dim"]) != 1024 or schema.get("auto_id"):
                raise ValueError("knowledge 集合必须为自带主键的1024维 dense 索引。")
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
            consistency_level="Strong",
            timeout=10,
        )
        return [
            (int(hit["id"]), float(hit["distance"]), hit["entity"].get("content_hash", ""))
            for hit in results[0]
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
