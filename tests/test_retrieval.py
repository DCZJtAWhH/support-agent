import asyncio
import json

import httpx
import pytest

from support_agent.database import Database
from support_agent.knowledge_chunks import KnowledgeDraft
from support_agent.knowledge_store import KnowledgeStore, synchronize
from support_agent.vectors import BGEEmbeddings, MilvusVectors


def test_hybrid_retrieves_model_with_prefilter_reranks_and_exposes_source():
    from support_agent.retrieval import HybridRetriever
    from support_agent.vectors import BGEReranker

    async def scenario():
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        store = KnowledgeStore(db)
        ids = await store.replace_source(
            "docs:model.md",
            [KnowledgeDraft("水杯", ["BX-500 容量"], "BX-500 容量为500毫升。", "水杯 > BX-500")],
        )
        rows = {}

        class MilvusSDK:
            def upsert(self, collection, data, **kwargs):
                rows.update({r["id"]: r for r in data})

            def hybrid_search(self, collection, reqs, ranker, limit, **kwargs):
                assert [r.limit for r in reqs] == [50, 50]
                assert [r.anns_field for r in reqs] == ["vector", "sparse"]
                assert all(r.expr == 'category == "水杯"' for r in reqs)
                assert reqs[1].data == ["BX-500容量是多少"]
                return [[{"id": ids[0], "distance": 0.03, "entity": rows[ids[0]]}]]

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path.endswith("embeddings"):
                return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
            assert body["model"] == "BAAI/bge-reranker-v2-m3"
            assert body["top_n"] == 1
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.97}]})

        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                embeddings = BGEEmbeddings("key", client=client, dimensions=2)
                vectors = MilvusVectors()
                vectors.client = MilvusSDK()
                await synchronize(store, embeddings, vectors)
                result = await HybridRetriever(
                    store,
                    embeddings,
                    vectors,
                    BGEReranker("key", client=client),
                ).search("BX-500容量是多少", category="水杯")
                assert len(result) == 1
                assert result[0]["chunk_id"] == ids[0]
                assert result[0]["section_path"] == "水杯 > BX-500"
                assert result[0]["source_ref"] == "docs:model.md"
                assert result[0]["score"] == 0.97
                assert rows[ids[0]]["text"].count("BX-500 容量为500毫升。") == 1
                assert rows[ids[0]]["category"] == "水杯"
        finally:
            await db.close()

    asyncio.run(scenario())


def test_query_understanding_preserves_model_numbers_and_only_expands_at_search():
    from support_agent.retrieval import QueryUnderstanding

    class LanguageAPI:
        async def ask(self, schema, prompt, payload):
            assert "history" not in payload
            return schema(normalized="BX-500的邮费是多少", synonyms=["运费", "配送费"])

    async def scenario():
        result = await QueryUnderstanding(LanguageAPI()).prepare("那个BX-500寄过来多少钱啊")
        assert result == {
            "normalized": "BX-500的邮费是多少",
            "retrieval_text": "BX-500的邮费是多少 运费 配送费",
        }

    asyncio.run(scenario())


def test_query_rewrite_cannot_change_explicit_model_or_quantity():
    from support_agent.retrieval import QueryUnderstanding

    class LanguageAPI:
        async def ask(self, schema, prompt, payload):
            return schema(normalized="BX-600的容量", synonyms=[])

    async def scenario():
        with pytest.raises(ValueError, match="型号|数字"):
            await QueryUnderstanding(LanguageAPI()).prepare("BX-500多大")

    asyncio.run(scenario())


def test_hybrid_never_returns_stale_pending_or_inactive_original():
    from support_agent.retrieval import HybridRetriever
    from support_agent.vectors import BGEReranker

    async def scenario():
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        store, rows = KnowledgeStore(db), {}

        class MilvusSDK:
            def upsert(self, collection, data, **kwargs):
                rows.update({r["id"]: r for r in data})

            def hybrid_search(self, collection, **kwargs):
                return [[{"id": key, "distance": 0.03, "entity": row} for key, row in rows.items()]]

        async def handler(request):
            if request.url.path.endswith("embeddings"):
                return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.95}]})

        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                vectors = MilvusVectors()
                vectors.client = MilvusSDK()
                embeddings = BGEEmbeddings("key", client=client, dimensions=2)
                retriever = HybridRetriever(
                    store, embeddings, vectors, BGEReranker("key", client=client)
                )
                await store.replace_source("policy", [KnowledgeDraft("配送", ["邮费"], "运费8元")])
                await synchronize(store, embeddings, vectors)
                await store.replace_source("policy", [KnowledgeDraft("配送", ["邮费"], "运费9元")])
                assert await retriever.search("邮费") == []
                await synchronize(store, embeddings, vectors)
                assert (await retriever.search("邮费"))[0]["answer"] == "运费9元"
                await store.replace_source("policy", [])
                assert await retriever.search("邮费") == []
        finally:
            await db.close()

    asyncio.run(scenario())
