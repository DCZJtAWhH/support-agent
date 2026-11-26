import asyncio
import json

import httpx
import pytest

from support_agent.knowledge.vectors import BGEEmbeddings


def test_embedding_uses_bge_dense_batch_and_restores_response_order():
    async def scenario():
        async def handler(request):
            assert str(request.url) == "https://api.siliconflow.cn/v1/embeddings"
            body = json.loads(request.content)
            assert body == {
                "model": "BAAI/bge-m3",
                "input": ["邮费是多少", "运费标准"],
                "encoding_format": "float",
            }
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            embedding = BGEEmbeddings("test-key", client=client, dimensions=2)
            assert await embedding.embed(["邮费是多少", "运费标准"]) == [[1.0, 0.0], [0.0, 1.0]]

    asyncio.run(scenario())


def test_malformed_embedding_response_fails_before_any_vector_write():
    async def scenario():
        async def handler(request):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError, match="维度|数量"):
                await BGEEmbeddings("test-key", client=client).embed(["测试"])

    asyncio.run(scenario())
