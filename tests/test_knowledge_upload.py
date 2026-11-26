import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest


@asynccontextmanager
async def upload_harness(harness, embedding_handler=None):
    async def embed(request):
        if embedding_handler:
            return await embedding_handler(request)
        texts = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={
                "data": [{"index": i, "embedding": [1.0] + [0.0] * 1023} for i in range(len(texts))]
            },
        )

    class MilvusSDK:
        def __init__(self):
            self.rows = {}

        def upsert(self, collection, rows, **kwargs):
            self.rows.update({row["id"]: row for row in rows})

        def delete(self, collection, ids, **kwargs):
            for key in ids:
                self.rows.pop(key, None)

        def search(self, collection, **kwargs):
            return [[{"id": key, "distance": 1.0, "entity": row} for key, row in self.rows.items()]]

        def close(self):
            pass

    async with harness(siliconflow_api_key="test-key") as (client, _, app):
        async with httpx.AsyncClient(transport=httpx.MockTransport(embed)) as embeddings:
            app.state.repository.retriever.embeddings.client = embeddings
            app.state.repository.retriever.vectors.client = MilvusSDK()
            yield client, app


def test_upload_creates_citable_searchable_knowledge_without_duplicate_chunks(harness):
    async def scenario():
        async with upload_harness(harness) as (client, app):
            text = "# 水杯\n## KB-680容量是多少\nKB-680容量为680毫升。\n## 如何清洗\n请温水手洗。"
            files = {"file": ("水杯手册.md", text.encode("utf-8"), "text/markdown")}
            uploaded = await client.post("/knowledge/upload", files=files)
            assert uploaded.status_code == 201, uploaded.text
            result = uploaded.json()
            assert result["status"] == "ready" and result["chunk_count"] == 2
            assert result["source_ref"] == "upload:水杯手册.md"
            chunk = (await client.get(f"/knowledge/chunks/{result['chunk_ids'][0]}")).json()
            assert chunk["answer"] == "KB-680容量为680毫升。"
            assert chunk["section_path"] == "水杯 / KB-680容量是多少"
            replay = (await client.post("/knowledge/upload", files=files)).json()
            assert replay["chunk_ids"] == result["chunk_ids"]
            assert (await app.state.knowledge_store.stats())["total"] == 2
            found = await app.state.repository.retriever.search(
                "KB-680容量",
                strategy="bm25",
                prepared={"normalized": "KB-680容量", "retrieval_text": "KB-680容量"},
            )
            assert found[0]["chunk_id"] == result["chunk_ids"][0]
            assert "680毫升" in found[0]["answer"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "filename,content,status,message",
    [
        ("手册.pdf", b"%PDF-test", 422, "格式"),
        ("../手册.md", b"body", 422, "文件名"),
        ("手册.md", b"\xff\xfe", 422, "UTF-8"),
        ("空白.md", b" \n\t", 422, "正文"),
        ("标题.md", "# 只有标题".encode(), 422, "正文"),
        ("注释.md", b"<!-- note -->", 422, "正文"),
        pytest.param("二进制.md", b"\x00\x01\x02", 422, "文本", id="binary-text"),
        ("超长句.md", ("长" * 901).encode("utf-8"), 422, "完整句"),
        ("过大.md", b"x" * (1024 * 1024 + 1), 413, "1 MiB"),
        (
            "过多.md",
            ("\n".join(f"## 问题{i}\n正文。" for i in range(201))).encode("utf-8"),
            422,
            "200",
        ),
    ],
)
def test_invalid_upload_never_changes_authoritative_knowledge(
    harness, filename, content, status, message
):
    async def scenario():
        async with upload_harness(harness) as (client, app):
            response = await client.post("/knowledge/upload", files={"file": (filename, content)})
            assert response.status_code == status, response.text
            assert message in response.json()["detail"]
            assert (await app.state.knowledge_store.stats())["total"] == 0

    asyncio.run(scenario())


def test_upload_only_indexes_its_own_source_and_updates_same_name(harness):
    from support_agent.knowledge.chunks import KnowledgeDraft

    async def scenario():
        async with upload_harness(harness) as (client, app):
            store = app.state.knowledge_store
            unrelated = await store.replace_source(
                "markdown:other",
                [
                    KnowledgeDraft("其他", ["旧任务"], "此来源尚待命令行同步。"),
                ],
            )
            first = (
                await client.post(
                    "/knowledge/upload",
                    files={
                        "file": ("容量.txt", "初始容量为680毫升。".encode()),
                    },
                )
            ).json()
            assert (await store.get(unrelated[0])).vectorize_status == "pending"
            second = (
                await client.post(
                    "/knowledge/upload",
                    files={
                        "file": ("容量.txt", "最新容量为780毫升。".encode()),
                    },
                )
            ).json()
            assert second["chunk_ids"] == first["chunk_ids"]
            chunk = (await client.get(f"/knowledge/chunks/{first['chunk_ids'][0]}")).json()
            assert chunk["answer"] == "最新容量为780毫升。"

    asyncio.run(scenario())


def test_index_failure_preserves_original_and_retry_resumes_without_duplication(harness):
    async def scenario():
        failed = True

        async def embedding_handler(request):
            if failed:
                return httpx.Response(503, json={"error": "private-provider-detail"})
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [1.0] + [0.0] * 1023},
                    ]
                },
            )

        async with upload_harness(harness, embedding_handler) as (client, app):
            files = {"file": ("重试.md", "# 测试\n## 容量\n容量为680毫升。".encode())}
            response = await client.post("/knowledge/upload", files=files)
            assert response.status_code == 503
            detail = response.json()["detail"]
            assert detail["code"] == "index_pending"
            assert "已保存" in detail["message"]
            assert "private-provider-detail" not in response.text
            original_id = detail["chunk_ids"][0]
            assert (await app.state.knowledge_store.get(original_id)).vectorize_status == "pending"
            failed = False
            response = await client.post("/knowledge/upload", files=files)
            assert response.status_code == 201
            assert response.json()["chunk_ids"] == [original_id]
            assert (await app.state.knowledge_store.stats())["pending"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "text", ["# " + "长" * 256 + "\n## 问题\n正文。", "# 知识\n## " + "长" * 510 + "\n正文。"]
)
def test_upload_rejects_heading_beyond_mysql_limits_and_keeps_previous_version(harness, text):
    async def scenario():
        async with upload_harness(harness) as (client, _):
            created = (
                await client.post(
                    "/knowledge/upload",
                    files={
                        "file": ("existing.md", "# 知识\n## 问题\n原内容。".encode()),
                    },
                )
            ).json()
            rejected = await client.post(
                "/knowledge/upload",
                files={
                    "file": ("existing.md", text.encode()),
                },
            )
            assert rejected.status_code == 422
            assert "标题" in rejected.json()["detail"]
            previous = await client.get(f"/knowledge/chunks/{created['chunk_ids'][0]}")
            assert previous.json()["answer"] == "原内容。"

    asyncio.run(scenario())
