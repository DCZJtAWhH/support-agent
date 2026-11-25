import asyncio

from support_agent.database import Database
from support_agent.repositories import ChatRepository


def test_problem_pool_preserves_retrieval_snapshot_and_request_idempotency():
    async def scenario():
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        repo = ChatRepository(db)
        try:
            await repo.begin_request("conversation-1", "request-1", "杯盖怎么清洗？")
            snapshot = [{"chunk_id": 7, "text": "杯盖使用温水清洗。", "score": 0.04}]
            await repo.record_retrieval_snapshot("conversation-1", "request-1", snapshot)
            await repo.record_low_confidence(
                "conversation-1", "request-1", "杯盖怎么清洗？", "workflow_gate", "证据不足"
            )
            await repo.record_low_confidence(
                "conversation-1", "request-1", "杯盖怎么清洗？", "feedback", "未解决"
            )
            rows = await repo.low_confidence_questions()
            assert len(rows) == 1
            assert rows[0]["retrieval_snapshot"] == snapshot
            assert rows[0]["entry_points"] == ["workflow_gate", "feedback"]
            assert rows[0]["matched_review_id"] is None
        finally:
            await db.close()

    asyncio.run(scenario())


def test_review_api_processes_pool_exposes_sources_and_rejects_without_publishing(harness):
    async def scenario():
        from uuid import uuid4

        async with harness() as (client, _, app):
            app.state.flywheel.language = FlywheelLanguage()
            cid, rid = str(uuid4()), str(uuid4())
            await app.state.repository.begin_request(cid, rid, "杯盖怎么洗")
            await app.state.repository.record_low_confidence(
                cid, rid, "杯盖怎么洗", "generation", "不足"
            )
            response = await client.post("/api/flywheel/process", json={})
            assert response.status_code == 200
            rows = (await client.get("/api/review-queue?status=pending")).json()
            assert len(rows) == 1
            detail = (await client.get(f"/api/review-queue/{rows[0]['id']}")).json()
            assert detail["questions"][0]["question"] == "杯盖怎么洗"
            result = await client.post(
                f"/api/review-queue/{rows[0]['id']}/decision",
                json={"decision": "rejected", "approved_answer": ""},
            )
            assert result.status_code == 200
            assert (await client.get("/api/review-queue?status=pending")).json() == []
            assert (await client.get("/api/review-queue?status=rejected")).json()[0][
                "status"
            ] == "rejected"
            assert await app.state.knowledge_store.active() == []

    asyncio.run(scenario())


def test_feedback_api_uses_original_turn_and_recovers_legacy_retrieval(harness):
    async def scenario():
        import json
        from uuid import uuid4

        async with harness() as (client, _, app):
            cid, rid = str(uuid4()), str(uuid4())
            repo = app.state.repository
            await repo.begin_request(cid, rid, "旧轮次的商品问题")
            snapshot = [{"chunk_id": 3, "text": "旧轮次召回原文", "score": 0.6}]
            await repo.save_message(cid, rid, 1, "workflow", json.dumps({"sources": snapshot}))
            await repo.save_message(cid, rid, 1000000, "assistant", "旧回复")
            await repo.set_request_state(rid, "completed")
            body = {"conversation_id": cid, "request_id": rid, "value": "down"}
            wrong = await client.post(
                "/api/feedback", json={**body, "conversation_id": str(uuid4())}
            )
            assert wrong.status_code == 404
            for _ in range(2):
                result = await client.post("/api/feedback", json=body)
                assert result.status_code == 200
            questions = await repo.low_confidence_questions()
            assert len(questions) == 1
            assert questions[0]["question"] == "旧轮次的商品问题"
            assert questions[0]["retrieval_snapshot"] == snapshot
            assert questions[0]["entry_point"] == "feedback"

    asyncio.run(scenario())


class FlywheelLanguage:
    async def ask(self, schema, prompt, payload):
        if payload["task"] == "normalize":
            return schema.model_validate(
                {
                    "question": "杯盖如何清洗？",
                    "example_answer": "待核实：请确认杯盖清洗方式。",
                }
            )
        return schema.model_validate({"matched_id": payload["candidates"][0]["id"]})


class Embeddings:
    async def embed(self, texts):
        return [[0.5, 0.5] for _ in texts]


class Vectors:
    def __init__(self):
        self.rows = {}
        self.fail = False

    async def upsert(self, rows):
        if self.fail:
            raise RuntimeError("索引不可用")
        self.rows.update({row["id"]: row for row in rows})


def test_approval_publishes_only_human_answer_and_retry_recovers_pending_index():
    async def scenario():
        import pytest

        from support_agent.flywheel import FlywheelService
        from support_agent.knowledge_store import KnowledgeStore

        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        repo, vectors = ChatRepository(db), Vectors()
        store = KnowledgeStore(db)
        service = FlywheelService(
            repo, FlywheelLanguage(), store=store, embeddings=Embeddings(), vectors=vectors
        )
        try:
            await repo.begin_request("c", "r", "杯盖怎么洗")
            await repo.record_low_confidence("c", "r", "杯盖怎么洗", "feedback", "未解决")
            await service.process_pending()
            review_id = (await service.list_reviews())[0]["id"]
            with pytest.raises(ValueError, match="核准答案"):
                await service.decide(review_id, "approved", " ")
            vectors.fail = True
            with pytest.raises(RuntimeError):
                await service.decide(review_id, "approved", "请用温水手洗杯盖。")
            assert (await service.review_details(review_id))["status"] == "publishing"
            assert (await store.stats())["pending"] == 1
            vectors.fail = False
            await service.decide(review_id, "approved", "请用温水手洗杯盖。")
            await service.decide(review_id, "approved", "请用温水手洗杯盖。")
            knowledge = await store.active()
            assert len(knowledge) == 1
            assert knowledge[0].answer == "请用温水手洗杯盖。"
            assert knowledge[0].vectorize_status == "done"
            assert len(vectors.rows) == 1
            assert (await service.review_details(review_id))["status"] == "approved"
        finally:
            await db.close()

    asyncio.run(scenario())


def test_semantic_duplicates_become_one_review_with_all_original_questions():
    async def scenario():
        from support_agent.flywheel import FlywheelService

        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        repo = ChatRepository(db)
        service = FlywheelService(repo, FlywheelLanguage())
        try:
            for index, question in enumerate(["杯盖怎么洗啊", "盖子该怎么清洗"]):
                rid = f"request-{index}"
                await repo.begin_request("conversation-1", rid, question)
                await repo.record_low_confidence(
                    "conversation-1", rid, question, "generation", "不足"
                )
            assert await service.process_pending() == 2
            assert await service.process_pending() == 0
            rows = await service.list_reviews()
            assert len(rows) == 1
            assert rows[0]["occurrence_count"] == 2
            details = await service.review_details(rows[0]["id"])
            assert [q["question"] for q in details["questions"]] == [
                "杯盖怎么洗啊",
                "盖子该怎么清洗",
            ]
            pooled = await repo.low_confidence_questions()
            assert {q["matched_review_id"] for q in pooled} == {rows[0]["id"]}
        finally:
            await db.close()

    asyncio.run(scenario())


def test_problem_pool_is_automatically_processed_and_model_failure_is_retried(tmp_path):
    async def scenario():
        from support_agent.flywheel import FlywheelService

        class FlakyLanguage(FlywheelLanguage):
            def __init__(self):
                self.failed = False

            async def ask(self, *args):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("临时模型故障")
                return await super().ask(*args)

        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'scan.db'}")
        await db.initialize()
        repo = ChatRepository(db)
        service = FlywheelService(repo, FlakyLanguage())
        try:
            await repo.begin_request("c", "r", "杯盖怎么洗")
            await repo.record_low_confidence("c", "r", "杯盖怎么洗", "workflow_gate", "不足")
            service.start_worker(0.01)
            rows = []
            for _ in range(100):
                rows = await service.list_reviews()
                if rows:
                    break
                await asyncio.sleep(0.01)
            assert len(rows) == 1
            assert rows[0]["occurrence_count"] == 1
        finally:
            await service.close()
            await db.close()

    asyncio.run(scenario())


def test_flywheel_language_retries_invalid_json_without_changing_schema_or_provider():
    import json

    import httpx

    from support_agent.config import Settings
    from support_agent.flywheel import NormalizedGap
    from support_agent.knowledge_mining import KnowledgeLLM

    async def scenario():
        calls = []

        async def handler(request):
            calls.append(json.loads(request.content))
            value = (
                {}
                if len(calls) == 1
                else {
                    "question": "杯盖如何清洗？",
                    "example_answer": "待核实：请核对清洗方法。",
                }
            )
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(value)},
                        }
                    ],
                },
            )

        config = Settings(
            _env_file=None,
            openai_base_url="https://model.invalid/v1",
            openai_api_key="test",
            openai_model="test",
            database_url="sqlite+aiosqlite:///:memory:",
            structured_output_method="json_mode",
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await KnowledgeLLM(config, client).ask(
                NormalizedGap, "只输出 JSON", {"question": "杯盖怎么洗"}
            )
        assert result.question == "杯盖如何清洗？"
        assert len(calls) == 2
        assert all(c["response_format"] == {"type": "json_object"} for c in calls)

    asyncio.run(scenario())
