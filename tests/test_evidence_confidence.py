from support_agent.confidence import EvidencePolicy, calibrate


def test_calibrated_gate_rejects_irrelevant_fragments_and_keeps_answerable_evidence():
    rows = [
        {"id": "known-a", "expected_refusal": False, "matches": [{"score": 0.9}, {"score": 0.6}]},
        {"id": "known-b", "expected_refusal": False, "matches": [{"score": 0.8}]},
        {"id": "unknown-a", "expected_refusal": True, "matches": [{"score": 0.02}]},
        {
            "id": "unknown-b",
            "expected_refusal": True,
            "matches": [{"score": 0.04}, {"score": 0.03}],
        },
    ]
    artifact = calibrate(rows, dataset_hash="annotated-dataset")
    policy = EvidencePolicy(artifact)
    assert policy.assess([{"score": 0.85}, {"score": 0.65}])["accepted"] is True
    assert policy.assess([{"score": 0.01}])["accepted"] is False
    assert policy.assess([])["accepted"] is False
    assert artifact["dataset_hash"] == "annotated-dataset"
    assert all(weight > 0 for weight in artifact["weights"])
    explicit_floor = EvidencePolicy({**artifact, "relevance_floor": 0.5})
    assert explicit_floor.assess([{"score": 0.85}, {"score": 0.65}])["signals"] == {
        "top1": 0.85,
        "effective_count": 2,
        "margin": 0.2,
    }


def test_calibration_keeps_known_multi_evidence_when_ambiguity_needs_generation_assessment():
    rows = [
        {
            "id": f"known-{i}",
            "expected_refusal": False,
            "matches": [{"score": 0.99}, {"score": 0.98}] if i < 2 else [{"score": 0.6}],
        }
        for i in range(14)
    ]
    rows += [
        {
            "id": "ambiguous",
            "expected_refusal": True,
            "matches": [{"score": 0.99}, {"score": 0.98}],
        },
        *[
            {"id": f"unknown-{i}", "expected_refusal": True, "matches": [{"score": 0.001}]}
            for i in range(5)
        ],
    ]
    artifact = calibrate(rows, dataset_hash="labeled-overlap")
    assert artifact["false_reject_ids"] == []
    assert artifact["false_accept_ids"] == ["ambiguous"]


def test_workflow_gate_keeps_rejected_fragments_for_review_before_agent():
    import asyncio
    from uuid import uuid4

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from test_workflow import IntentModel, settings

    from support_agent.database import Database
    from support_agent.repositories import ChatRepository
    from support_agent.workflow import SupportWorkflow

    class Search:
        async def search(self, question, **kwargs):
            return [{"chunk_id": 9, "score": 0.01, "text": "相似但不回答问题的原文"}]

    async def scenario():
        db = Database("sqlite+aiosqlite:///:memory:")
        await db.initialize()
        repo = ChatRepository(db, Search())
        cid, rid = str(uuid4()), str(uuid4())
        try:
            await repo.begin_request(cid, rid, "完全未知的杯子功能")
            async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
                workflow = SupportWorkflow(IntentModel("商品咨询"), repo, settings(), saver)
                events = [event async for event in workflow.stream(cid, rid, "完全未知的杯子功能")]
                assert "agent" not in [e.get("node") for e in events]
            pooled = await repo.low_confidence_questions()
            assert pooled[0]["retrieval_snapshot"][0]["text"] == "相似但不回答问题的原文"
            assert pooled[0]["entry_point"] == "workflow_gate"
        finally:
            await db.close()

    asyncio.run(scenario())


def test_generation_self_assessment_refuses_and_pools_high_scoring_partial_evidence(harness):
    import asyncio

    from conftest import sse_events, stream_response

    class Search:
        async def search(self, question, **kwargs):
            return [
                {"chunk_id": 9, "score": 0.99, "text": "杯盖可以手洗。", "answer": "杯盖可以手洗。"}
            ]

    async def handler(request, payload):
        prompt = payload["messages"][0]["content"]
        if prompt.startswith("将顾客"):
            return stream_response(('{"intent":"商品咨询","confidence":0.99}',))
        if "生成证据自评" in prompt:
            return stream_response(('{"sufficient":false,"reason":"未说明能否用洗碗机"}',))
        return stream_response(("可以用洗碗机。[1]",))

    async def scenario():
        async with harness(handler) as (client, _, app):
            app.state.repository.retriever = Search()
            response = await client.post("/chat", json={"message": "杯盖可以用洗碗机吗"})
            events = sse_events(response)
            released = "".join(d["text"] for event, d in events if event == "token")
            assert "无法确认" in released
            assert "可以用洗碗机。[1]" not in released
            row = (await app.state.repository.low_confidence_questions())[0]
            assert row["entry_point"] == "generation"
            assert row["retrieval_snapshot"][0]["score"] == 0.99

    asyncio.run(scenario())
