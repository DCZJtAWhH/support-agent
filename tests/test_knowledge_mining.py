import asyncio
import json

import httpx
from test_knowledge_store import DemoEmbedding

from support_agent.knowledge.chunks import KnowledgeDraft
from support_agent.knowledge.mining import KnowledgeLLM, MiningPipeline
from support_agent.knowledge.store import KnowledgeStore
from support_agent.platform.config import Settings
from support_agent.platform.database import Database


def test_reviewed_history_is_staged_then_globally_deduplicated_with_conflicts_retained(tmp_path):
    async def scenario():
        calls = []

        async def handler(request):
            body = json.loads(request.content)
            payload = json.loads(body["messages"][-1]["content"])
            calls.append(payload["task"])
            if payload["task"] == "extract":
                result = {
                    "pairs": [
                        {
                            "source_index": 0,
                            "category": "配送",
                            "question": "邮费多少钱？",
                            "answer": "满99元包邮，不满收8元。",
                        },
                        {
                            "source_index": 1,
                            "category": "配送",
                            "question": "配送费用多少？",
                            "answer": "所有订单免运费。",
                        },
                    ]
                }
            else:
                result = {
                    "relation": "duplicate"
                    if payload["candidate"]["question"] == "邮费多少钱？"
                    else "conflict",
                    "knowledge_id": payload["neighbors"][0]["id"],
                }
            return httpx.Response(
                200,
                json={
                    "id": "extract-test",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(result)},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
            )

        config = Settings(
            _env_file=None,
            openai_base_url="https://model.invalid/v1",
            openai_api_key="test",
            openai_model="test-model",
            database_url="sqlite+aiosqlite:///:memory:",
            structured_output_method="json_mode",
        )
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'mining.db'}")
        try:
            await db.initialize()
            store = KnowledgeStore(db)
            ids = await store.replace_source(
                "faq:freight",
                [
                    KnowledgeDraft("配送", ["运费标准是什么"], "满99元免运费，否则8元。"),
                ],
            )
            records = [
                {
                    "conversation_id": str(i),
                    "reviewed": True,
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                }
                for i, (question, answer) in enumerate(
                    [
                        ("邮费多少钱？", "满99元包邮，不满收8元。"),
                        ("配送费用多少？", "所有订单免运费。"),
                    ]
                )
            ]
            records.append({"conversation_id": "unreviewed", "reviewed": False, "messages": []})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                pipeline = MiningPipeline(store, KnowledgeLLM(config, client), DemoEmbedding())
                assert await pipeline.extract_records(records, "history:demo") == 2
                assert all(r["status"] == "extracted" for r in await pipeline.staging())
                assert "邮费多少钱？" not in (await store.get(ids[0])).questions
                assert await pipeline.extract_records(records, "history:demo") == 0
                await pipeline.publish()
                assert len(await store.active()) == 1
                assert "邮费多少钱？" in (await store.get(ids[0])).questions
                statuses = {r["question"]: r for r in await pipeline.staging()}
                assert statuses["邮费多少钱？"]["status"] == "kept"
                assert statuses["配送费用多少？"]["status"] == "extracted"
                assert statuses["配送费用多少？"]["reason"] == "conflict"
                assert calls.count("extract") == 1
                await store.replace_source(
                    "faq:freight",
                    [
                        KnowledgeDraft("配送", ["运费标准是什么"], "满99元免运费，否则8元。"),
                    ],
                )
                assert "邮费多少钱？" in (await store.get(ids[0])).questions
        finally:
            await db.close()

    asyncio.run(scenario())


def test_same_question_with_paraphrased_answer_is_judged_semantically(tmp_path):
    async def scenario():
        async def handler(request):
            payload = json.loads(json.loads(request.content)["messages"][-1]["content"])
            result = (
                {
                    "pairs": [
                        {
                            "source_index": 0,
                            "category": "配送",
                            "question": "运费标准是什么",
                            "answer": "订单满99元包邮，未满99元需要8元。",
                        }
                    ]
                }
                if payload["task"] == "extract"
                else {
                    "relation": "duplicate",
                    "knowledge_id": payload["neighbors"][0]["id"],
                }
            )
            return httpx.Response(
                200,
                json={
                    "id": "paraphrase",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(result)},
                        }
                    ],
                },
            )

        config = Settings(
            _env_file=None,
            openai_base_url="https://model.invalid/v1",
            openai_api_key="test",
            openai_model="test-model",
            database_url="sqlite+aiosqlite:///:memory:",
            structured_output_method="json_mode",
        )
        db = Database(f"sqlite+aiosqlite:///{tmp_path / 'paraphrase.db'}")
        try:
            await db.initialize()
            store = KnowledgeStore(db)
            await store.replace_source(
                "faq:1",
                [
                    KnowledgeDraft("配送", ["运费标准是什么"], "满99元免运费，否则8元。"),
                ],
            )
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                pipeline = MiningPipeline(store, KnowledgeLLM(config, client), DemoEmbedding())
                await pipeline.extract_records(
                    [
                        {
                            "conversation_id": "same-question",
                            "reviewed": True,
                            "messages": [
                                {"role": "user", "content": "运费标准是什么"},
                                {
                                    "role": "assistant",
                                    "content": "订单满99元包邮，未满99元需要8元。",
                                },
                            ],
                        }
                    ],
                    "history:paraphrase",
                )
                assert (await pipeline.publish())["merged"] == 1
                assert (await pipeline.staging())[0]["status"] == "kept"
        finally:
            await db.close()

    asyncio.run(scenario())
