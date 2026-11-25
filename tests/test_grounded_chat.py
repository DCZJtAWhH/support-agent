import asyncio
import json
from uuid import uuid4

import httpx
from conftest import sse_events, stream_response


def test_grounded_generation_keeps_ten_full_sources_in_budget_and_preserves_snapshots():
    from support_agent.config import Settings
    from support_agent.grounding import GroundedAnswerer
    from support_agent.knowledge_mining import KnowledgeLLM

    async def scenario():
        matches = [
            {
                "chunk_id": i + 1,
                "section_path": f"会员 / 第{i}节",
                "category": "会员",
                "answer": f"完整原文{i}：" + "政策正文。" * 60,
                "text": f"别名{i}" + "政策正文。" * 60,
                "question": "问法" * 100,
                "content_hash": "a" * 64,
                "source_ref": "来源" * 100,
                "source_url": f"/knowledge/chunks/{i + 1}",
            }
            for i in range(10)
        ]

        async def handler(request):
            payload = json.loads(request.content)
            sources = json.loads(payload["messages"][-1]["content"])["sources"]
            assert len(sources) == 10
            assert [s["citation"] for s in sources] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
            assert sources[0]["text"] == "完整原文0：" + "政策正文。" * 60
            assert sources[-1]["text"] == "完整原文1：" + "政策正文。" * 60
            assert sources[0]["category"] == "会员"
            assert "content_hash" not in sources[0]
            answer = json.dumps(
                {"sufficient": True, "reason": "条款完整", "answer": "依据会员条款。[1]"}
            )
            return httpx.Response(
                200,
                json={
                    "id": "answer",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": answer},
                        }
                    ],
                },
            )

        config = Settings(
            _env_file=None,
            openai_base_url="https://model.invalid/v1",
            openai_model="test-model",
            openai_api_key="test",
            database_url="sqlite+aiosqlite:///:memory:",
            structured_output_method="json_mode",
            context_window_tokens=8192,
            max_output_tokens=1024,
            token_safety_margin=256,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await GroundedAnswerer(KnowledgeLLM(config, client)).generate(
                "会员有哪些条款", matches
            )
            assert result["answer"] == "依据会员条款。[1]"
            assert result["sources"][0]["content_hash"] == "a" * 64
            assert result["sources"][0]["source_url"] == "/knowledge/chunks/1"

    asyncio.run(scenario())


def test_strong_evidence_streams_through_agent_with_projected_prompt_and_snapshot(harness):
    class SearchAPI:
        async def search(self, question, **kwargs):
            return [
                {
                    "chunk_id": i + 1,
                    "score": 0.99,
                    "section_path": f"会员 / 第{i}节",
                    "category": "会员",
                    "answer": "政策正文。" * 60,
                    "text": "政策正文。" * 60,
                    "question": "问法" * 100,
                    "content_hash": "a" * 64,
                    "source_ref": "来源" * 100,
                    "source_url": f"/knowledge/chunks/{i + 1}",
                }
                for i in range(10)
            ]

    async def handler(request, payload):
        if payload["messages"][0]["content"].startswith("将顾客"):
            return stream_response(('{"intent":"商品咨询","confidence":0.99}',))
        if "生成证据自评" in payload["messages"][0]["content"]:
            return stream_response(('{"sufficient":true,"reason":"测试证据支持回答"}',))
        assert "tools" in payload and payload["stream"]
        evidence = json.loads(payload["messages"][-1]["content"].split("\n", 1)[1])["evidence"]
        assert "政策正文" not in payload["messages"][0]["content"]
        assert len(evidence) == 10 and "content_hash" not in evidence[0]
        return stream_response(("会员条款。", "[1]"))

    async def scenario():
        async with harness(handler) as (client, _, app):
            app.state.repository.retriever = SearchAPI()
            response = await client.post("/chat", json={"message": "会员条款是什么"})
            events = sse_events(response)
            assert events[-1][0] == "done", events
            sources = next(d["sources"] for e, d in events if e == "sources")
            assert sources[0]["content_hash"] == "a" * 64
            assert [d["text"] for e, d in events if e == "token"] == ["会员条款。", "[1]"]

    asyncio.run(scenario())


def test_unknown_knowledge_refuses_before_agent_and_pool_is_idempotent_on_retry(harness):
    class SearchAPI:
        async def search(self, question, **kwargs):
            return []

    async def handler(request, payload):
        assert "tools" not in payload
        return stream_response(('{"intent":"商品咨询","confidence":0.99}',))

    async def scenario():
        async with harness(handler) as (client, calls, app):
            app.state.repository.retriever = SearchAPI()
            body = {"message": "火星补贴是多少", "request_id": str(uuid4())}
            response = await client.post("/chat", json=body)
            events = sse_events(response)
            assert events[-1][0] == "done" and "无法确认" in response.text
            rows = await app.state.repository.low_confidence_questions()
            assert len(rows) == 1 and rows[0]["entry_point"] == "workflow_gate"
            assert rows[0]["question"] == body["message"]
            assert sse_events(await client.post("/chat", json=body))[-1][0] == "done"
            assert len(await app.state.repository.low_confidence_questions()) == 1
            assert len(calls) == 2
            audit = (await client.get("/conversations/" + events[0][1]["conversation_id"])).json()
            assert audit["tickets"] == []

    asyncio.run(scenario())


def test_invalid_intent_cannot_release_a_model_answer_or_create_ticket(harness):
    async def handler(request, payload):
        return stream_response(("ZX-999额定功率是999瓦。",))

    async def scenario():
        async with harness(handler) as (client, _, _):
            response = await client.post("/chat", json={"message": "ZX-999额定功率是多少"})
            assert sse_events(response)[-1][0] == "error"
            assert "999瓦" not in response.text

    asyncio.run(scenario())
