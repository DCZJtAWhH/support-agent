import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from conftest import sse_events, stream_response, tool_response


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


def test_chat_budgets_actual_grounded_prompt_instead_of_archived_retrieval_metadata(harness):
    class SearchAPI:
        async def search(self, question):
            return [
                {
                    "chunk_id": i + 1,
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
        if "tools" in payload:
            return tool_response(
                [{"id": "faq", "name": "query_faq", "args": {"keyword": "会员条款"}}]
            )
        evidence = json.loads(payload["messages"][-1]["content"])
        assert len(evidence["sources"]) == 10
        assert evidence["verified_tools"] == []
        return httpx.Response(
            200,
            json={
                "id": "answer",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "sufficient": True,
                                    "reason": "条款完整",
                                    "answer": "会员条款。[1]",
                                }
                            ),
                        },
                    }
                ],
            },
        )

    async def scenario():
        async with harness(
            handler,
            context_window_tokens=8192,
            max_output_tokens=1024,
            token_safety_margin=256,
            structured_output_method="json_mode",
        ) as (client, _, app):
            app.state.repository.retriever = SearchAPI()
            response = await client.post("/chat", json={"message": "会员条款是什么"})
            events = sse_events(response)
            assert events[-1][0] == "done", events
            sources = next(data["sources"] for name, data in events if name == "sources")
            assert sources[0]["content_hash"] == "a" * 64

    asyncio.run(scenario())


def test_unknown_knowledge_is_refused_and_enters_pool_once_on_chat_retry(harness):
    class SearchAPI:
        async def search(self, question):
            return []

    async def handler(request, payload):
        return tool_response([{"id": "faq", "name": "query_faq", "args": {"keyword": "火星补贴"}}])

    async def scenario():
        async with harness(handler) as (client, calls, app):
            app.state.repository.retriever = SearchAPI()
            rid = str(uuid4())
            body = {"message": "火星补贴是多少", "request_id": rid}
            response = await client.post("/chat", json=body)
            events = sse_events(response)
            answer = "".join(d["text"] for e, d in events if e == "token")
            assert "无法" in answer and "知识" in answer
            rows = await app.state.repository.low_confidence_questions()
            assert len(rows) == 1
            assert rows[0]["question"] == body["message"]
            assert rows[0]["entry_point"] == "retrieval"
            assert rows[0]["reason"] and rows[0]["created_at"]
            assert rows[0]["conversation_id"] == events[0][1]["conversation_id"]
            assert sse_events(await client.post("/chat", json=body))[-1][0] == "done"
            assert len(await app.state.repository.low_confidence_questions()) == 1
            assert len(calls) == 1

    asyncio.run(scenario())


def test_model_cannot_bypass_evidence_gate_by_answering_without_tools(harness):
    class SearchAPI:
        async def search(self, question):
            return []

    async def handler(request, payload):
        return stream_response(("ZX-999额定功率是999瓦。",))

    async def scenario():
        async with harness(handler) as (client, _, app):
            app.state.repository.retriever = SearchAPI()
            response = await client.post("/chat", json={"message": "ZX-999额定功率是多少"})
            assert "999瓦" not in response.text
            assert "无法确认" in response.text
            assert len(await app.state.repository.low_confidence_questions()) == 1

    asyncio.run(scenario())


def test_refusal_still_reports_successfully_created_ticket(harness):
    class SearchAPI:
        async def search(self, question):
            return []

    async def handler(request, payload):
        return tool_response(
            [
                {"id": "faq", "name": "query_faq", "args": {"keyword": "火星补贴"}},
                {
                    "id": "ticket",
                    "name": "create_ticket",
                    "args": {
                        "description": "请人工核实火星补贴",
                        "ticket_type": "其他",
                    },
                },
            ]
        )

    async def scenario():
        async with harness(handler) as (client, _, app):
            app.state.repository.retriever = SearchAPI()
            response = await client.post("/chat", json={"message": "火星补贴是多少，请人工处理"})
            events = sse_events(response)
            audit = await app.state.repository.read_conversation(events[0][1]["conversation_id"])
            assert "无法确认" in response.text
            assert audit["tickets"][0]["ticket_no"] in response.text

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "sufficient,answer,reason",
    [
        (True, "BX-500容量为500毫升。[1]", "证据充分"),
        (False, "", "原文只有容量，没有保温时长"),
        (True, "BX-500保温24小时。[99]", "伪造引用"),
    ],
)
def test_generation_self_assesses_and_only_releases_valid_citations(
    harness, sufficient, answer, reason
):
    from support_agent.knowledge_chunks import KnowledgeDraft
    from support_agent.knowledge_store import KnowledgeStore

    async def handler(request, payload):
        if "tools" in payload:
            return tool_response(
                [{"id": "faq", "name": "query_faq", "args": {"keyword": "BX-500"}}]
            )
        assert not payload.get("stream")
        content = json.dumps({"sufficient": sufficient, "reason": reason, "answer": answer})
        return httpx.Response(
            200,
            json={
                "id": "judgment",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    async def scenario():
        async with harness(handler) as (client, _, app):
            store = KnowledgeStore(app.state.repository.database)
            ids = await store.replace_source(
                "docs:杯子.md",
                [
                    KnowledgeDraft(
                        "水杯", ["BX-500容量"], "BX-500容量为500毫升。", "水杯 > BX-500"
                    ),
                ],
            )
            row = await store.get(ids[0])
            await store.mark_done(row)

            class SearchAPI:
                async def search(self, question):
                    return [
                        {
                            "chunk_id": row.id,
                            "content_hash": row.content_hash,
                            "score": 0.99,
                            "question": row.questions,
                            "answer": row.answer,
                            "text": row.answer,
                            "section_path": row.section_path,
                            "source_ref": row.source_ref,
                            "source_url": f"/knowledge/chunks/{row.id}",
                        }
                    ]

            app.state.repository.retriever = SearchAPI()
            response = await client.post("/chat", json={"message": "BX-500容量是多少"})
            events = sse_events(response)
            text = "".join(d["text"] for e, d in events if e == "token")
            assert events[-1][0] == "done", events
            if sufficient and "[1]" in answer:
                assert text == answer
                sources = next(d["sources"] for e, d in events if e == "sources")
                assert sources[0]["section_path"] == "水杯 > BX-500"
                assert sources[0]["text"] == row.answer
                original = await client.get(sources[0]["source_url"])
                assert original.json()["answer"] == row.answer
                document = await client.get(sources[0]["source_url"] + "/original?version=old")
                assert document.status_code == 200
                assert "水杯 &gt; BX-500" in document.text
                assert f'id="chunk-{row.id}"' in document.text
                assert "版本已更新" in document.text
                assert await app.state.repository.low_confidence_questions() == []
            else:
                assert "无法确认" in text
                assert len(await app.state.repository.low_confidence_questions()) == 1

    asyncio.run(scenario())
