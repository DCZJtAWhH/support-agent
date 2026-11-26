import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx


def test_evaluation_command_reports_known_ranks_claim_faithfulness_and_type_buckets(tmp_path):
    rows = []
    for strategy in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        rows.extend(
            [
                {
                    "id": "a",
                    "query_type": "model",
                    "difficulty": "easy",
                    "strategy": strategy,
                    "relevant": ["A", "B"],
                    "ranked": ["X", "B", "A"],
                    "refused": False,
                    "expected_refusal": False,
                    "claims": [{"supported": True}, {"supported": False}],
                },
                {
                    "id": "b",
                    "query_type": "unknown",
                    "difficulty": "hard",
                    "strategy": strategy,
                    "relevant": [],
                    "ranked": ["X"],
                    "refused": True,
                    "expected_refusal": True,
                    "claims": [],
                },
            ]
        )
    replay = tmp_path / "replay.jsonl"
    replay.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_retrieval.py",
            "--replay",
            str(replay),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    for metrics in report["strategies"].values():
        assert metrics["recall@1"] == 0
        assert metrics["recall@5"] == 1
        assert metrics["mrr"] == 0.5
        assert metrics["faithfulness"] == 0.5
        assert metrics["answerable_count"] == 1
        assert metrics["refusal_accuracy"] == 1
    assert report["by_query_type"]["unknown"]["dense"]["faithfulness"] is None
    assert report["by_query_type"]["model"]["dense"]["mrr"] == 0.5
    assert output.with_suffix(".md").exists()


def test_faithfulness_preserves_ten_originals_without_duplicate_retrieval_metadata():
    from support_agent.evaluation.metrics import assess_faithfulness
    from support_agent.knowledge.mining import KnowledgeLLM
    from support_agent.platform.config import Settings

    async def scenario():
        contexts = [
            {
                "chunk_id": i + 1,
                "section_path": f"资料 / 第{i}节",
                "category": "会员适用范围",
                "answer": f"原文{i}：" + "政策正文。" * 60,
                "text": f"分类资料。问法别名{i}。" + "政策正文。" * 60,
                "question": "问题别名" * 60,
                "source_ref": "来源标识" * 30,
                "content_hash": "a" * 64,
                "source_url": "/knowledge/chunks/1",
                "score": 0.8,
            }
            for i in range(10)
        ]

        async def handler(request):
            payload = json.loads(request.content)
            evidence = json.loads(payload["messages"][-1]["content"])["contexts"]
            assert len(evidence) == 10
            assert evidence[0]["text"] == "原文0：" + "政策正文。" * 60
            assert evidence[9]["text"] == "原文9：" + "政策正文。" * 60
            assert "content_hash" not in evidence[0]
            assert evidence[0]["category"] == "会员适用范围"
            content = json.dumps(
                {
                    "claims": [
                        {"statement": "有明确条款", "supported": True, "reason": "原文支持"},
                        {"statement": "编造承诺", "supported": False, "reason": "原文没有"},
                    ],
                    "answers_question": False,
                },
                ensure_ascii=False,
            )
            return httpx.Response(
                200,
                json={
                    "id": "judge",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": content},
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
            language = KnowledgeLLM(config, client)
            result = await assess_faithfulness(language, "有哪些条款", "有明确条款并承诺", contexts)
            assert [c.supported for c in result.claims] == [True, False]

    asyncio.run(scenario())
