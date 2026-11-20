import asyncio
import json

from support_agent.evaluation import build_report


def saved_report():
    rows = []
    for strategy in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        rows.extend(
            [
                {
                    "id": "model",
                    "question": "容量多少",
                    "query_type": "model",
                    "difficulty": "easy",
                    "strategy": strategy,
                    "relevant": ["A"],
                    "ranked": ["X", "A"],
                    "refused": False,
                    "expected_refusal": False,
                    "claims": [{"supported": True}, {"supported": False}],
                    "expected_facts": ["500"],
                    "matches": [],
                    "answer": "容量500毫升。",
                    "sources": [],
                },
                {
                    "id": "unknown",
                    "question": "不存在的知识",
                    "query_type": "unknown",
                    "difficulty": "hard",
                    "strategy": strategy,
                    "relevant": [],
                    "ranked": [],
                    "refused": True,
                    "expected_refusal": True,
                    "claims": [],
                    "expected_facts": [],
                    "matches": [],
                    "answer": "无法确认。",
                    "sources": [],
                    "refusal_reason": "缺少证据",
                },
            ]
        )
    return build_report(
        rows,
        mode="live",
        metadata={
            "created_at": "2026-11-20T07:46:03+00:00",
            "corpus_size": 14,
            "corpus_hash": "old-corpus",
            "dataset_hash": "old-dataset",
        },
    )


def test_report_api_uses_saved_numbers_combines_filters_and_identifies_old_snapshot(
    harness, tmp_path
):
    from support_agent.evaluation_dashboard import EvaluationDashboard

    async def scenario():
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps(saved_report(), ensure_ascii=False), encoding="utf-8")
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text('{"id":"unknown","relevant":[]}\n', encoding="utf-8")
        async with harness() as (client, _, app):
            app.state.evaluations = EvaluationDashboard(
                app.state.knowledge_store,
                baseline_path=baseline,
                dataset_path=dataset,
                runs_directory=tmp_path / "runs",
            )
            response = await client.get("/api/evaluations?query_type=model&difficulty=easy")
            assert response.status_code == 200, response.text
            result = response.json()
            assert len(result["rows"]) == 4
            assert result["strategies"]["dense"]["mrr"] == 0.5
            assert result["strategies"]["dense"]["faithfulness"] == 0.5
            assert result["report"]["case_count"] == 2
            assert result["corpus"]["size"] == 0
            assert result["stale"] is True and result["dataset_changed"] is True
            assert result["filters"]["query_types"] == ["model", "unknown"]
            unknown = (await client.get("/api/evaluations?query_type=unknown")).json()
            assert unknown["strategies"]["dense"]["faithfulness"] is None
            assert (await client.get("/evaluations")).status_code == 200

    asyncio.run(scenario())


def make_runner(tmp_path, *, fail=False, failure_message="private-provider-error"):
    script = tmp_path / "runner.py"
    script.write_text(
        "import argparse, hashlib, json, sys, time\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--output');"
        "p.add_argument('--concurrency');a=p.parse_args()\n"
        + (f"print({failure_message!r},flush=True);sys.exit(2)\n" if fail else "")
        + "rows=[]\n"
        "for s in ['dense','bm25','hybrid','hybrid_rerank']:\n"
        " time.sleep(0.15);print('unknown / '+s+': refused',flush=True)\n"
        " rows.append(dict(id='unknown',question='无知识',query_type='unknown',difficulty='hard',"
        "strategy=s,relevant=[],ranked=[],refused=True,expected_refusal=True,claims=[],"
        "expected_facts=[],matches=[],answer='无法确认。',sources=[]))\n"
        "r=dict(mode='live',rows=rows,metadata=dict(created_at='2026-11-20T00:00:00Z',corpus_size=0,"
        "corpus_hash='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',"
        "dataset_hash=hashlib.sha256(Path(a.dataset).read_text(encoding='utf-8').encode()).hexdigest()))\n"
        "Path(a.output).write_text(json.dumps(r),encoding='utf-8')\n",
        encoding="utf-8",
    )
    return script


async def wait_for_job(client):
    states = []
    for _ in range(200):
        job = (await client.get("/api/evaluations/status")).json()["job"]
        states.append(job)
        if job["status"] != "running":
            return job, states
        await asyncio.sleep(0.03)
    raise AssertionError("评估未结束")


def test_rerun_exposes_progress_blocks_duplicates_and_publishes_only_completed_report(
    harness, tmp_path
):
    from support_agent.evaluation_dashboard import EvaluationDashboard

    async def scenario():
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text('{"id":"unknown","relevant":[]}\n', encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps(saved_report()), encoding="utf-8")
        original = baseline.read_bytes()
        async with harness() as (client, _, app):
            app.state.evaluations = EvaluationDashboard(
                app.state.knowledge_store,
                baseline_path=baseline,
                dataset_path=dataset,
                runs_directory=tmp_path / "runs",
                script_path=make_runner(tmp_path),
            )
            started = await client.post("/api/evaluations/run")
            assert started.status_code == 202, started.text
            assert started.json()["job"]["total"] == 4
            assert (await client.post("/api/evaluations/run")).status_code == 409
            final, states = await wait_for_job(client)
            assert final["status"] == "completed", final
            assert final["completed"] == 4
            assert any(0 < state["completed"] < 4 for state in states)
            current = (await client.get("/api/evaluations")).json()
            assert current["report"]["id"] == started.json()["job"]["id"]
            assert current["report"]["row_count"] == 4
            assert current["stale"] is False and current["dataset_changed"] is False
            assert baseline.read_bytes() == original

    asyncio.run(scenario())


def test_failed_run_keeps_previous_report_and_never_exposes_provider_error(harness, tmp_path):
    from support_agent.evaluation_dashboard import EvaluationDashboard

    async def scenario():
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text('{"id":"unknown","relevant":[]}\n', encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps(saved_report()), encoding="utf-8")
        async with harness() as (client, _, app):
            app.state.evaluations = EvaluationDashboard(
                app.state.knowledge_store,
                baseline_path=baseline,
                dataset_path=dataset,
                runs_directory=tmp_path / "runs",
                script_path=make_runner(tmp_path, fail=True),
            )
            assert (await client.post("/api/evaluations/run")).status_code == 202
            final, _ = await wait_for_job(client)
            assert final["status"] == "failed"
            assert "private-provider-error" not in json.dumps(final)
            assert (await client.get("/api/evaluations")).json()["report"]["id"] == "baseline"

    asyncio.run(scenario())


def test_knowledge_change_during_run_does_not_publish_mixed_snapshot(harness, tmp_path):
    from support_agent.evaluation_dashboard import EvaluationDashboard
    from support_agent.knowledge_chunks import KnowledgeDraft

    async def scenario():
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text('{"id":"unknown","relevant":[]}\n', encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps(saved_report()), encoding="utf-8")
        async with harness() as (client, _, app):
            app.state.evaluations = EvaluationDashboard(
                app.state.knowledge_store,
                baseline_path=baseline,
                dataset_path=dataset,
                runs_directory=tmp_path / "runs",
                script_path=make_runner(tmp_path),
            )
            assert (await client.post("/api/evaluations/run")).status_code == 202
            await app.state.knowledge_store.replace_source(
                "upload:changed.md",
                [
                    KnowledgeDraft("水杯", ["容量"], "容量500毫升。"),
                ],
            )
            final, _ = await wait_for_job(client)
            assert final["status"] == "failed"
            assert "知识库" in final["message"] and "变化" in final["message"]
            assert (await client.get("/api/evaluations")).json()["report"]["id"] == "baseline"

    asyncio.run(scenario())


def test_restart_marks_orphaned_running_job_interrupted(harness, tmp_path):
    from support_agent.evaluation_dashboard import EvaluationDashboard

    async def scenario():
        runs = tmp_path / "runs"
        runs.mkdir()
        (runs / "job.json").write_text(
            json.dumps(
                {
                    "id": "b3d322cc-434c-40c9-a633-d4bd758b3288",
                    "status": "running",
                    "completed": 2,
                    "total": 4,
                }
            ),
            encoding="utf-8",
        )
        async with harness() as (client, _, app):
            app.state.evaluations = EvaluationDashboard(
                app.state.knowledge_store, runs_directory=runs
            )
            job = (await client.get("/api/evaluations/status")).json()["job"]
            assert job["status"] == "interrupted"
            assert "重启" in job["message"]
            assert job["completed"] == 2

    asyncio.run(scenario())


def test_shutdown_interrupts_active_evaluation_and_preserves_previous_report(harness, tmp_path):
    from support_agent.evaluation_dashboard import EvaluationDashboard

    async def scenario():
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text('{"id":"unknown","relevant":[]}\n', encoding="utf-8")
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps(saved_report()), encoding="utf-8")
        async with harness() as (client, _, app):
            app.state.evaluations = EvaluationDashboard(
                app.state.knowledge_store,
                baseline_path=baseline,
                dataset_path=dataset,
                runs_directory=tmp_path / "runs",
                script_path=make_runner(tmp_path),
            )
            assert (await client.post("/api/evaluations/run")).status_code == 202
            for _ in range(100):
                progress = (await client.get("/api/evaluations/status")).json()["job"]
                if progress["completed"] > 0:
                    break
                await asyncio.sleep(0.01)
            await app.state.evaluations.close()
            stopped = (await client.get("/api/evaluations/status")).json()["job"]
            assert stopped["status"] == "interrupted"
            assert stopped["completed"] < stopped["total"]
            assert (await client.get("/api/evaluations")).json()["report"]["id"] == "baseline"

    asyncio.run(scenario())


def test_context_budget_failure_has_actionable_reason_without_leaking_document(harness, tmp_path):
    from support_agent.evaluation_dashboard import EvaluationDashboard

    async def scenario():
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text('{"id":"unknown","relevant":[]}\n', encoding="utf-8")
        async with harness() as (client, _, app):
            app.state.evaluations = EvaluationDashboard(
                app.state.knowledge_store,
                dataset_path=dataset,
                runs_directory=tmp_path / "runs",
                script_path=make_runner(
                    tmp_path,
                    fail=True,
                    failure_message="ContextBudgetExceeded: private-document-data",
                ),
            )
            await client.post("/api/evaluations/run")
            final, _ = await wait_for_job(client)
            assert final["status"] == "failed"
            assert "输入预算" in final["message"]
            assert "private-document-data" not in json.dumps(final)

    asyncio.run(scenario())
