import asyncio


def test_completed_evaluations_persist_two_distinct_runs_and_api_shows_trend(harness):
    async def scenario():
        from support_agent.evaluation.runs import EvaluationRuns

        async with harness() as (client, _, app):
            runs = EvaluationRuns(app.state.repository.database)
            first = {
                "mode": "live",
                "rows": [{"id": "a"}, {"id": "b"}],
                "metadata": {"dataset_hash": "same", "corpus_hash": "same"},
                "strategies": {"hybrid_rerank": {"recall@5": 1.0, "mrr": 0.9, "faithfulness": 1.0}},
            }
            second = {
                **first,
                "strategies": {
                    "hybrid_rerank": {
                        "recall@5": 0.5,
                        "mrr": 0.4,
                        "faithfulness": 0.8,
                    }
                },
            }
            await runs.record("run-a", "manual", first)
            await runs.record("run-a", "manual", first)
            await runs.record("run-b", "scheduled", second)
            response = await client.get("/api/evaluations/runs")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert [r["dataset_count"] for r in data] == [2, 2]
            assert [r["trigger_type"] for r in data] == ["manual", "scheduled"]
            assert [r["metrics"]["hybrid_rerank"]["mrr"] for r in data] == [0.9, 0.4]

    asyncio.run(scenario())


def test_periodic_evaluation_runs_existing_pipeline_and_persists_trigger(harness, tmp_path):
    from test_evaluation_dashboard import make_runner

    from support_agent.evaluation.dashboard import EvaluationDashboard

    async def scenario():
        dataset = tmp_path / "cases.jsonl"
        dataset.write_text('{"id":"unknown","relevant":[]}\n', encoding="utf-8")
        async with harness() as (client, _, app):
            app.state.evaluations = EvaluationDashboard(
                app.state.knowledge_store.corpus,
                runs=app.state.evaluation_runs,
                dataset_path=dataset,
                runs_directory=tmp_path / "runs",
                script_path=make_runner(tmp_path),
            )
            dashboard = app.state.evaluations
            dashboard.start_scheduler(0.02)
            for _ in range(200):
                rows = (await client.get("/api/evaluations/runs")).json()
                if len(rows) >= 2:
                    break
                await asyncio.sleep(0.02)
            assert len(rows) >= 2
            assert all(r["trigger_type"] == "scheduled" for r in rows)

    asyncio.run(scenario())
