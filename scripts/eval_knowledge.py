"""使用真实 BGE-M3、MySQL、Milvus 与标注集验证第三章 demo，会调用配置的模型。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.config import Settings  # noqa: E402
from support_agent.database import Database  # noqa: E402
from support_agent.knowledge_mining import MiningPipeline  # noqa: E402
from support_agent.knowledge_store import (  # noqa: E402
    KnowledgeStore,
    SimulatedInterruption,
    synchronize,
)
from support_agent.vectors import BGEEmbeddings, KnowledgeRetriever, MilvusVectors  # noqa: E402


async def evaluate_mining(store, embeddings):
    """将已落暂存的真实抽取结果与人工标注比较，保留实际输出用于语义复核。"""
    staging = await MiningPipeline(store, None, embeddings).staging()
    cases = []
    for line in (ROOT / "evals/knowledge-mining.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        rows = [row for row in staging if row["question"] == case["question"]]
        expected = case["expected"]
        if expected == "exclude":
            passed = not rows
        elif expected == "conflict":
            passed = any(
                row["status"] == "extracted" and row["reason"] == "conflict" for row in rows
            )
        else:
            reason = "distinct" if expected == "new" else "duplicate"
            passed = any(row["status"] == "kept" and row["reason"] == reason for row in rows)
        cases.append({**case, "passed": passed})
    return {"cases": cases, "staging": staging, "passed": all(case["passed"] for case in cases)}


async def evaluate(args):
    """保存检索与恢复的实际证据，答案语义仍按标注 criteria 人工复核。"""
    config = Settings()
    database = Database(config.database_url.get_secret_value())
    store = KnowledgeStore(database)
    embeddings = BGEEmbeddings(config.siliconflow_api_key.get_secret_value())
    vectors = MilvusVectors(config.milvus_uri, config.milvus_collection)
    report = {}
    try:
        await database.initialize()
        retriever = KnowledgeRetriever(
            store,
            embeddings,
            vectors,
            top_k=config.knowledge_top_k,
            min_score=config.knowledge_min_score,
        )
        results = []
        for line in (ROOT / "evals/knowledge.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            matches = await retriever.search(case["question"])
            hits = await vectors.search((await embeddings.embed([case["question"]]))[0], 3)
            passed = (
                (not matches)
                if case.get("expected_empty")
                else any(
                    all(value in match["answer"] for value in case["expected_answer_contains"])
                    for match in matches
                )
            )
            results.append(
                {
                    **case,
                    "passed": passed,
                    "matches": matches,
                    "scores": [{"id": key, "score": score} for key, score, _ in hits],
                }
            )
        report["retrieval"] = results
        if args.mining:
            report["mining"] = await evaluate_mining(store, embeddings)
        if args.recovery:
            await store.requeue()
            before = await store.stats()
            try:
                await synchronize(store, embeddings, vectors, interrupt_after_upsert=True)
            except SimulatedInterruption:
                interrupted = await store.stats()
            else:
                raise ValueError("未触发中断，检查是否已建库。")
            resumed = await synchronize(store, embeddings, vectors)
            after = await store.stats()
            replayed = await synchronize(store, embeddings, vectors)
            report["recovery"] = {
                "before": before,
                "interrupted": interrupted,
                "resumed": resumed,
                "after": after,
                "replayed": replayed,
                "passed": before["pending"] > 0
                and interrupted["pending"] == before["pending"]
                and after["pending"] == 0
                and replayed == 0,
            }
        if args.chat_url:
            from eval_tools import parse_events

            async with httpx.AsyncClient(
                base_url=args.chat_url, timeout=180, trust_env=False
            ) as client:
                response = await client.post("/chat", json={"message": "邮费是多少"})
                response.raise_for_status()
                events = parse_events(response.text)
                answer = "".join(data.get("text", "") for name, data in events if name == "token")
                report["chat"] = {
                    "answer": answer,
                    "events": events,
                    "passed": events[-1][0] == "done" and "99" in answer and "8" in answer,
                }
        report["passed"] = all(r["passed"] for r in results) and all(
            value.get("passed", True) for key, value in report.items() if isinstance(value, dict)
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            output.write_text,
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report["passed"]
    finally:
        await vectors.close()
        await database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--chat-url")
    parser.add_argument("--mining", action="store_true", help="检查已执行 mine 的标注结果")
    parser.add_argument("--output", default="dev-notes/ch03-eval-results.json")
    sys.exit(0 if asyncio.run(evaluate(parser.parse_args())) else 1)
