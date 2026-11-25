"""运行四策略真实评估，或回放已保存的边界样本核验指标；不会伪造模型输出。"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.config import Settings  # noqa: E402
from support_agent.database import Database  # noqa: E402
from support_agent.evaluation import (  # noqa: E402
    assess_faithfulness,
    build_report,
    markdown_report,
)
from support_agent.grounding import GroundedAnswerer  # noqa: E402
from support_agent.knowledge_mining import KnowledgeLLM  # noqa: E402
from support_agent.knowledge_store import KnowledgeStore, digest  # noqa: E402
from support_agent.retrieval import STRATEGIES, HybridRetriever, QueryUnderstanding  # noqa: E402
from support_agent.vectors import BGEEmbeddings, BGEReranker, MilvusVectors  # noqa: E402


def load_jsonl(path):
    """以 UTF-8 读取标注或已记录的运行样本。"""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_label(row):
    """用来源文件名与完整章节组成跨数据库主键稳定的标注身份。"""
    source = row["source_ref"]
    filename = source.rsplit(":", 1)[-1] if source.startswith("markdown:") else source
    return filename + "#" + row["section_path"].splitlines()[0]


async def live_evaluation(args):
    """对同一语料、归一问题和标签执行四策略，逐样本保存实际输出与评委判据。"""
    config = Settings()
    config = config.model_copy(update={"max_output_tokens": config.eval_max_output_tokens})
    database = Database(config.database_url.get_secret_value())
    vectors = MilvusVectors(config.milvus_uri, config.milvus_hybrid_collection)
    rows = []
    dataset_text = await asyncio.to_thread(Path(args.dataset).read_text, encoding="utf-8")
    cases = [json.loads(line) for line in dataset_text.splitlines() if line.strip()]
    if args.limit:
        cases = cases[: args.limit]
    try:
        await database.initialize()
        store = KnowledgeStore(database)
        corpus = await store.active()
        labels = {
            source_label(
                {"source_ref": r.source_ref, "section_path": r.section_path or r.questions}
            )
            for r in corpus
        }
        for case in cases:
            missing = set(case["relevant"]) - labels
            if missing:
                raise ValueError(f"标注来源不存在，请先 build：{case['id']} {sorted(missing)}")
        if any(r.vectorize_status != "done" for r in corpus):
            raise ValueError("知识仍有 pending，请先 sync。")
        with httpx.Client() as sync_client:
            async with httpx.AsyncClient() as client:
                language = KnowledgeLLM(config, client, sync_client)
                understanding = QueryUnderstanding(language)
                retriever = HybridRetriever(
                    store,
                    BGEEmbeddings(config.siliconflow_api_key.get_secret_value()),
                    vectors,
                    BGEReranker(config.siliconflow_api_key.get_secret_value()),
                    min_score=config.rerank_min_score,
                )
                answerer = GroundedAnswerer(language)
                raw = Path(args.output).with_suffix(".rows.jsonl")
                raw.parent.mkdir(parents=True, exist_ok=True)
                semaphore = asyncio.Semaphore(args.concurrency)

                async def run_strategy(case, prepared, strategy):
                    """对同一问题独立运行一种策略，限制外部请求并发且不共享生成结论。"""
                    async with semaphore:
                        matches = await retriever.search(
                            case["question"],
                            category=case.get("category"),
                            strategy=strategy,
                            prepared=prepared,
                            apply_threshold=False,
                        )
                        usable = matches[:10]
                        if strategy == "hybrid_rerank":
                            usable = [m for m in usable if m["score"] >= config.rerank_min_score]
                        generated = await answerer.generate(case["question"], usable)
                        refused = generated["reason"] is not None
                        assessment = None
                        if not refused:
                            assessment = await assess_faithfulness(
                                language,
                                case["question"],
                                generated["answer"],
                                usable,
                            )
                            if not assessment.claims:
                                raise ValueError("非拒答的知识回答未获得逐事实评估。")
                        return {
                            **case,
                            "strategy": strategy,
                            "prepared_query": prepared,
                            "ranked": [source_label(m) for m in matches],
                            "matches": matches,
                            "answer": generated["answer"],
                            "sources": generated["sources"],
                            "refused": refused,
                            "refusal_reason": generated["reason"],
                            "claims": [c.model_dump() for c in assessment.claims]
                            if assessment
                            else [],
                            "answers_question": assessment.answers_question
                            if assessment
                            else False,
                            "gold_terms_present": all(
                                t in generated["answer"] for t in case["expected_facts"]
                            ),
                        }

                with raw.open("w", encoding="utf-8") as log:
                    for case in cases:
                        prepared = await understanding.prepare(case["question"])
                        async with asyncio.TaskGroup() as group:
                            tasks = [
                                group.create_task(run_strategy(case, prepared, strategy))
                                for strategy in STRATEGIES
                            ]
                        for task in tasks:
                            row = task.result()
                            rows.append(row)
                            log.write(json.dumps(row, ensure_ascii=False) + "\n")
                            log.flush()
                            state = "refused" if row["refused"] else "answered"
                            print(f"{case['id']} / {row['strategy']}: {state}", flush=True)
        return build_report(
            rows,
            mode="live",
            metadata={
                "created_at": datetime.now(UTC).isoformat(),
                "model": config.openai_model,
                "max_output_tokens": config.max_output_tokens,
                "thinking": config.model_thinking,
                "embedding": "BAAI/bge-m3",
                "reranker": "BAAI/bge-reranker-v2-m3",
                "collection": config.milvus_hybrid_collection,
                "rrf_k": 60,
                "recall_per_route": 50,
                "rerank_min_score": config.rerank_min_score,
                "corpus_size": len(corpus),
                "corpus_hash": digest("\n".join(f"{r.id}:{r.content_hash}" for r in corpus)),
                "dataset_hash": digest(dataset_text),
                "pymilvus_version": version("pymilvus"),
                "judge": "与生成同一提供方的独立评估调用，需人工抽查，非客观真值",
            },
        )
    finally:
        await vectors.close()
        await database.close()


def main():
    """解析真实运行或回放参数，生成 JSON 明细和 Markdown 分桶报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "evals/retrieval.jsonl"))
    parser.add_argument("--output", default="dev-notes/ch04-eval-results.json")
    parser.add_argument("--replay", help="回放 rows JSONL；报告明确标记 replay，不当作在线效果")
    parser.add_argument("--limit", type=int, help="仅评估前N条，用于快速检查")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    report = (
        build_report(load_jsonl(args.replay), mode="replay")
        if args.replay
        else asyncio.run(live_evaluation(args))
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["strategies"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
