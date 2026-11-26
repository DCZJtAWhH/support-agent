"""用标注样例验证真实聊天上游的问题理解、八类意图和核心场景 JSON 扩写。"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, HumanMessage

from support_agent.dialogue.routing import RouterLanguage  # noqa: E402
from support_agent.knowledge.mining import KnowledgeLLM  # noqa: E402
from support_agent.platform.config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


async def main():
    """并发评估独立样例，逐条保留实际输出，任一样例失败以非零码退出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/router.jsonl")
    parser.add_argument("--output", default="dev-notes/ch06-router-eval-final.json")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in (ROOT / args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = Settings()
    semaphore = asyncio.Semaphore(args.concurrency)
    rows = []
    output = ROOT / args.output
    async with httpx.AsyncClient() as client:
        language = RouterLanguage(
            KnowledgeLLM(config, client).model.bind(model=config.router_model),
            config,
        )

        async def evaluate(case):
            """按独立标签核对主体、原样透传、意图及扩写数量，不用模型自评代替标注。"""
            async with semaphore:
                started = time.monotonic()
                row = {"id": case["id"], "expected_intent": case["intent"], "passed": False}
                try:
                    history = []
                    for question, answer in case["history"]:
                        history.extend([HumanMessage(question), AIMessage(answer)])
                    resolved = await language.resolve(case["message"], history)
                    decision = await language.classify(resolved)
                    checks = {
                        "intent": decision.intent == case["intent"],
                        "exact": "exact_question" not in case or resolved == case["exact_question"],
                        "subjects": all(value in resolved for value in case.get("contains", [])),
                        "no_invention": all(
                            value not in resolved for value in case.get("forbidden", [])
                        ),
                    }
                    queries = await language.expand(resolved) if case.get("expand") else []
                    if case.get("expand"):
                        checks["expansion"] = 2 <= len(queries) <= 4
                    row.update(
                        resolved_question=resolved,
                        **decision.model_dump(),
                        queries=queries,
                        checks=checks,
                        passed=all(checks.values()),
                    )
                except Exception as exc:
                    row["error"] = type(exc).__name__ + ": " + str(exc)
                row["seconds"] = round(time.monotonic() - started, 2)
                rows.append(row)
                report = {
                    "model": config.router_model,
                    "passed": sum(r["passed"] for r in rows),
                    "total": len(cases),
                    "cases": sorted(rows, key=lambda r: r["id"]),
                }
                output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                print(f"{case['id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)

        await asyncio.gather(*(evaluate(case) for case in cases))
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
