"""用标注样例验证标准化、条件保留与语义查重，不把 Prompt 效果当单元测试。"""

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.config import Settings  # noqa: E402
from support_agent.flywheel import DEDUP_PROMPT, NORMALIZE_PROMPT, GapMatch, NormalizedGap  # noqa: E402
from support_agent.knowledge_mining import KnowledgeLLM  # noqa: E402


async def main():
    """逐条真实调用同一模型，保存输出与标注判定，任何不一致都返回失败。"""
    config = Settings().model_copy(update={"model_thinking": "disabled"})
    cases = [
        json.loads(line)
        for line in (ROOT / "evals/flywheel.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    with httpx.Client() as sync_client:
        async with httpx.AsyncClient() as client:
            language = KnowledgeLLM(config, client, sync_client)
            for case in cases:
                normalized = await language.ask(
                    NormalizedGap,
                    NORMALIZE_PROMPT,
                    {
                        "task": "normalize",
                        "question": case["question"],
                        "retrieval_snapshot": None,
                    },
                )
                matched = None
                if case["candidates"]:
                    matched = (
                        await language.ask(
                            GapMatch,
                            DEDUP_PROMPT,
                            {
                                "task": "deduplicate",
                                "question": normalized.question,
                                "candidates": case["candidates"],
                            },
                        )
                    ).matched_id
                passed = (
                    all(t in normalized.question for t in case["required_terms"])
                    and normalized.example_answer.startswith("待核实：")
                    and matched == case["expected_match"]
                )
                results.append(
                    {
                        "id": case["id"],
                        **normalized.model_dump(),
                        "matched_id": matched,
                        "passed": passed,
                    }
                )
    output = {
        "model": config.openai_model,
        "thinking": "disabled",
        "results": results,
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
    }
    (ROOT / "dev-notes/ch09-prompt-eval.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["passed"] != output["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
