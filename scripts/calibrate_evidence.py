"""以已保存的 ch04 真实精排结果校准证据闸，明确记录来源与误判。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.confidence import calibrate  # noqa: E402
from support_agent.knowledge_store import digest  # noqa: E402


def main():
    """读取完整报告及标注集，拒绝使用样本标签不一致的报告进行校准。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="dev-notes/ch04-eval-final-results.json")
    parser.add_argument("--dataset", default="evals/retrieval.jsonl")
    parser.add_argument("--output", default="config/evidence-confidence.json")
    args = parser.parse_args()
    report_text = Path(args.report).read_text(encoding="utf-8")
    report = json.loads(report_text)
    dataset = Path(args.dataset).read_text(encoding="utf-8")
    labels = {r["id"]: r for r in map(json.loads, dataset.splitlines()) if r}
    rows = [r for r in report["rows"] if r["strategy"] == "hybrid_rerank"]
    if report["mode"] != "live" or {r["id"] for r in rows} != set(labels):
        raise ValueError("校准需要覆盖完整当前标注集的真实精排报告。")
    for row in rows:
        for key in ("question", "expected_refusal", "relevant"):
            if row[key] != labels[row["id"]][key]:
                raise ValueError("报告与当前标注不一致，请重新运行评估。")
    artifact = calibrate(
        rows,
        dataset_hash=digest(dataset),
        metadata={
            "report": args.report,
            "report_hash": digest(report_text),
            "embedding": report["metadata"]["embedding"],
            "reranker": report["metadata"]["reranker"],
            "corpus_hash": report["metadata"]["corpus_hash"],
            "note": "复用 ch04 真实召回；语料变更后应重新评估并校准。",
        },
    )
    Path(args.output).write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
