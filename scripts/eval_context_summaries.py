"""离线校验摘要标注样例及人工提供的响应，不调用任何模型。"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """检查必须保留的事实、禁止编造的内容及长度；不将静态样例称作模型质量评估。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses", type=Path, default=ROOT / "evals/context-summary-responses.jsonl"
    )
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in (ROOT / "evals/context-summaries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    responses = {
        row["id"]: row["summary"]
        for row in (
            json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines()
        )
    }
    rows = []
    for case in cases:
        content = responses.get(case["id"], "")
        missing = [s for s in case["required"] if s not in content]
        forbidden = [s for s in case["forbidden"] if s in content]
        rows.append(
            {
                "id": case["id"],
                "summary": content,
                "missing": missing,
                "forbidden": forbidden,
                "passed": 0 < len(content) <= 200 and not missing and not forbidden,
            }
        )
    result = {
        "model_calls": 0,
        "kind": "静态标注样例校验，不代表模型质量评估",
        "passed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "rows": rows,
    }
    (ROOT / "dev-notes/ch07-summary-samples-validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert all(row["passed"] for row in rows)


if __name__ == "__main__":
    main()
