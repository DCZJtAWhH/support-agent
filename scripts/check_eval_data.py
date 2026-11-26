"""离线检查标注集能否由公开输出契约表达；不访问模型，不评估模型语义。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.commerce.schemas import AfterSalesRequest  # noqa: E402


def main():
    seen = set()
    path = ROOT / "evals" / "after_sales.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        if case["id"] in seen or not case["description"].strip():
            raise ValueError("标注集包含重复 ID 或空描述。")
        AfterSalesRequest.model_validate(case["expected"])
        seen.add(case["id"])
    print(f"{len(seen)} labeled examples validated offline; model semantics not evaluated.")


if __name__ == "__main__":
    main()
