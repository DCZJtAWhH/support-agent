"""用标注样例验证已启动的客服服务；会调用实际配置的模型并写入演示记录。"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]


def parse_events(text: str) -> list:
    """读取完整 SSE 响应，用于评估工具选择和终止状态，不替代前端流式解析器。"""
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


def main():
    """逐条运行人工标注样例，保存可复核的工具结果、回答和验收判据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="dev-notes/ch03-tools-eval-results.json")
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in (ROOT / "evals/tools.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    with httpx.Client(base_url=args.base_url, timeout=180, trust_env=False) as client:
        for case in cases:
            response = client.post(
                "/chat", json={"message": case["message"], "request_id": str(uuid4())}
            )
            response.raise_for_status()
            events = parse_events(response.text)
            cid = events[0][1]["conversation_id"]
            audit = client.get("/conversations/" + cid).json()
            calls = [call for row in audit["messages"] for call in (row["tool_calls"] or [])]
            selected = [call["name"] for call in calls]
            passed = events[-1][0] == "done" and len(selected) <= 2
            if "tools" in case:
                passed = passed and sorted(selected) == sorted(case["tools"])
            passed = passed and not set(selected).intersection(case.get("forbidden_tools", []))
            tool_results = [
                json.loads(row["content"]) for row in audit["messages"] if row["role"] == "tool"
            ]
            if "faq_keyword" in case:
                faq = next(
                    (r["data"] for r in tool_results if r.get("ok") and "matches" in r["data"]),
                    None,
                )
                passed = (
                    passed
                    and faq is not None
                    and 1 <= len(faq["keyword"]) <= 32
                    and faq["keyword"] in case["message"]
                    and bool(faq["matches"]) == case["faq_match"]
                )
            answer = "".join(data["text"] for event, data in events if event == "token")
            results.append(
                {
                    **case,
                    "passed": passed,
                    "conversation_id": cid,
                    "selected_tools": selected,
                    "tool_results": tool_results,
                    "answer": answer,
                }
            )
            print(f"{case['id']}: {'PASS' if passed else 'FAIL'} {selected}", flush=True)
    (ROOT / args.output).write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
