"""用标注集验证实际模型的意图、固定路径与 ReAct 轨迹，保留原始证据。"""

import argparse
import json
import re
import time
from pathlib import Path
from uuid import uuid4

import httpx
from eval_tools import parse_events

ROOT = Path(__file__).resolve().parents[1]


def main():
    """逐条请求正在运行的服务，核对日志与审计，不自动点击或创建工单。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--output", default="dev-notes/ch05-eval-results.json")
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in (ROOT / "evals/workflow.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = []
    with httpx.Client(base_url=args.base_url, timeout=180, trust_env=False) as client:
        for case in cases:
            rid = str(uuid4())
            response = client.post("/chat", json={"message": case["message"], "request_id": rid})
            response.raise_for_status()
            events = parse_events(response.text)
            cid = events[0][1]["conversation_id"]
            for _ in range(30):
                audit = client.get("/conversations/" + cid).json()
                if audit["messages"][0]["state"] != "running":
                    break
                time.sleep(0.1)
            metadata = next(
                (json.loads(m["content"]) for m in audit["messages"] if m["role"] == "workflow"), {}
            )
            nodes = [d["node"] for e, d in events if e == "workflow"]
            tools = [
                d["name"] for e, d in events if e == "tool_status" and d["state"] == "succeeded"
            ]
            answer = "".join(d["text"] for e, d in events if e == "token")
            passed = events[-1][0] == "done" and metadata.get("intent") == case["intent"]
            passed &= not audit["tickets"]
            passed &= set(case.get("tools", [])) <= set(tools)
            passed &= metadata.get("steps", 0) >= case.get("min_steps", 0)
            if case["route"] == "knowledge":
                passed &= "retrieve" in nodes and "confidence" in nodes
                if case.get("weak"):
                    passed &= "agent" not in nodes and "无法确认" in answer
                if case.get("evidence"):
                    passed &= "agent" in nodes and bool(metadata.get("sources"))
                    citations = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
                    valid = {s["citation"] for s in metadata.get("sources", [])}
                    passed &= bool(citations) and citations <= valid
                    passed &= "[citation" not in answer.lower()
            elif case["route"] == "business":
                passed &= "retrieve" not in nodes and "confidence" not in nodes
            else:
                passed &= "agent" not in nodes and case["route"] in nodes
            if case.get("actions"):
                passed &= metadata.get("actions") == case["actions"]
            if case.get("clarification"):
                passed &= not tools and "订单" in answer
            report.append(
                {
                    **case,
                    "passed": bool(passed),
                    "actual_intent": metadata.get("intent"),
                    "nodes": nodes,
                    "tools": tools,
                    "steps": metadata.get("steps"),
                    "tokens": metadata.get("tokens"),
                    "answer": answer,
                    "conversation_id": cid,
                    "events": events,
                }
            )
            print(
                f"{case['id']}: {'PASS' if passed else 'FAIL'} {metadata.get('intent')} {tools}",
                flush=True,
            )
            (ROOT / args.output).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    if not all(row["passed"] for row in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
