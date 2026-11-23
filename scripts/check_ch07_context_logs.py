"""核对离线会话日志与完整 checkpoint，不调用模型。"""

import asyncio
import json
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

ROOT = Path(__file__).resolve().parents[1]


async def main():
    """只读现有报告、日志和检查点，验证固定前缀、预算、原文及真实摘要级联。"""
    lines = (ROOT / "log/app.log").read_text(encoding="utf-8").splitlines()
    results = {}
    for mode, window, output in (("demo", 18000, 2000), ("default", 128000, 1024)):
        report = json.loads(
            (ROOT / f"dev-notes/ch07-offline-{mode}-acceptance.json").read_text(encoding="utf-8")
        )
        cid = report["conversation_id"]
        request_ids = {t["request_id"] for t in report["turns"]}
        model, history, tier = [], [], []
        for line in lines:
            if cid not in line:
                continue
            if "model_ctx " in line:
                entry = json.loads(line.split("model_ctx ", 1)[1])
                if entry["request_id"] in request_ids:
                    model.append(entry)
            if "history_ctx " in line:
                entry = json.loads(line.split("history_ctx ", 1)[1])
                if entry["request_id"] in request_ids:
                    history.append(entry)
            if "层1 降级" in line or "summary " in line:
                tier.append(line)
        assert model and history
        assert max(m["tokens"] for m in model) + output + 250 <= window
        prefixes = {
            json.dumps(
                {"system": m["messages"][0], "tools": m["tools"]},
                ensure_ascii=False,
                sort_keys=True,
            )
            for m in model
        }
        assert len(prefixes) == 1
        questions = {t["question"] for t in report["turns"]}
        for context in model:
            assert context["messages"][1 + context["window_count"]]["content"] in questions
            assert sum(m["role"] == "system" for m in context["messages"]) == 1
        async with AsyncSqliteSaver.from_conn_string(report["checkpoint_path"]) as saver:
            saved = await saver.aget_tuple({"configurable": {"thread_id": cid}})
        raw = saved.checkpoint["channel_values"]["messages"]
        assert raw[0].content == report["turns"][0]["question"]
        if mode == "demo":
            assert any("层1 降级" in line for line in tier)
            assert any("summary trigger" in line for line in tier)
            assert any("summary done 第1段" in line for line in tier)
            summary = "\n".join(line for line in tier if "summary done" in line)
            assert all(value in summary for value in ("1001", "13800138000", "换货"))
        else:
            assert not tier
        results[mode] = {
            "conversation_id": cid,
            "successful_turns": sum(t["done"] for t in report["turns"]),
            "model_calls": len(model),
            "history_ctx_rounds": len(history),
            "max_estimated_input_tokens": max(m["tokens"] for m in model),
            "checkpoint_messages": len(raw),
            "fixed_prefix": True,
            "original_first_question_retained": True,
            "tier_trace": tier,
            "full_acceptance_complete": report["passed"],
        }
    result = {
        "context_checks_passed": True,
        "model_mode": "offline",
        "full_acceptance_complete": all(r["full_acceptance_complete"] for r in results.values()),
        "results": results,
    }
    (ROOT / "dev-notes/ch07-context-verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
