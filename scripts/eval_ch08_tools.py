"""用标注样例评估真实模型的工具选择、追问、修参和确认结果复述；不执行写操作。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.agent import model_step  # noqa: E402
from support_agent.config import Settings  # noqa: E402
from support_agent.knowledge_mining import KnowledgeLLM  # noqa: E402
from support_agent.model_budget import BudgetLedger, BudgetTransport  # noqa: E402
from support_agent.tool_catalog import ToolCatalog  # noqa: E402
from support_agent.workflow import AGENT_PROMPT  # noqa: E402


@tool
async def echo_support_code(code: str) -> dict:
    """原样回显顾客提供的服务码，不连接其他系统，不执行写操作。"""
    return {"code": code}


async def main():
    """逐条记录模型实际输出，与人工标签比较，不把离线协议替身当作语义评估。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/ch08-tools.jsonl")
    parser.add_argument("--output", default="dev-notes/ch08-prompt-eval.json")
    parser.add_argument("--case", action="append", help="只执行指定样例，可重复传入")
    args = parser.parse_args()
    settings = Settings()
    catalog = ToolCatalog(
        servers={
            "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:8101/mcp"},
            "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:8102/mcp"},
        },
        policy_path=settings.mcp_policy_path,
    )
    await catalog.refresh()
    if catalog.discovery_errors:
        raise RuntimeError("请先启动两个 MCP Server，再评估真实工具定义。")
    catalog.register(echo_support_code)
    registry = catalog.bind(None, "eval", "eval", "")
    cases = [
        json.loads(line)
        for line in (ROOT / args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
    rows = []
    output = ROOT / args.output
    ledger = BudgetLedger(settings.model_test_budget_path or ROOT / "data/ch08-real-budget.sqlite")
    async with httpx.AsyncClient(
        transport=BudgetTransport(httpx.AsyncHTTPTransport(retries=0), ledger)
    ) as client:
        model = KnowledgeLLM(settings, client).model
        for case in cases:
            row = {"id": case["id"], "passed": False}
            budget_blocked = False
            try:
                messages = [SystemMessage(AGENT_PROMPT)]
                for question, answer in case.get("history", []):
                    messages.extend([HumanMessage(question), AIMessage(answer)])
                messages.append(HumanMessage(case["message"]))
                if "observation" in case:
                    observation = case["observation"]
                    messages.extend(
                        [
                            AIMessage(
                                "",
                                tool_calls=[
                                    {
                                        "id": "observed",
                                        "name": observation["name"],
                                        "args": observation["args"],
                                    }
                                ],
                            ),
                            ToolMessage(
                                json.dumps(observation["result"], ensure_ascii=False),
                                tool_call_id="observed",
                            ),
                        ]
                    )
                decision = None
                async for event in model_step(
                    model,
                    registry,
                    messages,
                    input_budget=settings.input_token_budget,
                    output_budget=settings.max_output_tokens,
                    chars_per_token=settings.token_chars_per_token,
                ):
                    if event["event"] == "decision":
                        decision = event["message"]
                if decision is None:
                    raise RuntimeError("模型未完成决策")
                preliminary = []
                expected = case["expected_tool"]
                if (
                    expected
                    and decision.tool_calls
                    and expected not in [c["name"] for c in decision.tool_calls]
                ):
                    if all(
                        registry.permissions.get(c["name"]) == "read" for c in decision.tool_calls
                    ):
                        preliminary = decision.tool_calls
                        messages.append(decision)
                        for call in decision.tool_calls:
                            result = await registry.execute(
                                call["name"], call["args"], tool_call_id=call["id"]
                            )
                            messages.append(
                                ToolMessage(
                                    json.dumps(result, ensure_ascii=False), tool_call_id=call["id"]
                                )
                            )
                        async for event in model_step(
                            model,
                            registry,
                            messages,
                            input_budget=settings.input_token_budget,
                            output_budget=settings.max_output_tokens,
                        ):
                            if event["event"] == "decision":
                                decision = event["message"]
                names = [call["name"] for call in decision.tool_calls]
                expected = case["expected_tool"]
                checks = {
                    "tool_choice": expected in names if expected else not names,
                    "forbidden_tools": not set(names).intersection(case.get("forbidden_tools", [])),
                    "reply": any(word in decision.content for word in case["reply_any"])
                    if "reply_any" in case
                    else True,
                }
                if expected == "create_ticket":
                    call = next(c for c in decision.tool_calls if c["name"] == expected)
                    checks["description"] = case["description_contains"] in call["args"].get(
                        "description", ""
                    )
                    checks["literal_description"] = (
                        call["args"].get("description", "") in case["message"]
                    )
                if "order_id" in case:
                    checks["corrected_type"] = any(
                        c["args"].get("order_id") == case["order_id"] for c in decision.tool_calls
                    )
                row.update(
                    answer=decision.content,
                    calls=[*preliminary, *decision.tool_calls],
                    checks=checks,
                    passed=all(checks.values()),
                )
            except Exception as exc:
                cause = exc
                while cause is not None:
                    if type(cause).__name__ == "BudgetExceeded":
                        budget_blocked = True
                    cause = cause.__cause__
                row["error"] = type(exc).__name__
                row["error_detail"] = str(exc).replace(
                    settings.openai_api_key.get_secret_value(), "<REDACTED>"
                )
            rows.append(row)
            output.write_text(
                json.dumps(
                    {
                        "model": settings.openai_model,
                        "passed": sum(r["passed"] for r in rows),
                        "total": len(cases),
                        "cases": rows,
                        "cost": ledger.report(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"{case['id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
            (ROOT / "dev-notes/ch08-cost-report.json").write_text(
                json.dumps(ledger.report(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if ledger.report()["halted"] or budget_blocked:
                print("扣费状态不确定，停止全部后续真实测试。", flush=True)
                break
    if not all(r["passed"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
