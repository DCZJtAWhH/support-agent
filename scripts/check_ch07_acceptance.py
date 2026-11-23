"""使用离线模型替身连续聊天23轮，验证摘要级联、回忆及原文回载，不调用真实模型。"""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path
from time import monotonic
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]


def events(text):
    """解析完整 SSE 文本，供验收检查服务端正常结束及真实回复。"""
    result = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        lines = frame.splitlines()
        name = next((line[7:] for line in lines if line.startswith("event: ")), "")
        data = next((line[6:] for line in lines if line.startswith("data: ")), "")
        if name and data:
            result.append((name, json.loads(data)))
    return result


async def main():
    """只创建新的验收会话，既有会话和知识库保持原样；不会提交工单或退款申请。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="可选的离线应用地址，脚本会先检查离线标记")
    parser.add_argument("--mode", choices=["demo", "default"], default="demo")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = ROOT / f"dev-notes/ch07-offline-{args.mode}-acceptance.json"
    report = {"mode": args.mode, "model_mode": "offline", "turns": [], "passed": False}
    cid = None
    if args.resume and output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        cid = report.get("conversation_id")
        report.pop("error", None)
        if report["turns"] and not report["turns"][-1]["done"]:
            report["turns"].pop()
    try:
        async with AsyncExitStack() as resources:
            options = {}
            if not args.base_url:
                from ch07_offline import create_offline_app

                app = create_offline_app(args.mode)
                await resources.enter_async_context(app.router.lifespan_context(app))
                options["transport"] = httpx.ASGITransport(app)
            client = await resources.enter_async_context(
                httpx.AsyncClient(
                    base_url=args.base_url or "http://offline.test",
                    timeout=180,
                    **options,
                )
            )
            status = await client.get("/_offline/status")
            assert status.status_code == 200 and status.json().get("offline"), "仅允许离线服务"
            report["checkpoint_path"] = status.json()["checkpoint_path"]
            for index in range(len(report["turns"]), 23):
                if index == 0:
                    question = (
                        "最开始请记住：我问的是BX-500保温杯，订单1001，手机号13800138000，"
                        "我的明确诉求是换货，目前尚未解决。请确认这些信息。"
                    )
                elif index == 22:
                    question = (
                        "最开始那个订单后来怎么说？请复述当时的订单号和我的诉求，"
                        "别把申请说成已办好。"
                    )
                else:
                    question = (
                        "继续查询另一笔演示订单1002当前状态，以下是补充描述。"
                        "这轮只查状态，不申请退款，不处理最初订单的换货。"
                        + "补充情况：外包装完好，暂时没有发现破损；"
                        "我只是核对订单记录，尚未申请售后。" * 12 + "请简短答复当前订单状态即可。"
                    )
                started = monotonic()
                rid = str(uuid4())
                if cid:
                    prior = (await client.get(f"/conversations/{cid}")).json()
                    failed = [
                        m
                        for m in prior["messages"]
                        if m["role"] == "user"
                        and m["content"] == question
                        and m["state"] == "failed"
                    ]
                    if failed:
                        rid = failed[-1]["request_id"]
                failures = []
                for generation in range(3):
                    for _attempt in range(10):
                        response = await client.post(
                            "/chat",
                            json={"message": question, "conversation_id": cid, "request_id": rid},
                        )
                        if response.status_code != 409 or "正在执行" not in response.text:
                            break
                        await asyncio.sleep(0.15)
                    response.raise_for_status()
                    stream = events(response.text)
                    cid = stream[0][1]["conversation_id"]
                    if stream[-1][0] == "done":
                        break
                    failures.append(stream[-1])
                    print(f"{args.mode} round={index + 1} model_retry={generation + 1}", flush=True)
                answer = "".join(d["text"] for e, d in stream if e == "token")
                row = {
                    "round": index + 1,
                    "question": question,
                    "answer": answer,
                    "request_id": rid,
                    "transient_errors": failures,
                    "elapsed_seconds": round(monotonic() - started, 3),
                    "done": stream[-1][0] == "done",
                }
                audit = (await client.get(f"/conversations/{cid}")).json()
                row.update(
                    summary_count=len(audit["summaries"]),
                    summary_upto=audit["summary_upto_msg_id"],
                    layer1_from=audit["layer1_from_msg_id"],
                )
                report["turns"].append(row)
                report["conversation_id"] = cid
                output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    f"{args.mode} round={index + 1} done={row['done']} "
                    f"summaries={row['summary_count']} elapsed={row['elapsed_seconds']}",
                    flush=True,
                )
                assert row["done"], answer
                assert "本轮查询已达到" not in answer, answer
            # 后台摘要只轮询只读接口；用户回复路径没有等待这个任务。
            audit = (await client.get(f"/conversations/{cid}")).json()
            original = (await client.get(f"/api/conversations/{cid}/messages")).json()
            users = [m["content"] for m in original if m["role"] == "user"]
            assert users == [t["question"] for t in report["turns"]]
            assert bool(audit["summaries"]) == (args.mode == "demo")
            recall = report["turns"][-1]["answer"]
            assert "1001" in recall and "换货" in recall, recall
            report.update(
                passed=True, summaries=audit["summaries"], original_messages=len(original)
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
