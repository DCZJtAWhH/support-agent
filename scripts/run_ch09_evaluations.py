"""连续运行完整 ch04 评估，等待每轮成功落库后再启动下一轮。"""

import argparse
import asyncio
import json
import time

import httpx


async def run(args):
    """复用已在运行的一轮，不并发评估；失败立即退出，绝不将部分结果记作成功。"""
    async with httpx.AsyncClient(base_url=args.base_url, timeout=15) as client:
        for number in range(1, args.rounds + 1):
            state = (await client.get("/api/evaluations/status")).json()["job"]
            if not state or state["status"] != "running":
                response = await client.post("/api/evaluations/run")
                response.raise_for_status()
                state = response.json()["job"]
            print(f"轮次 {number}: {state['id']}", flush=True)
            deadline = time.monotonic() + args.timeout
            previous = -1
            while state["status"] == "running":
                if state["completed"] != previous:
                    print(f"{state['completed']}/{state['total']}", flush=True)
                    previous = state["completed"]
                if time.monotonic() >= deadline:
                    raise TimeoutError("评估仍在后台运行，等待超时；请查询状态后继续。")
                await asyncio.sleep(2)
                response = await client.get("/api/evaluations/status")
                response.raise_for_status()
                state = response.json()["job"]
            if state["status"] != "completed":
                raise RuntimeError(state["message"])
        response = await client.get("/api/evaluations/runs")
        response.raise_for_status()
        print(json.dumps(response.json(), ensure_ascii=False, indent=2), flush=True)


def main():
    """默认运行两轮，每轮最多等20分钟，服务端固定使用当前完整评估集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--rounds", type=int, choices=range(1, 11), default=2)
    parser.add_argument("--timeout", type=int, default=1200)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
