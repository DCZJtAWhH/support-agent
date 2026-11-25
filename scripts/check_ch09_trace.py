"""向真实客服发送知识与工具请求，并从本地 Langfuse v4 回读完整链路证据。"""

import asyncio
import json
import sys
from pathlib import Path

import httpx
from langfuse import Langfuse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_agent.config import Settings  # noqa: E402


async def main():
    """通过官方 observations v2 API 校验树、prompt、检索原文、工具及真实 token。"""
    config = Settings()
    summaries = []
    auth = (
        config.langfuse_public_key.get_secret_value(),
        config.langfuse_secret_key.get_secret_value(),
    )
    async with httpx.AsyncClient(timeout=120) as client:
        for kind, question in [
            ("knowledge", "CH09演示杯支持激光刻字服务吗"),
            ("tools", "请查询订单1001的状态和金额"),
        ]:
            response = await client.post("http://127.0.0.1:8010/chat", json={"message": question})
            assert "event: done" in response.text and "event: error" not in response.text
            meta = json.loads(response.text.split("data: ", 1)[1].splitlines()[0])
            trace_id = Langfuse.create_trace_id(seed=meta["request_id"])
            for _ in range(30):
                observations = await client.get(
                    config.langfuse_base_url + "/api/public/v2/observations",
                    auth=auth,
                    params={
                        "traceId": trace_id,
                        "limit": 1000,
                        "fields": "core,basic,io,usage,metadata,metrics",
                    },
                )
                observations.raise_for_status()
                rows = observations.json()["data"]
                if any(r["name"] == "support-request" and r.get("endTime") for r in rows):
                    break
                await asyncio.sleep(1)
            root = next(r for r in rows if r["name"] == "support-request")
            generations = [r for r in rows if r["type"] == "GENERATION"]
            assert root["metadata"]["intent"]
            assert generations and all(r.get("input") for r in generations)
            assert all(r.get("usageDetails", {}).get("total", 0) > 0 for r in generations)
            assert all(r.get("latency") is not None for r in generations)
            names = {r["name"] for r in rows}
            assert {"resolve", "classify", "log"} <= names
            if kind == "knowledge":
                retrieval = next(r for r in rows if r["name"] == "retrieve")
                assert "retrieval_snapshot" in retrieval.get("output", "")
            else:
                assert any(r["type"] == "TOOL" or r["name"] == "query_order" for r in rows)
            for row in rows:
                row.get("metadata", {}).pop("scope.attributes.public_key", None)
            (ROOT / f"dev-notes/ch09-trace-{kind}.json").write_text(
                json.dumps(
                    {"request_id": meta["request_id"], "trace_id": trace_id, "observations": rows},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            summaries.append(
                {
                    "kind": kind,
                    "trace_id": trace_id,
                    "observations": len(rows),
                    "intent": root["metadata"]["intent"],
                    "tokens": sum(r["usageDetails"]["total"] for r in generations),
                    "url": (
                        f"{config.langfuse_base_url}/project/{root['projectId']}/traces/{trace_id}"
                    ),
                }
            )
    (ROOT / "dev-notes/ch09-trace-verification.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
