import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import httpx

from support_agent.tooling.catalog import ToolCatalog


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextmanager
def running_server(script, port, log_path):
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=log,
            stderr=log,
            env={**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            for _ in range(150):
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text(encoding="utf-8"))
                try:
                    httpx.get(f"http://127.0.0.1:{port}/mcp", timeout=0.2)
                    break
                except httpx.TransportError:
                    time.sleep(0.05)
            else:
                raise TimeoutError("MCP Server 未就绪")
            yield process
        finally:
            process.terminate()
            process.wait(timeout=10)


def test_two_processes_discovery_call_and_server_only_restart(tmp_path):
    logistics_port, aftersales_port = free_port(), free_port()
    logistics = tmp_path / "logistics.py"
    aftersales = tmp_path / "aftersales.py"
    logistics_setup = (
        "from support_agent.mcp_servers.logistics import create_server\n"
        f"server = create_server(port={logistics_port})\n"
    )
    logistics.write_text(
        logistics_setup + "server.run(transport='streamable-http')\n", encoding="utf-8"
    )
    aftersales.write_text(
        "from support_agent.mcp_servers.aftersales import create_server\n"
        f"create_server(port={aftersales_port}).run(transport='streamable-http')\n",
        encoding="utf-8",
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "logistics": {"query_logistics": "read", "query_delivery_window": "read"},
                "aftersales": {"query_warranty": "read", "query_return_progress": "read"},
            }
        ),
        encoding="utf-8",
    )
    catalog = ToolCatalog(
        servers={
            "logistics": {
                "transport": "streamable_http",
                "url": f"http://127.0.0.1:{logistics_port}/mcp",
            },
            "aftersales": {
                "transport": "streamable_http",
                "url": f"http://127.0.0.1:{aftersales_port}/mcp",
            },
        },
        policy_path=policy,
    )
    assert "query_logistics" not in catalog.entries.tools

    async def query_original():
        await catalog.refresh()
        registry = catalog.bind("conversation", "request", "物流查询")
        for name in ["query_logistics", "query_warranty", "query_return_progress"]:
            result = await registry.execute(name, {"order_id": "1001"})
            assert result["ok"], result
            assert result["data"]["simulated"]
            assert registry.sources[name].startswith("mcp:")
        assert "query_delivery_window" not in registry.tools

    with running_server(aftersales, aftersales_port, tmp_path / "after.log"):
        with running_server(logistics, logistics_port, tmp_path / "logistics.log"):
            asyncio.run(query_original())

        async def unavailable_server_is_isolated():
            await catalog.refresh()
            registry = catalog.bind("c", "r", "查询")
            assert "query_logistics" not in registry.tools
            assert "logistics" in catalog.discovery_errors
            assert (await registry.execute("query_warranty", {"order_id": "1001"}))["ok"]

        asyncio.run(unavailable_server_is_isolated())
        logistics.write_text(
            logistics_setup
            + '''
@server.tool()
def query_delivery_window(order_id: str) -> dict:
    """查询演示配送时间段。"""
    return {"order_id": order_id, "window": "明天下午", "simulated": True}
server.run(transport="streamable-http")
''',
            encoding="utf-8",
        )

        async def query_new():
            await catalog.refresh()
            registry = catalog.bind("conversation", "next", "配送时间段")
            result = await registry.execute("query_delivery_window", {"order_id": "1001"})
            assert result["ok"] and result["data"]["window"] == "明天下午"
            policy.write_text("{}", encoding="utf-8")
            denied = await catalog.bind("c", "r", "").execute(
                "query_delivery_window",
                {"order_id": "1001"},
            )
            assert denied["error"]["code"] == "permission_denied"

        with running_server(logistics, logistics_port, tmp_path / "logistics.log"):
            asyncio.run(query_new())
