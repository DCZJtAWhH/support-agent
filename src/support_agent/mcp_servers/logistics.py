"""通过官方 MCP SDK 的 Streamable HTTP 提供随机物流演示数据。"""

import argparse
import asyncio
import os
import random
from datetime import datetime
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field


def create_server(*, port=8101):
    """构造无状态 HTTP 服务；端口可注入以便独立进程验收。"""
    server = FastMCP(
        "logistics",
        host="127.0.0.1",
        port=port,
        stateless_http=True,
        json_response=True,
    )

    @server.tool()
    async def query_logistics(
        order_id: Annotated[str, Field(min_length=1, max_length=64)],
    ) -> dict:
        """按订单号查询随机演示物流轨迹、当前位置与预计剩余天数；不连接真实物流。"""
        await asyncio.sleep(float(os.environ.get("MCP_MOCK_DELAY_SECONDS", "0")))
        location = random.choice(["杭州转运中心", "上海分拨中心", "苏州配送站"])
        return {
            "order_id": order_id,
            "carrier": random.choice(["顺丰", "中通", "圆通"]),
            "status": "in_transit",
            "location": location,
            "estimated_days": random.randint(1, 3),
            "timeline": [
                {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "event": "快件已揽收"},
                {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "event": f"到达{location}"},
            ],
            "simulated": True,
        }

    return server


def main():
    """启动独立物流进程；不依附客服 Web 服务的生命周期。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args()
    create_server(port=args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
