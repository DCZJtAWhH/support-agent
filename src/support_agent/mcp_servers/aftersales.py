"""通过官方 MCP SDK 提供随机在保和退货进度演示，不访问数据库。"""

import argparse
import random
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field


def create_server(*, port=8102):
    """创建与物流进程隔离的售后 Streamable HTTP 服务。"""
    server = FastMCP(
        "aftersales",
        host="127.0.0.1",
        port=port,
        stateless_http=True,
        json_response=True,
    )

    @server.tool()
    async def query_warranty(
        order_id: Annotated[str, Field(min_length=1, max_length=64)],
    ) -> dict:
        """按订单号随机返回商品在保情况，仅为演示数据。"""
        remaining = random.choice([0, 30, 180, 365])
        return {
            "order_id": order_id,
            "status": "in_warranty" if remaining else "expired",
            "remaining_days": remaining,
            "simulated": True,
        }

    @server.tool()
    async def query_return_progress(
        order_id: Annotated[str, Field(min_length=1, max_length=64)],
    ) -> dict:
        """按订单号随机查询退货处理进度，不承诺实际退款或处理时效。"""
        return {
            "order_id": order_id,
            "status": random.choice(["processing", "received", "refunded"]),
            "updated_at": "本次演示查询时生成",
            "simulated": True,
        }

    return server


def main():
    """启动独立售后服务进程，供客服 MultiServerMCPClient 连接。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8102)
    args = parser.parse_args()
    create_server(port=args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
