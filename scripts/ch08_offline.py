"""第八章离线界面验收：只替换模型 HTTP 边界，MCP、图、数据库均走实际实现。"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx

from support_agent.app import create_app  # noqa: E402
from support_agent.platform.config import Settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def response(text="", *, name=None, arguments=None):
    """生成与 OpenAI 聊天流协议一致的模型替身响应。"""
    delta = {"role": "assistant", "content": text}
    finish = "stop"
    if name:
        delta["tool_calls"] = [
            {
                "index": 0,
                "id": uuid4().hex,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ]
        finish = "tool_calls"
    frames = []
    for content, reason in [(delta, None), ({}, finish)]:
        frames.append(
            "data: "
            + json.dumps(
                {
                    "id": "ch08-offline",
                    "object": "chat.completion.chunk",
                    "model": "offline",
                    "choices": [{"index": 0, "delta": content, "finish_reason": reason}],
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
    return httpx.Response(
        200,
        text="".join(frames) + "data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


def create_offline_app():
    """提供可复现的本地演示应用；不会读取模型密钥或连接真实业务系统。"""
    directory = ROOT / "data/ch08-offline"
    directory.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        _env_file=None,
        openai_base_url="https://offline.invalid/v1",
        openai_api_key="offline",
        openai_model="offline",
        database_url=f"sqlite+aiosqlite:///{directory / 'chat.db'}",
        checkpoint_path=str(directory / "checkpoints.sqlite"),
        mcp_servers={
            "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:8101/mcp"},
            "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:8102/mcp"},
        },
    )

    async def handle(request):
        """按演示输入产生决策，工具数据始终取真实执行观察，不伪造工单或物流结果。"""
        payload = json.loads(request.content)
        messages = payload["messages"]
        current = next(
            m["content"]
            for m in reversed(messages)
            if m["role"] == "user" and not m["content"].startswith("以下仅为")
        )
        system = messages[0]["content"]
        if "问题理解器" in system:
            return response(json.dumps({"question": current}, ensure_ascii=False))
        if system.startswith("将顾客"):
            intent = "投诉" if "投诉" in current and "工单" not in current else "订单"
            return response(json.dumps({"intent": intent, "confidence": 0.99}, ensure_ascii=False))
        if messages[-1]["role"] == "tool":
            result = json.loads(messages[-1]["content"])
            if not result["ok"]:
                return response("本次未创建工单。" + result["error"]["message"])
            data = result["data"]
            if "ticket_no" in data:
                return response(
                    f"已登记待处理工单，工单号：{data['ticket_no']}。将由人工继续跟进。"
                )
            return response("以下是本次查询的演示数据：" + json.dumps(data, ensure_ascii=False))
        if any(word in current for word in ["不加热", "损坏", "漏水"]):
            return response(
                name="create_ticket", arguments={"description": current, "ticket_type": "维修"}
            )
        if "工单" in current:
            return response("请描述商品遇到的具体问题，以及希望如何处理。")
        if "物流" in current:
            return response(name="query_logistics", arguments={"order_id": "1001"})
        if "在保" in current:
            return response(name="query_warranty", arguments={"order_id": "1001"})
        if "退货进度" in current:
            return response(name="query_return_progress", arguments={"order_id": "1001"})
        return response("您好，可以查询订单物流、在保与退货进度，也可以明确要求建工单。")

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    app = create_app(settings, upstream)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        """共同释放模型 HTTP 替身与原应用实际资源。"""
        async with upstream, original_lifespan(application):
            yield

    app.router.lifespan_context = lifespan
    return app
