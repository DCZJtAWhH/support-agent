"""第七章离线验收模型 HTTP 替身，不读取模型密钥、不访问模型网络。"""

import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.app import create_app  # noqa: E402
from support_agent.config import Settings  # noqa: E402


def response(text="", *, call=None):
    """返回受控 SSE，经过真实 LangChain HTTP 解析路径。"""
    delta, finish = {"content": text}, "stop"
    if call:
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": "query_order",
                        "arguments": json.dumps({"order_id": call}),
                    },
                }
            ]
        }
        finish = "tool_calls"
    delta["role"] = "assistant"
    frames = []
    for change, reason in ((delta, None), ({}, finish)):
        frames.append(
            "data: "
            + json.dumps(
                {
                    "id": "offline",
                    "object": "chat.completion.chunk",
                    "model": "offline-model",
                    "choices": [{"index": 0, "delta": change, "finish_reason": reason}],
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


def summarize(data):
    """从本批提取验收事实，不处理 background；仅作协议替身，不评判 Prompt 质量。"""
    source = "\n".join(m["content"] for m in data["new_dialogue"])
    orders = list(dict.fromkeys(re.findall(r"订单\s*(\d{4,})", source)))
    facts = ["用户问过" + "、".join("订单" + value for value in orders)] if orders else []
    phones = list(dict.fromkeys(re.findall(r"1\d{10}", source)))
    if phones:
        facts.append("手机号" + "、".join(phones))
    if "BX-500" in source:
        facts.append("商品BX-500保温杯")
    if "1001" in orders and "换货" in source:
        facts.append("订单1001的诉求是换货，尚未解决")
    if "1002" in orders:
        facts.append("订单1002仅查询状态")
    return "；".join(facts)[:200] if facts else "本段无业务事实"


def create_offline_app(mode=None):
    """复用生产图、仓储与 HTTP 接口，只替换最外层模型 HTTP 边界。"""
    mode = mode or os.environ.get("CH07_OFFLINE_MODE", "demo")
    assert mode in {"demo", "default"}
    directory = ROOT / "data/ch07-offline"
    directory.mkdir(parents=True, exist_ok=True)
    config = Settings(
        _env_file=None,
        openai_base_url="https://offline.invalid/v1",
        openai_api_key="offline",
        openai_model="offline-model",
        router_model="offline-router",
        siliconflow_api_key="",
        database_url="sqlite+aiosqlite:///" + (directory / "conversations.db").as_posix(),
        checkpoint_path=str(directory / "checkpoints.sqlite"),
        model_context_window=18000 if mode == "demo" else 128000,
        max_output_tokens=2000 if mode == "demo" else 1024,
        max_user_input_tokens=2000,
        max_agent_steps=3 if mode == "demo" else 6,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
        token_safety_margin=250,
        token_chars_per_token=2,
        system_tools_tokens=1000,
        evidence_tokens_per_result=100,
        summary_injection_tokens=200,
        history_target_rounds=40,
        steady_round_tokens=1000,
        agent_token_budget=1000000,
    )

    async def handle(request):
        """回忆题只从收到的投影取事实，替身自身没有会话记忆。"""
        assert str(request.url) == "https://offline.invalid/v1/chat/completions"
        messages = json.loads(request.content)["messages"]
        prompt = messages[0]["content"]
        if "事实摘要员" in prompt:
            return response(summarize(json.loads(messages[-1]["content"])))
        current = next(
            m["content"]
            for m in reversed(messages)
            if m["role"] == "user"
            and not m["content"].startswith(("以下仅为", "以下是历史对话数据"))
        )
        if "问题理解器" in prompt:
            return response(json.dumps({"question": current}, ensure_ascii=False))
        if prompt.startswith("将顾客"):
            return response('{"intent":"订单","confidence":0.99}')
        if "最开始那个订单" in current:
            if mode == "demo":
                background = next(
                    (
                        json.loads(m["content"].split("\n", 1)[1])
                        for m in messages
                        if m["role"] == "user" and m["content"].startswith("以下仅为")
                    ),
                    {},
                )
                known = background.get("summaries", "")
            else:
                known = "\n".join(m.get("content") or "" for m in messages[1:])
            return response(
                "最开始是订单1001，您的诉求为换货，目前尚未解决。"
                if "1001" in known and "换货" in known
                else "没有足够历史，无法回忆。"
            )
        order = re.search(r"订单\s*(\d+)", current)
        order_id = order[1] if order else "1002"
        if messages[-1]["role"] != "tool":
            return response(call=order_id)
        return response(
            f"演示订单{order_id}已核对，换货未办理。" + "请保留相关原始信息，待核实后处理。" * 45
        )

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    app = create_app(settings=config, upstream_client=upstream)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        """管理离线 HTTP 替身和原应用的完整生命周期。"""
        async with upstream, original_lifespan(application):
            yield

    app.router.lifespan_context = lifespan

    @app.get("/_offline/status")
    async def status():
        """发消息前验证离线标记，避免测试脚本误用真实模型服务。"""
        return {"offline": True, "mode": mode, "checkpoint_path": config.checkpoint_path}

    return app
