"""演示无 LangGraph 的原始工具循环；--offline 使用明确标注的脚本模型。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from support_agent.agent import run_bare_agent  # noqa: E402
from support_agent.config import Settings  # noqa: E402
from support_agent.tools import ToolRegistry, build_tools  # noqa: E402


class OfflineModel:
    """展示订单→物流→回答的确定性脚本模型，不代表真实模型评估。"""

    def bind_tools(self, tools):
        """接收与真实模型相同的只读工具绑定接口。"""
        return self

    async def astream(self, messages):
        """根据已有工具观察推进下一步，不跳过中间结果。"""
        observations = [m for m in messages if m.type == "tool"]
        if len(observations) < 2:
            name = "query_order" if not observations else "query_logistics"
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": name,
                        "args": '{"order_id":"1001"}',
                        "id": f"step-{len(observations)}",
                        "index": 0,
                    }
                ],
                response_metadata={"finish_reason": "tool_calls"},
            )
        else:
            result = json.loads(observations[-1].content)["data"]
            yield AIMessageChunk(
                content="演示物流位置：" + result["location"],
                response_metadata={"finish_reason": "stop"},
            )


async def main():
    """运行可选择真实上游或离线脚本的裸循环，仅使用第二章已有只读工具。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--question", default="先查订单1001，得到结果后再查物流")
    args = parser.parse_args()
    config = None if args.offline else Settings()
    model = (
        OfflineModel()
        if args.offline
        else ChatOpenAI(
            model=config.openai_model,
            base_url=config.openai_base_url,
            api_key=config.openai_api_key.get_secret_value(),
            max_retries=0,
            timeout=config.model_timeout_seconds,
            stream_usage=False,
            use_responses_api=False,
            max_tokens=config.max_output_tokens
            if config.output_token_parameter == "max_completion_tokens"
            else None,
            extra_body={"max_tokens": config.max_output_tokens}
            if config.output_token_parameter == "max_tokens"
            else None,
        )
    )
    registry = ToolRegistry(build_tools(None, "bare-demo", "bare-demo", args.question))
    messages = [
        SystemMessage("你是客服，缺信息就追问。按用户要求分步查工具，结果是随机演示数据。"),
        HumanMessage(args.question),
    ]
    limits = (
        {}
        if config is None
        else {
            "input_budget": config.input_token_budget,
            "output_budget": config.max_output_tokens,
            "token_budget": config.agent_token_budget,
            "max_steps": config.agent_max_steps,
        }
    )
    async for event in run_bare_agent(model, registry, messages, **limits):
        print(json.dumps(event, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
