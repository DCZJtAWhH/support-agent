"""通过所配置的模型协议执行可校验的 JSON 任务。"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from support_agent.platform.context import fit_messages
from support_agent.platform.model_gateway import create_chat_model


class KnowledgeLLM:
    def __init__(self, config, client, sync_client=None):
        """沿用现有聊天提供方和结构化方式，允许注入 HTTP 边界客户端。"""
        self.config = config
        self.model = create_chat_model(config, client, sync_client)

    async def ask(self, schema, prompt: str, payload: dict):
        """请求并校验完整 JSON；格式错误最多重试两次，不切换模型或结构化协议。"""
        schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        messages = fit_messages(
            SystemMessage(prompt + "\nJSON Schema：" + schema_text),
            [],
            HumanMessage(json.dumps(payload, ensure_ascii=False)),
            self.config.input_token_budget - len(schema_text),
        )
        options = {"strict": True} if self.config.structured_output_method == "json_schema" else {}
        structured = self.model.with_structured_output(
            schema,
            method=self.config.structured_output_method,
            include_raw=True,
            **options,
        )
        for attempt in range(3):
            result = await structured.ainvoke(messages)
            if (
                not result["parsing_error"]
                and result["parsed"] is not None
                and result["raw"].response_metadata.get("finish_reason") == "stop"
            ):
                return result["parsed"]
            logging.getLogger(__name__).warning(
                "结构化输出无效 schema=%s attempt=%s finish=%s error_type=%s",
                schema.__name__,
                attempt + 1,
                result["raw"].response_metadata.get("finish_reason"),
                type(result["parsing_error"]).__name__,
            )
        raise ValueError("知识模型未返回完整且符合约束的 JSON；未记录成功检查点，请重跑。")
