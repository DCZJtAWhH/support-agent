"""将提供方协议差异限制在模型接入层，业务工具只使用统一 LangChain 接口。"""

from typing import Any
from urllib.parse import urlsplit

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI


class DeepSeekChat(ChatDeepSeek):
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        """补齐当前官方适配器遗漏的消息字段回传；不修改正文或虚构推理内容。"""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for message, row in zip(messages, payload["messages"], strict=True):
            if message.type == "ai":
                row["reasoning_content"] = message.additional_kwargs.get("reasoning_content", "")
        return payload


def create_chat_model(config, async_client=None, sync_client=None):
    """按显式配置选择协议；auto 只识别官方域名，其他 OpenAI 兼容地址走通用适配。"""
    provider = config.model_provider
    if provider == "auto":
        provider = (
            "deepseek"
            if urlsplit(config.openai_base_url).hostname == "api.deepseek.com"
            else "openai"
        )
    model_class = DeepSeekChat if provider == "deepseek" else ChatOpenAI
    options: dict[str, Any] = {
        "model": config.openai_model,
        "base_url": config.openai_base_url,
        "api_key": config.openai_api_key.get_secret_value(),
        "http_async_client": async_client,
        "http_client": sync_client,
        "timeout": config.model_timeout_seconds,
        "max_retries": 0,
        "stream_usage": provider == "deepseek",
        "use_responses_api": False,
    }
    if provider == "deepseek" or config.output_token_parameter == "max_tokens":
        options["extra_body"] = {"max_tokens": config.max_output_tokens}
    else:
        options["max_tokens"] = config.max_output_tokens
    if config.model_thinking is not None:
        options.setdefault("extra_body", {})["thinking"] = {"type": config.model_thinking}
    return model_class(**options)
