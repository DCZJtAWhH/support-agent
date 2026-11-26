"""将上游调用与模型输出异常转换为可返回给调用方的错误信息。"""

from json import JSONDecodeError

from langchain_core.exceptions import OutputParserException
from openai import (
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
)
from pydantic import ValidationError

from support_agent.platform.model_budget import BudgetExceeded


class ModelResponseError(ValueError):
    """上游返回了不适用于本次请求的内容。"""


def model_error(
    exc: Exception, *, structured_output_method: str | None = None
) -> dict[str, str | int]:
    """将模型异常映射为错误码与中文提示，避免返回上游响应体等敏感信息。

    structured_output_method 用于识别结构化输出参数被拒绝的情况，并提供配置提示。
    """
    cause = exc
    while cause is not None:
        if isinstance(cause, BudgetExceeded):
            return {
                "code": "budget_exceeded",
                "message": "真实模型测试预算不足或扣费不确定，已停止调用。",
            }
        cause = cause.__cause__
    if isinstance(exc, APITimeoutError):
        return {"code": "upstream_timeout", "message": "模型响应超时，请稍后重试。"}
    if isinstance(exc, ModelResponseError):
        return {"code": "invalid_model_response", "message": str(exc)}
    if isinstance(exc, APIStatusError):
        # 只用上游原文做分类，绝不将响应体、请求头或 URL 回传给调用方。
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        error = error if isinstance(error, dict) else {}
        message = str(error.get("message", "")).lower()
        param = error.get("param")
        format_rejected = param in (
            "response_format",
            "response_format.type",
            "response_format.json_schema",
        ) or any(word in message for word in ("response_format", "json_schema", "json schema"))
        if structured_output_method and exc.status_code in (400, 422) and format_rejected:
            hint = (
                "请在 .env 中显式设置 STRUCTURED_OUTPUT_METHOD=json_mode 并重启服务。"
                if structured_output_method == "json_schema"
                else "当前 json_mode 要求上游支持 response_format=json_object，请核对上游能力。"
            )
            return {
                "code": "structured_output_rejected",
                "message": f"上游拒绝结构化输出参数（{structured_output_method}）。{hint}",
                "upstream_status": exc.status_code,
                "error_type": type(exc).__name__,
            }
        return {
            "code": "upstream_error",
            "message": "上游拒绝请求，请根据 HTTP 状态检查配置、权限或服务状态。",
            "upstream_status": exc.status_code,
            "error_type": type(exc).__name__,
        }
    if isinstance(
        exc,
        (
            ValidationError,
            OutputParserException,
            JSONDecodeError,
            ContentFilterFinishReasonError,
            LengthFinishReasonError,
        ),
    ):
        return {
            "code": "invalid_model_response",
            "message": "模型未返回符合字段约束的完整内容，请检查上游能力。",
        }
    return {
        "code": "upstream_error",
        "message": "模型调用失败，请检查 .env 中的地址、模型、密钥与上游能力。",
        "error_type": type(exc).__name__,
    }
