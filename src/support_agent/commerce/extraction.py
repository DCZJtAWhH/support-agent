"""从售后描述提取结构化事实，提取不代表订单已核实。"""

import json
import warnings

from langchain_core.messages import HumanMessage, SystemMessage

from support_agent.platform.context import fit_messages
from support_agent.platform.errors import ModelResponseError

from .prompts import AFTER_SALES_EXTRACTION
from .schemas import AfterSalesRequest


class AfterSalesExtractor:
    def __init__(self, model, config):
        """绑定模型与结构化协议，禁止静默降级输出方式。"""
        self.model, self.config = model, config

    async def extract(self, description):
        """校验输入预算与完整 JSON；模型调用不持有业务事务，无业务写入。"""
        schema = json.dumps(AfterSalesRequest.model_json_schema(), ensure_ascii=False)
        schema_cost = len(schema) if self.config.structured_output_method == "json_schema" else 0
        messages = fit_messages(
            SystemMessage(AFTER_SALES_EXTRACTION.format(schema=schema)),
            [],
            HumanMessage(description),
            self.config.input_token_budget - schema_cost,
        )
        options = {"strict": True} if self.config.structured_output_method == "json_schema" else {}
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message=".*Overriding.*", category=UserWarning)
                structured = self.model.with_structured_output(
                    AfterSalesRequest,
                    method=self.config.structured_output_method,
                    include_raw=True,
                    **options,
                )
        except UserWarning as exc:
            raise ModelResponseError(
                "LangChain 无法保留所选结构化方式；请在 .env 中明确选择兼容方式，"
                "不会降级为工具调用。"
            ) from exc
        result = await structured.ainvoke(messages)
        if (
            result["parsing_error"] is not None
            or result["parsed"] is None
            or result["raw"].response_metadata.get("finish_reason") != "stop"
        ):
            raise ModelResponseError("模型未返回符合售后字段约束的完整 JSON，请检查上游能力。")
        return result["parsed"]
