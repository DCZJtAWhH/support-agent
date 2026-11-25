"""从环境变量或 .env 加载模型及数据库配置，并校验上下文预算。"""

import logging
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    openai_base_url: str
    openai_api_key: SecretStr
    openai_model: str = Field(min_length=1)
    model_provider: Literal["auto", "openai", "deepseek"] = "auto"
    model_thinking: Literal["enabled", "disabled"] | None = None
    model_test_budget_path: str | None = None
    langfuse_enabled: bool = True
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_base_url: str = "http://127.0.0.1:3000"
    structured_output_method: Literal["json_schema", "json_mode"] = "json_schema"
    output_token_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    model_timeout_seconds: float = Field(default=60, gt=0)
    max_output_tokens: int = Field(default=1024, gt=0)
    context_window_tokens: int = Field(
        default=128000,
        gt=0,
        validation_alias=AliasChoices("model_context_window", "context_window_tokens"),
    )
    token_safety_margin: int = Field(default=250, ge=0)
    max_user_input_tokens: int = Field(default=2000, gt=0)
    tool_result_max_tokens: int = Field(default=1200, gt=0)
    rerank_top_k: int = Field(default=5, ge=1, le=20)
    # 中文按两字符约一 token，与下面的稳态/峰值预算共同校准。
    token_chars_per_token: float = Field(default=2.0, ge=1, le=4)
    system_tools_tokens: int = Field(default=1000, gt=0)
    evidence_tokens_per_result: int = Field(default=100, gt=0)
    summary_injection_tokens: int = Field(default=200, gt=0)
    history_target_rounds: int = Field(default=40, ge=1)
    steady_round_tokens: int = Field(default=1000, gt=0)
    layer2_assistant_chars: int = Field(default=60, ge=1)
    database_url: SecretStr
    tool_timeout_seconds: float = Field(default=30, gt=0)
    tool_max_retries: int = Field(default=1, ge=0, le=3)
    mcp_servers: dict[str, dict] = Field(default_factory=dict)
    mcp_policy_path: str = "config/tool-permissions.json"
    siliconflow_api_key: SecretStr = SecretStr("")
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    milvus_hybrid_collection: str = "knowledge_hybrid_v1"
    rerank_min_score: float = Field(default=0.05, ge=0, le=1)
    evidence_calibration_path: str = str(
        Path(__file__).resolve().parents[2] / "config/evidence-confidence.json"
    )
    flywheel_interval_seconds: float = Field(default=10, ge=0)
    eval_interval_seconds: float = Field(default=86400, ge=0)
    eval_max_output_tokens: int = Field(default=4096, ge=1024)
    knowledge_top_k: int = Field(default=3, ge=1, le=20)
    knowledge_min_score: float = Field(default=0.4, ge=-1, le=1)
    checkpoint_path: str = "data/ch05-checkpoints.sqlite"
    intent_min_confidence: float = Field(default=0.65, ge=0, le=1)
    router_model: str | None = Field(default=None, min_length=1)
    agent_max_steps: int = Field(
        default=6,
        ge=1,
        le=20,
        validation_alias=AliasChoices("max_agent_steps", "agent_max_steps"),
    )
    agent_token_budget: int = Field(default=1000000, gt=0)

    @model_validator(mode="after")
    def validate_budget(self):
        """补齐路由模型默认值，并确保预留输出和安全余量后仍有输入空间。"""
        if self.router_model is None:
            self.router_model = self.openai_model
        if self.input_token_budget <= 0:
            raise ValueError("CONTEXT_WINDOW_TOKENS 必须大于输出预算与安全余量之和。")
        from support_agent.context import ContextBudget

        budget = ContextBudget.from_settings(self)
        if budget.history < self.steady_round_tokens:
            logging.getLogger(__name__).warning(
                "上下文预算不足：历史预算=%s，连一轮稳态=%s 都装不下；"
                "请联调窗口、输入、输出及工具峰值",
                budget.history,
                self.steady_round_tokens,
            )
        return self

    @property
    def input_token_budget(self) -> int:
        """返回上下文窗口扣除最大输出 token 数与安全余量后的输入预算。"""
        return self.context_window_tokens - self.max_output_tokens - self.token_safety_margin
