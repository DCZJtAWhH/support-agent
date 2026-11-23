"""从环境变量或 .env 加载模型及数据库配置，并校验上下文预算。"""

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    openai_base_url: str
    openai_api_key: SecretStr
    openai_model: str = Field(min_length=1)
    structured_output_method: Literal["json_schema", "json_mode"] = "json_schema"
    output_token_parameter: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    model_timeout_seconds: float = Field(default=60, gt=0)
    max_output_tokens: int = Field(default=1024, gt=0)
    context_window_tokens: int = Field(default=8192, gt=0)
    token_safety_margin: int = Field(default=256, ge=0)
    database_url: SecretStr
    tool_timeout_seconds: float = Field(default=30, gt=0)
    tool_max_retries: int = Field(default=1, ge=0, le=3)
    siliconflow_api_key: SecretStr = SecretStr("")
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    milvus_hybrid_collection: str = "knowledge_hybrid_v1"
    rerank_min_score: float = Field(default=0.05, ge=0, le=1)
    knowledge_top_k: int = Field(default=3, ge=1, le=20)
    knowledge_min_score: float = Field(default=0.4, ge=-1, le=1)
    checkpoint_path: str = "data/ch05-checkpoints.sqlite"
    agent_max_steps: int = Field(default=6, ge=1, le=20)
    agent_token_budget: int = Field(default=50000, gt=0)

    @model_validator(mode="after")
    def validate_budget(self):
        """确保预留输出和安全余量后仍有输入空间，否则抛出配置校验错误。"""
        if self.input_token_budget <= 0:
            raise ValueError("CONTEXT_WINDOW_TOKENS 必须大于输出预算与安全余量之和。")
        return self

    @property
    def input_token_budget(self) -> int:
        """返回上下文窗口扣除最大输出 token 数与安全余量后的输入预算。"""
        return self.context_window_tokens - self.max_output_tokens - self.token_safety_margin
