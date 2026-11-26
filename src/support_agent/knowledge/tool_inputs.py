"""知识工具的用户原话校验结构。"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class FAQInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    keyword: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)
    ] = Field(description="用户本轮原文中的连续片段，最多32字；优先保留完整问题，例如邮费是多少")
