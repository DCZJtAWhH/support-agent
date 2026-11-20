"""定义接口输入约束，以及售后信息提取结果的字段与取值范围。"""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

InputText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16384)]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: InputText
    conversation_id: UUID | None = None
    request_id: UUID | None = None
    category: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
        | None
    ) = None


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: InputText


class AfterSalesRequest(BaseModel):
    """从顾客售后描述中提取的信息，不代表订单核实或已批准方案。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    order_id: str | None = Field(
        description="唯一明确的订单号，缺失或多个订单无法唯一确定时为 null"
    )
    request_type: Literal["退款", "退货退款", "换货", "维修", "其他", "未知"] = Field(
        description="顾客明确表达的诉求分类，未表达或冲突无法唯一确定时为未知"
    )
    expected_solution: str | None = Field(
        description="顾客明确表达的具体处理期望，未表达时为 null；保留多订单的对应关系"
    )
