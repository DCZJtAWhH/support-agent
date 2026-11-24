"""定义接口输入约束，以及售后信息提取结果的字段与取值范围。"""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints

InputText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16384)]
REFUND_REASONS = [
    "不想要了",
    "商品质量问题",
    "收到商品破损",
    "商品与描述不符",
    "发错货或少件",
    "其他",
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: InputText
    conversation_id: UUID | None = None
    request_id: UUID | None = None
    ticket_confirmation: StrictBool | None = None
    ticket_tool_call_id: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = (
        None
    )
    selected_order_id: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)] | None
    ) = None
    category: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
        | None
    ) = None


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: InputText


class TicketConfirmation(BaseModel):
    """显式建单确认仅引用已完成轮次，不接受客户端伪造的描述或处理结果。"""

    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID
    request_id: UUID


class RefundSubmission(BaseModel):
    """只接受原轮次与固定退款原因；订单从服务端已核实快照中读取。"""

    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID
    request_id: UUID
    reason: Literal[
        "不想要了", "商品质量问题", "收到商品破损", "商品与描述不符", "发错货或少件", "其他"
    ]


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
