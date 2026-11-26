"""定义接口输入约束，以及售后信息提取结果的字段与取值范围。"""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StrictBool, StringConstraints

InputText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16384)]


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
