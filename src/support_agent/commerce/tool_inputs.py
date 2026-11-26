"""电商工具输入与业务字段约束。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OrderInput(ToolInput):
    order_id: Identifier = Field(description="用户提供的订单号，例如1001")


class ProductInput(ToolInput):
    product_id: Identifier = Field(description="用户提供的商品编号或商品名称")


class TicketInput(ToolInput):
    description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ] = Field(
        description="顾客已提供的具体问题，引用顾客原话中的连续片段，不得编造或只填写建单请求"
    )
    ticket_type: Literal["退款", "退货退款", "换货", "维修", "其他", "未知"]
