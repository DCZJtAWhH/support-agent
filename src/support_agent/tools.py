"""注册 LangChain 业务工具，集中校验参数并处理超时、重试与脱敏错误。"""

import asyncio
import random
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from sqlalchemy.exc import OperationalError

from support_agent.repositories import ChatRepository

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OrderInput(ToolInput):
    order_id: Identifier = Field(description="用户提供的订单号，例如1001")


class ProductInput(ToolInput):
    product_id: Identifier = Field(description="用户提供的商品编号或商品名称")


class FAQInput(ToolInput):
    keyword: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)
    ] = Field(description="用户本轮原文中的连续片段，最多32字；优先保留完整问题，例如邮费是多少")


class TicketInput(ToolInput):
    description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    ticket_type: Literal["退款", "退货退款", "换货", "维修", "其他", "未知"]


def build_tools(
    repo: ChatRepository, conversation_id: str, request_id: str, user_text: str, *, category=None
) -> list[BaseTool]:
    """为当前轮次绑定可信会话和请求标识；模型不能自行指定身份或工单去重键。"""

    @tool(args_schema=OrderInput)
    async def query_order(order_id: str) -> dict:
        """按订单号查询订单概况。返回随机演示数据，不访问真实电商系统。"""
        return {
            "order_id": order_id,
            "status": random.choice(["已支付", "已发货", "已签收"]),
            "total": random.choice([99, 199, 299]),
            "currency": "CNY",
            "simulated": True,
        }

    @tool(args_schema=ProductInput)
    async def query_product(product_id: str) -> dict:
        """按商品编号或名称查询商品价格和库存。返回随机演示数据，不访问真实电商系统。"""
        return {
            "product_id": product_id,
            "price": random.choice([59, 99, 199]),
            "stock": random.randint(0, 100),
            "currency": "CNY",
            "simulated": True,
        }

    @tool(args_schema=OrderInput)
    async def query_logistics(order_id: str) -> dict:
        """直接按订单号查询物流位置和状态，无需先查订单。返回随机演示数据，不查询真实物流。"""
        return {
            "order_id": order_id,
            "carrier": random.choice(["顺丰", "中通", "圆通"]),
            "status": "运输中",
            "location": random.choice(["杭州转运中心", "上海分拨中心", "苏州配送站"]),
            "estimated_days": random.randint(1, 3),
            "simulated": True,
        }

    @tool(args_schema=FAQInput)
    async def query_faq(keyword: str) -> dict:
        """用本轮完整问题做混合检索和精排；keyword 仅用于核对用户原话。"""
        if keyword not in user_text:
            raise ToolArgumentsError("关键词必须是用户本轮原文中的连续片段。")
        matches = await repo.query_faq(user_text, category=category)
        return {"keyword": keyword, "category": category, "matches": matches}

    @tool(args_schema=TicketInput)
    async def create_ticket(description: str, ticket_type: str) -> dict:
        """仅在用户明确要求人工处理或创建工单时登记人工工单；FAQ 未命中不自动建单。"""
        return await repo.create_ticket(conversation_id, request_id, description, ticket_type)

    return [query_order, query_product, query_logistics, query_faq, create_ticket]


class ToolArgumentsError(ValueError):
    """表示通过 Schema 校验后仍违反业务参数约束。"""


class ToolRegistry:
    def __init__(
        self,
        tools: list[BaseTool],
        *,
        timeout: float = 5,
        retries: int = 1,
        retry_delay: float = 0.1,
    ):
        """注册唯一命名的工具，配置单次超时和最多额外重试次数。"""
        self.tools = {item.name: item for item in tools}
        if len(self.tools) != len(tools):
            raise ValueError("工具名称不能重复。")
        self.timeout, self.retries, self.retry_delay = timeout, retries, retry_delay

    async def execute(self, name: str, arguments: dict) -> dict:
        """执行工具并返回统一结果；仅超时和数据库瞬时故障可重试，取消信号直接传播。"""
        selected = self.tools.get(name)
        if not selected:
            return self.failure("unknown_tool", "模型申请了未注册的工具。", 0)
        try:
            validated = selected.get_input_schema().model_validate(arguments)
        except (ValidationError, TypeError):
            return self.failure("invalid_arguments", "工具参数不符合 Schema 约束。", 0)
        for attempt in range(1, self.retries + 2):
            try:
                async with asyncio.timeout(self.timeout):
                    data = await selected.ainvoke(validated.model_dump())
                return {"ok": True, "data": data, "attempts": attempt}
            except ToolArgumentsError as exc:
                return self.failure("invalid_arguments", str(exc), attempt)
            except (TimeoutError, OperationalError) as exc:
                if attempt <= self.retries:
                    await asyncio.sleep(self.retry_delay)
                    continue
                code = "timeout" if isinstance(exc, TimeoutError) else "unavailable"
                return self.failure(code, "工具暂时不可用，请稍后重试或联系人工。", attempt)
            except Exception:
                return self.failure(
                    "execution_failed", "工具执行失败，请稍后重试或联系人工。", attempt
                )

    @staticmethod
    def failure(code: str, message: str, attempts: int) -> dict:
        """构造不含异常原文、数据库地址或密钥的失败结果，供模型如实说明。"""
        return {"ok": False, "error": {"code": code, "message": message}, "attempts": attempts}
