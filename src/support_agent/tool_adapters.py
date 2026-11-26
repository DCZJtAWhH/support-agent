"""注册 LangChain 业务工具，集中校验参数并处理超时、重试与脱敏错误。"""

import logging

from langchain_core.tools import BaseTool, tool

from support_agent.commerce.orders import get_order, get_product
from support_agent.commerce.tool_inputs import OrderInput, ProductInput, TicketInput
from support_agent.knowledge.tool_inputs import FAQInput
from support_agent.tooling import ToolArgumentsError, ToolNotFoundError

logger = logging.getLogger(__name__)


def build_tools(
    commerce, knowledge, conversation_id: str, request_id: str, user_text: str, *, category=None
) -> list[BaseTool]:
    """为当前轮次绑定可信会话和请求标识；模型不能自行指定身份或工单去重键。"""

    @tool(args_schema=OrderInput)
    async def query_order(order_id: str) -> dict:
        """按明确编号读取与选择器一致的演示订单，不访问真实电商系统。"""
        order = get_order(order_id)
        if order is None:
            raise ToolNotFoundError("未找到该演示订单，请重新选择订单。")
        return order

    @tool(args_schema=ProductInput)
    async def query_product(product_id: str) -> dict:
        """按商品编号或名称查询商品价格和库存。返回随机演示数据，不访问真实电商系统。"""
        return get_product(product_id)

    @tool(args_schema=FAQInput)
    async def query_faq(keyword: str) -> dict:
        """用本轮完整问题做混合检索和精排；keyword 仅用于核对用户原话。"""
        if keyword not in user_text:
            raise ToolArgumentsError("关键词必须是用户本轮原文中的连续片段。")
        matches = await knowledge.query_faq(user_text, category=category)
        return {"keyword": keyword, "category": category, "matches": matches}

    @tool(args_schema=TicketInput)
    async def create_ticket(description: str, ticket_type: str) -> dict:
        """顾客明确请求且问题齐全时申请工单预览；前端确认后才登记，建单不代表问题解决。"""
        return await commerce.create_ticket(
            conversation_id, request_id, description, ticket_type, confirmed=True
        )

    return [query_order, query_product, query_faq, create_ticket]
