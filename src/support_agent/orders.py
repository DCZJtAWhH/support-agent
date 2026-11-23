"""复用现有订单工具的演示边界，为查询与聊天选择器提供一致的数据。"""

import re

DEMO_ORDERS = (
    {
        "order_id": "1001",
        "product_name": "BX-500 保温杯",
        "product_id": "BX-500",
        "status": "已签收",
        "days_since_received": 3,
        "total": 99,
        "currency": "CNY",
        "category": "普通商品",
        "simulated": True,
    },
    {
        "order_id": "1002",
        "product_name": "FD-200 智能喂食器",
        "product_id": "FD-200",
        "status": "已签收",
        "days_since_received": 12,
        "total": 299,
        "currency": "CNY",
        "category": "电子类商品",
        "simulated": True,
    },
    {
        "order_id": "1003",
        "product_name": "定制刻字保温杯",
        "product_id": "CUSTOM-01",
        "status": "已签收",
        "days_since_received": 2,
        "total": 199,
        "currency": "CNY",
        "category": "定制类",
        "simulated": True,
    },
)


def list_orders():
    """返回当前演示顾客可选订单的独立副本，不代表真实电商或身份认证。"""
    return [dict(order) for order in DEMO_ORDERS]


def get_order(order_id):
    """按明确编号读取相同演示订单；不存在时返回空，禁止编造订单事实。"""
    return next((order for order in list_orders() if order["order_id"] == order_id), None)


def explicit_order_ids(text):
    """识别带订单标识的单号与并列编号，避免把多订单静默当成第一笔。"""
    prefix = r"(?:订单(?:号|编号)?\s*[:：#]?\s*|order(?:_id)?\s*[:：#]?\s*)"
    identifiers = []
    for match in re.finditer(prefix + r"([A-Za-z0-9_-]+)", text, flags=re.IGNORECASE):
        identifiers.append(match[1])
        tail = text[match.end() :]
        while following := re.match(
            r"\s*(?:和|与|及|、|,|，|/)\s*(?:" + prefix + r")?([A-Za-z0-9_-]+)",
            tail,
            flags=re.IGNORECASE,
        ):
            identifiers.append(following[1])
            tail = tail[following.end() :]
    return list(dict.fromkeys(identifiers))
