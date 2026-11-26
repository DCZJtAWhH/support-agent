from sqlalchemy import select

from .models import FAQ

FAQ_SEEDS = [
    (
        "退货政策是什么",
        "签收后7天内，商品完好且不影响二次销售可申请无理由退货。质量问题请联系人工核实。",
        "售后",
    ),
    (
        "运费标准是什么",
        "普通商品订单满99元免运费，未满99元收取8元运费；偏远地区需人工确认。",
        "配送",
    ),
    ("如何申请换货", "请提供订单号和商品问题，客服登记后由人工确认换货条件。", "售后"),
]


async def seed_faq(database):
    """幂等补充旧 FAQ 演示来源，不覆盖已有知识。"""
    async with database.sessions.begin() as session:
        for question, answer, category in FAQ_SEEDS:
            if not await session.scalar(select(FAQ).where(FAQ.question == question)):
                session.add(FAQ(question=question, answer=answer, category=category))
