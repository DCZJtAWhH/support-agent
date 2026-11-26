import re


def explicit_ticket_request(text):
    """仅从顾客原话识别明确建单请求，否定、假设和规则讨论一律不授予写意图。"""
    if re.search(r"不要|不用|无需|别建|不建|取消|先不|暂不|如果|假如|假设|例如|比如|规则", text):
        return False
    return bool(
        re.search(
            r"(?:创建|建立|建|开|登记|提交)(?:一[个张]|[个张])?(?:人工|售后)?"
            r"(?:退款|退货退款|换货|维修|投诉)?工单",
            text,
        )
    )


def has_problem_description(text):
    """剔除单纯建单话术后检查是否仍有问题内容，防止把申请本身当作故障描述。"""
    remaining = re.sub(
        r"(?:创建|建立|建|开|登记|提交)(?:一[个张]|[个张])?(?:人工|售后)?"
        r"(?:退款|退货退款|换货|维修|投诉)?工单",
        "",
        text,
    )
    remaining = re.sub(
        r"帮我|替我|为我|我要|我想|麻烦|请|给我|谢谢|一下|吧|。|，|！|？|\s", "", remaining
    )
    return len(remaining) >= 2


def valid_ticket_description(arguments, sources):
    """校验工单描述来自顾客原话且包含具体问题，不把建单话术当作故障。"""
    description = arguments.get("description", "")
    return has_problem_description(description) and any(description in source for source in sources)
