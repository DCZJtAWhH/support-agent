"""按输入 token 预算裁剪客服对话历史，保留系统提示与当前消息。"""

import logging
from pathlib import Path


def ensure_app_logging():
    """为业务日志增加 UTF-8 文件输出，让所有启动方式都能直接检查上下文原文。"""
    path = Path("log/app.log").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("support_agent")
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(path) for h in logger.handlers
    ):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
