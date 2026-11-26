"""保留应用工厂入口，仅负责装配和路由挂载。"""

from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from support_agent.bootstrap import create_lifespan
from support_agent.commerce.routes import mount_routes as mount_commerce
from support_agent.conversation.routes import mount_routes as mount_conversation
from support_agent.dialogue.routes import mount_routes as mount_dialogue
from support_agent.evaluation.routes import mount_routes as mount_evaluation
from support_agent.improvement.routes import router as improvement_router
from support_agent.knowledge.routes import mount_routes as mount_knowledge
from support_agent.platform.capabilities import CapabilityError
from support_agent.platform.config import Settings
from support_agent.platform.logging import ensure_app_logging
from support_agent.platform.routes import mount_routes as mount_platform


def create_app(
    settings: Settings | None = None, upstream_client: httpx.AsyncClient | None = None
) -> FastAPI:
    """创建应用并挂载兼容入口；注入的上游客户端由调用方负责关闭。"""
    config = settings or Settings()
    ensure_app_logging()
    app = FastAPI(title="电商智能客服", lifespan=create_lifespan(config, upstream_client))

    @app.exception_handler(CapabilityError)
    async def capability_error(request: Request, exc: CapabilityError):
        """将公开业务失败原因映射为既有 HTTP 状态，内部错误不作为能力契约。"""
        status = {
            "conflict": 409,
            "not_found": 404,
            "input_budget": 413,
            "invalid_input": 422,
            "unavailable": 503,
        }[exc.reason]
        return JSONResponse(status_code=status, content={"detail": exc.message})

    web_directory = Path(__file__).with_name("web")
    app.mount("/static", StaticFiles(directory=web_directory), name="static")
    for mount in (
        mount_conversation,
        mount_dialogue,
        mount_commerce,
        mount_knowledge,
        mount_evaluation,
    ):
        mount(app, web_directory)
    mount_platform(app)
    app.include_router(improvement_router)
    return app
