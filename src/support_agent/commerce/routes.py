"""commerce 的 HTTP 入口；协议转换与业务能力分离。"""

from fastapi import HTTPException

from support_agent.platform.context import ContextBudgetExceeded
from support_agent.platform.errors import model_error

from .schemas import AfterSalesRequest, ExtractionRequest


def mount_routes(app, web_directory):
    """挂载既有路径和响应契约，页面资源路径由启动装配传入。"""

    @app.post("/after-sales/extract", response_model=AfterSalesRequest)
    async def extract(body: ExtractionRequest):
        """调用售后提取能力，预算与模型故障分别映射为原有 413 和 502。"""
        try:
            return await app.state.extraction.extract(body.description)
        except ContextBudgetExceeded as exc:
            raise HTTPException(413, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                502,
                detail=model_error(
                    exc,
                    structured_output_method=app.state.extraction.config.structured_output_method,
                ),
            ) from exc
