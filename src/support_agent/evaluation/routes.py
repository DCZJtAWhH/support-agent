"""evaluation 的 HTTP 入口；协议转换与业务能力分离。"""

from fastapi.responses import FileResponse


def mount_routes(app, web_directory):
    """挂载既有路径和响应契约，页面资源路径由启动装配传入。"""

    @app.get("/evaluations", response_class=FileResponse, include_in_schema=False)
    async def evaluation_page():
        """提供独立的RAG评估结果页，让用户保留原聊天并查看评估证据。"""
        return FileResponse(
            web_directory / "evaluation.html", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/api/evaluations")
    async def evaluation_report(query_type: str = "", difficulty: str = ""):
        """返回真实报告、联合分桶指标、逐题证据与当前知识库版本信息。"""
        return await app.state.evaluations.overview(query_type=query_type, difficulty=difficulty)

    @app.get("/api/evaluations/status")
    async def evaluation_status():
        """轮询评估任务进度，不传输重复的检索原文与回答明细。"""
        return await app.state.evaluations.status()

    @app.post("/api/evaluations/run", status_code=202)
    async def start_evaluation():
        """针对当前知识库和固定标注集启动完整四策略评估，不接受任意命令或路径。"""
        return await app.state.evaluations.start()

    @app.get("/api/evaluations/runs")
    async def evaluation_runs():
        """读取完整真实评估的历史指标与数据版本。"""
        return await app.state.evaluation_runs.history()
