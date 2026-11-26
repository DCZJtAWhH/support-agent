"""dialogue 的 HTTP 入口；协议转换与业务能力分离。"""

from fastapi.responses import FileResponse, StreamingResponse

from .schemas import ChatRequest, RefundSubmission, TicketConfirmation


def mount_routes(app, web_directory):
    """挂载既有路径和响应契约，页面资源路径由启动装配传入。"""

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def chat_page():
        """提供同源客服聊天页，静态资源路径基于模块位置，不依赖启动目录。"""
        return FileResponse(web_directory / "index.html", headers={"Cache-Control": "no-cache"})

    @app.post(
        "/chat",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def chat(body: ChatRequest):
        """从现有入口执行确定性 Workflow、ReAct 迭代和最终 SSE 回复。"""
        return await app.state.chat.respond(body)

    @app.post("/actions/ticket")
    async def confirm_ticket(body: TicketConfirmation):
        """用户点击并确认建工单后，调用既有工具并返回幂等工单结果。"""
        return await app.state.chat.confirm_ticket(body)

    @app.post("/actions/refund")
    async def submit_refund(body: RefundSubmission):
        """接收用户在固定原因表单中的显式提交，返回待核实的申请凭据。"""
        return await app.state.chat.submit_refund(body)
