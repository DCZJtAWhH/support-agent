"""conversation 的 HTTP 入口；协议转换与业务能力分离。"""

from uuid import UUID

from fastapi import HTTPException


def mount_routes(app, web_directory):
    """挂载既有路径和响应契约，页面资源路径由启动装配传入。"""

    @app.get("/api/conversations")
    async def conversations():
        """只读列出演示用户全部会话，按新在前排序，不接受客户端指定其他用户。"""
        return await app.state.conversations.list_conversations()

    @app.get("/api/conversations/{conversation_id}/messages")
    async def messages(conversation_id: UUID):
        """回载本会话用户与客服原文，不暴露工作流审计或工具结果，未知会话返回404。"""
        audit = await app.state.conversations.read_conversation(str(conversation_id))
        if audit is None:
            raise HTTPException(404, detail="会话不存在。")
        return [
            m
            for m in audit["messages"]
            if m["role"] in {"user", "assistant"} and not m["tool_calls"]
        ]

    @app.get("/conversations/{conversation_id}")
    async def conversation_history(conversation_id: UUID):
        """读取演示会话的持久消息和工单审计；尚未接入用户身份认证。"""
        result = await app.state.read_audit(str(conversation_id))
        if result is None:
            raise HTTPException(404, detail="会话不存在。")
        return result
