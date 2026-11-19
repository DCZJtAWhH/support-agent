"""管理 SSE 响应发送、客户端断开监听与会话清理回调。"""

import asyncio
from collections.abc import AsyncIterator, Callable

from fastapi.responses import StreamingResponse


class ChatStreamResponse(StreamingResponse):
    """持续监听断流，包括上游尚未产生下个增量时；成功发送后才提交历史。"""

    def __init__(
        self,
        content: AsyncIterator[str],
        on_complete: Callable[[], None],
        on_close: Callable[[], None],
    ):
        """配置 SSE 响应头，并保存发送完成和响应关闭时调用的回调。

        on_complete 仅在完整发送后执行，on_close 在发送结束或中断后的清理阶段执行。
        """
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self.on_complete = on_complete
        self.on_close = on_close

    async def __call__(self, scope, receive, send):
        """并行发送事件流与监听断连，结束时取消任务、关闭迭代器并调用清理回调。

        scope、receive 和 send 为 ASGI 调用参数；发送中断时跳过完成回调。
        """
        async def send_stream():
            """完整发送响应及结束消息后调用完成回调，使会话历史可以提交。"""
            await self.stream_response(send)
            # 最后一个 ASGI body 已发送；之后的 disconnect 可能只是连接正常结束。
            self.on_complete()

        stream = asyncio.create_task(send_stream())
        disconnect = asyncio.create_task(self.listen_for_disconnect(receive))
        try:
            done, _ = await asyncio.wait((stream, disconnect), return_when=asyncio.FIRST_COMPLETED)
            if stream in done:
                await stream
        except OSError:
            # 客户端在发送过程中关闭连接，不提交本轮。
            pass
        finally:
            stream.cancel()
            disconnect.cancel()
            await asyncio.gather(stream, disconnect, return_exceptions=True)
            try:
                await self.body_iterator.aclose()
            finally:
                self.on_close()
