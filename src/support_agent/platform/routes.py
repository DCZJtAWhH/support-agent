"""通用运行观测的 HTTP 查询入口。"""


def mount_routes(app):
    """挂载按最终意图归集真实模型用量的接口。"""

    @app.get("/api/costs")
    async def costs():
        """返回真实输入输出 token 和未知用量计数。"""
        return await app.state.read_costs()
