"""保留业务失败原因，传输状态由入口映射。"""


class CapabilityError(Exception):
    def __init__(self, reason: str, message: str):
        """记录稳定原因和可公开提示，不携带依赖异常原文。"""
        super().__init__(message)
        self.reason, self.message = reason, message
