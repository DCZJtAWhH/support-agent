"""客服模块化单体；导入时仅登记元数据，不连接数据库或模型。"""

from support_agent.schema import register_schema

register_schema()
