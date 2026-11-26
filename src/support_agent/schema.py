"""启动装配的数据库登记；平台不反向导入功能模块。"""

from support_agent.commerce import models as commerce_models  # noqa: F401
from support_agent.conversation.summaries import migrate_context_schema
from support_agent.evaluation import models as evaluation_models  # noqa: F401
from support_agent.improvement.migrations import migrate_flywheel_schema
from support_agent.knowledge.seeds import seed_faq
from support_agent.knowledge.store import migrate_knowledge_schema
from support_agent.platform.database import schema_migrations, seed_initializers
from support_agent.tooling import models as tooling_models  # noqa: F401


def register_schema():
    """统一登记共享元数据和兼容迁移；保留所有物理表及历史主键。"""
    schema_migrations[:] = [
        migrate_knowledge_schema,
        migrate_context_schema,
        migrate_flywheel_schema,
    ]
    seed_initializers[:] = [seed_faq]
