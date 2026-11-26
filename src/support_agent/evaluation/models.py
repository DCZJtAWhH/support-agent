from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from support_agent.platform.models import Base, utc_now


class EvalRun(Base):
    """一行记录一轮真实评估的触发方式、题数、指标与时间及数据版本。"""

    __tablename__ = "eval_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    trigger_type: Mapped[str] = mapped_column(String(16))
    dataset_count: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict] = mapped_column(JSON)
    dataset_hash: Mapped[str] = mapped_column(String(64), default="")
    corpus_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
