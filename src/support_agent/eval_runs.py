"""持久化完整在线评估的逐轮指标，供定期回归与趋势对照。"""

from sqlalchemy import select

from support_agent.evaluation import summarize
from support_agent.models import EvalRun
from support_agent.retrieval import STRATEGIES


class EvaluationRuns:
    def __init__(self, database):
        """绑定与客服共享的数据库，报告文件继续保留完整逐题证据。"""
        self.database = database

    async def record(self, run_id, trigger_type, report):
        """幂等记录一轮真实评估，回放报告不能伪装成在线效果。"""
        if report.get("mode") != "live" or not report.get("rows"):
            raise ValueError("只有完整真实评估可以写入指标趋势。")
        if trigger_type not in {"manual", "scheduled", "cli"}:
            raise ValueError("未知评估触发方式。")
        metadata = report.get("metadata", {})
        metrics = report.get("strategies") or {
            strategy: summarize([r for r in report["rows"] if r["strategy"] == strategy])
            for strategy in STRATEGIES
        }
        async with self.database.sessions.begin() as session:
            if await session.scalar(select(EvalRun).where(EvalRun.run_id == run_id)):
                return
            session.add(
                EvalRun(
                    run_id=run_id,
                    trigger_type=trigger_type,
                    dataset_count=len({r["id"] for r in report["rows"]}),
                    metrics=metrics,
                    dataset_hash=metadata.get("dataset_hash", ""),
                    corpus_hash=metadata.get("corpus_hash", ""),
                )
            )

    async def history(self):
        """按时间与主键稳定升序返回每轮指标及版本，空分母保留 null。"""
        async with self.database.sessions() as session:
            rows = await session.scalars(select(EvalRun).order_by(EvalRun.created_at, EvalRun.id))
            return [
                {
                    "id": row.id,
                    "run_id": row.run_id,
                    "trigger_type": row.trigger_type,
                    "dataset_count": row.dataset_count,
                    "metrics": row.metrics,
                    "dataset_hash": row.dataset_hash,
                    "corpus_hash": row.corpus_hash,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]
