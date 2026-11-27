"""为评估界面读取真实报告、复用指标聚合并核对知识与标注版本。"""

import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from support_agent.evaluation.metrics import summarize
from support_agent.knowledge import STRATEGIES, CorpusVersion, digest
from support_agent.platform.capabilities import CapabilityError

from .runs import EvaluationRuns

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_json(path: Path) -> dict[str, Any] | None:
    """用 UTF-8 读取已完成的报告或状态，缺失文件表示尚未生成。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def write_json(path: Path, value: dict):
    """原子替换UTF-8状态文件，防止页面读取到半份JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


class EvaluationDashboard:
    def __init__(
        self,
        read_corpus: Callable[[], Awaitable[CorpusVersion]],
        *,
        dataset_path=None,
        runs_directory=None,
        script_path=None,
        runs: EvaluationRuns,
    ):
        """绑定权威原文与本地运行报告路径，构造时不读取文件或触发模型调用。"""
        self.read_corpus = read_corpus
        self.runs = runs
        self.dataset_path = dataset_path or PROJECT_ROOT / "evals/retrieval.jsonl"
        self.runs_directory = runs_directory or PROJECT_ROOT / "data/rag-evaluations"
        self.script_path = script_path or PROJECT_ROOT / "scripts/eval_retrieval.py"
        self.lock = asyncio.Lock()
        self.job: dict[str, Any] | None = None
        self.task: asyncio.Task | None = None
        self.initialized = False
        self.scheduler = None

    async def _initialize(self):
        """首次访问时恢复最近任务记录，不因构造测试应用而读写真实运行状态。"""
        async with self.lock:
            if not self.initialized:
                self.job = await asyncio.to_thread(read_json, self.runs_directory / "job.json")
                if self.job and self.job.get("status") == "running":
                    self.job.update(
                        status="interrupted", message="服务重启导致评估中断，请重新运行。"
                    )
                    await self._save_job()
                self.initialized = True

    async def _save_job(self):
        """记录真实进度供刷新页面后恢复，不记录模型凭据或上游错误正文。"""
        if self.job is not None:
            await asyncio.to_thread(write_json, self.runs_directory / "job.json", dict(self.job))

    async def status(self):
        """返回任务快照，轮询时不重复发送全部逐题原文。"""
        await self._initialize()
        return {"job": dict(self.job) if self.job else None}

    async def _report(self):
        """只读取最新指针发布的报告；指针或报告缺失时返回空，不使用预置报告。"""
        pointer = await asyncio.to_thread(read_json, self.runs_directory / "latest.json")
        if pointer:
            run_id = str(UUID(pointer["id"]))
            report = await asyncio.to_thread(
                read_json, self.runs_directory / run_id / "report.json"
            )
            return (run_id, report) if report is not None else (None, None)
        return None, None

    async def corpus(self):
        """获取当前有效知识的稳定身份，和CLI报告使用相同的摘要口径。"""
        return await self.read_corpus()

    async def dataset(self):
        """读取标注集摘要及题目数，不把新上传知识自动视为已有评估覆盖。"""
        text = await asyncio.to_thread(self.dataset_path.read_text, encoding="utf-8")
        cases = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {"hash": digest(text), "case_count": len(cases)}

    async def overview(self, *, query_type="", difficulty=""):
        """读取报告并联合筛选，指标复用后端唯一实现，历史快照不冒充当前结果。"""
        job = (await self.status())["job"]
        report_id, report = await self._report()
        corpus, dataset = await asyncio.gather(self.corpus(), self.dataset())
        all_rows = report["rows"] if report else []
        query_types = sorted({r["query_type"] for r in all_rows})
        difficulties = sorted({r["difficulty"] for r in all_rows})
        if report and (
            (query_type and query_type not in query_types)
            or (difficulty and difficulty not in difficulties)
        ):
            raise CapabilityError("invalid_input", "评估筛选条件不存在。")
        rows = [
            r
            for r in all_rows
            if (not query_type or r["query_type"] == query_type)
            and (not difficulty or r["difficulty"] == difficulty)
        ]
        metadata = report.get("metadata", {}) if report else {}
        return {
            "report": {
                "id": report_id,
                "mode": report["mode"],
                "metadata": metadata,
                "case_count": len({r["id"] for r in all_rows}),
                "row_count": len(all_rows),
            }
            if report
            else None,
            "rows": rows,
            "strategies": {
                s: summarize([r for r in rows if r["strategy"] == s]) for s in STRATEGIES
            },
            "case_count": len({r["id"] for r in rows}),
            "filters": {"query_types": query_types, "difficulties": difficulties},
            "corpus": corpus,
            "dataset": dataset,
            "stale": bool(
                report
                and (metadata.get("corpus_hash") != corpus["hash"] or corpus["pending_count"])
            ),
            "dataset_changed": bool(report and metadata.get("dataset_hash") != dataset["hash"]),
            "job": job,
        }

    async def start(self, trigger_type="manual"):
        """在单工作进程启动真实CLI评估，重复申请返回409且保留正在运行的任务。"""
        await self._initialize()
        async with self.lock:
            if self.task is not None and not self.task.done():
                raise CapabilityError("conflict", "已有评估正在运行，请等待当前任务结束。")
            corpus, dataset = await asyncio.gather(self.corpus(), self.dataset())
            if corpus["pending_count"]:
                raise CapabilityError("conflict", "知识库还有待同步的内容，请完成同步后重新评估。")
            if not dataset["case_count"]:
                raise CapabilityError("invalid_input", "评估集没有标注题目，无法开始评估。")
            self.job = {
                "id": str(uuid4()),
                "status": "running",
                "completed": 0,
                "total": dataset["case_count"] * len(STRATEGIES),
                "case_count": dataset["case_count"],
                "started_at": datetime.now(UTC).isoformat(),
                "message": "正在准备题目与模型，评估会调用真实模型服务。",
                "corpus_hash": corpus["hash"],
                "dataset_hash": dataset["hash"],
                "trigger_type": trigger_type,
            }
            await self._save_job()
            self.task = asyncio.create_task(self._execute())
            return {"job": dict(self.job)}

    def start_scheduler(self, interval):
        """按配置间隔运行既有评估流水线，首次等一个周期，零值表示禁用。"""
        if interval > 0 and self.scheduler is None:
            self.scheduler = asyncio.create_task(self._periodic(interval))

    async def _periodic(self, interval):
        """定期启动完整评估，已有运行则跳过本次，其他错误脱敏记录后下轮重试。"""
        while True:
            await asyncio.sleep(interval)
            try:
                await self.start(trigger_type="scheduled")
            except CapabilityError as exc:
                logging.getLogger(__name__).warning("定期评估未启动，状态码=%s", exc.reason)
            except Exception as exc:
                logging.getLogger(__name__).error("定期评估启动失败，类型=%s", type(exc).__name__)

    async def _execute(self):
        """执行自身子进程并消费真实结果行，成功后原子发布新的报告指针。"""
        job = self.job
        if job is None:
            raise RuntimeError("评估任务尚未初始化。")
        process = None
        try:
            output = self.runs_directory / job["id"] / "report.json"
            await asyncio.to_thread(output.parent.mkdir, parents=True, exist_ok=True)
            environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self.script_path),
                "--dataset",
                str(self.dataset_path),
                "--output",
                str(output),
                "--concurrency",
                "4",
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert process.stdout is not None
            seen = set()
            budget_exceeded = False
            while line := await process.stdout.readline():
                decoded = line.decode("utf-8", errors="replace").strip()
                # 仅写本地诊断文件；网页继续使用脱敏状态，不暴露提供方错误正文。
                with (output.parent / "process.log").open("a", encoding="utf-8") as diagnostic:
                    diagnostic.write(decoded + "\n")
                budget_exceeded = budget_exceeded or "ContextBudgetExceeded" in decoded
                match = re.fullmatch(
                    r"(.+?) / (dense|bm25|hybrid|hybrid_rerank): (answered|refused)",
                    decoded,
                )
                if match:
                    seen.add((match[1], match[2]))
                    job["completed"] = min(len(seen), job["total"])
                    job["message"] = f"已完成 {match[1]} · {match[2]}"
                    await self._save_job()
            if await process.wait() != 0:
                if budget_exceeded:
                    job.update(
                        status="failed",
                        message="评估输入超过当前输入预算，请检查知识正文长度和模型窗口配置后重试；上一份报告已保留。",
                    )
                    return
                raise RuntimeError("评估命令失败")
            report = await asyncio.to_thread(read_json, output)
            if (
                not report
                or report.get("mode") != "live"
                or len(report.get("rows", [])) != job["total"]
                or report.get("metadata", {}).get("corpus_hash") != job["corpus_hash"]
                or report.get("metadata", {}).get("dataset_hash") != job["dataset_hash"]
            ):
                raise RuntimeError("评估报告不完整")
            corpus, dataset = await asyncio.gather(self.corpus(), self.dataset())
            if corpus["hash"] != job["corpus_hash"] or corpus["pending_count"]:
                job.update(
                    status="failed", message="评估期间知识库发生变化，本次结果未发布，请重新运行。"
                )
                return
            if dataset["hash"] != job["dataset_hash"]:
                job.update(
                    status="failed", message="评估期间标注集发生变化，本次结果未发布，请重新运行。"
                )
                return
            await self.runs.record(
                job["id"],
                job.get("trigger_type", "manual"),
                report,
            )
            await asyncio.to_thread(
                write_json, self.runs_directory / "latest.json", {"id": job["id"]}
            )
            job.update(
                status="completed", completed=job["total"], message="评估已完成，报告已更新。"
            )
        except asyncio.CancelledError:
            job.update(status="interrupted", message="服务已关闭，评估中断；请重新运行。")
        except Exception:
            job.update(
                status="failed",
                message="评估未完成，请检查模型、知识库和标注集后重试；上一份报告已保留。",
            )
        finally:
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                await process.communicate()
            job["finished_at"] = datetime.now(UTC).isoformat()
            await self._save_job()

    async def close(self):
        """停止自身评估任务和子进程，避免服务重启后仍消耗模型请求。"""
        if self.scheduler is not None:
            self.scheduler.cancel()
            await asyncio.gather(self.scheduler, return_exceptions=True)
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
