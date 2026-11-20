"""第三章知识库 demo 命令入口；配置来自本地 .env，输出不包含凭据。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_agent.config import Settings  # noqa: E402
from support_agent.database import Database  # noqa: E402
from support_agent.knowledge_jobs import (  # noqa: E402
    import_documents,
    import_faq,
    mine_directory,
    schedule,
)
from support_agent.knowledge_mining import KnowledgeLLM, MiningPipeline  # noqa: E402
from support_agent.knowledge_store import (  # noqa: E402
    KnowledgeStore,
    SimulatedInterruption,
    synchronize,
)
from support_agent.vectors import BGEEmbeddings, KnowledgeRetriever, MilvusVectors  # noqa: E402


async def main(args):
    """执行建库、检索、抽取或常驻调度，退出时关闭数据库和 Milvus 连接。"""
    config = Settings()
    database = Database(config.database_url.get_secret_value())
    store = KnowledgeStore(database)
    embeddings = BGEEmbeddings(config.siliconflow_api_key.get_secret_value())
    vectors = MilvusVectors(config.milvus_uri, config.milvus_collection)
    try:
        await database.initialize(seed=True)
        if args.command == "build":
            result = {"faq": await import_faq(store)}
            result["document_chunks"] = await import_documents(store, Path(args.docs))
            if not args.no_sync:
                result["synchronized"] = await synchronize(
                    store,
                    embeddings,
                    vectors,
                    interrupt_after_upsert=args.interrupt_after_upsert,
                )
        elif args.command == "sync":
            if args.rebuild:
                await store.requeue()
            result = {
                "synchronized": await synchronize(
                    store,
                    embeddings,
                    vectors,
                    interrupt_after_upsert=args.interrupt_after_upsert,
                )
            }
        elif args.command == "query":
            result = {
                "keyword": args.question,
                "matches": await KnowledgeRetriever(
                    store,
                    embeddings,
                    vectors,
                    top_k=config.knowledge_top_k,
                    min_score=config.knowledge_min_score,
                ).search(args.question),
            }
        elif args.command == "stats":
            result = await store.stats()
        else:
            with httpx.Client() as sync_client:
                async with httpx.AsyncClient() as client:
                    pipeline = MiningPipeline(
                        store,
                        KnowledgeLLM(config, client, sync_client),
                        embeddings,
                    )
                    if args.command == "staging":
                        result = await pipeline.staging()
                    elif args.command == "mine":
                        result = await mine_directory(
                            pipeline,
                            Path(args.history),
                            retry_conflicts=args.retry_conflicts,
                        )
                        result["synchronized"] = await synchronize(store, embeddings, vectors)
                    else:
                        print("知识挖掘任务已启动：北京时间每日02:00；Ctrl+C退出。", flush=True)
                        await schedule(pipeline, Path(args.history), vectors, run_now=args.run_now)
                        return
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await vectors.close()
        await database.close()


def parse_args():
    """解析明确的 demo 操作；故意中断开关仅影响向量写后、状态回填前的窗口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["build", "sync", "query", "stats", "mine", "staging", "schedule"]
    )
    parser.add_argument("question", nargs="?", default="邮费是多少")
    parser.add_argument("--docs", default=str(ROOT / "examples/knowledge/docs"))
    parser.add_argument("--history", default=str(ROOT / "examples/knowledge/history"))
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--interrupt-after-upsert", action="store_true")
    parser.add_argument("--run-now", action="store_true")
    parser.add_argument("--retry-conflicts", action="store_true", help="mine 时重新核对留存冲突")
    parser.add_argument("--rebuild", action="store_true", help="sync 时将生效知识重新标为待向量化")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        print("任务已停止；已提交的原文、暂存和批次检查点均保留。")
    except SimulatedInterruption as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(
            f"任务失败（{type(exc).__name__}），检查配置或服务状态后重跑；凭据及上游原文未输出。",
            file=sys.stderr,
        )
        sys.exit(1)
