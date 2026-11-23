"""以整组已校准参数启动第七章服务，避免只缩模型窗口导致历史预算归零。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """显式覆盖本次进程的预算配置；模型、数据库和检索凭据仍从现有 .env 读取。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--offline", action="store_true", help="使用模型 HTTP 替身，不访问模型网络")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    mode = "demo" if args.demo else "default"
    environment = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        **{
            "MODEL_CONTEXT_WINDOW": "18000" if args.demo else "128000",
            "MAX_OUTPUT_TOKENS": "2000" if args.demo else "1024",
            "MAX_USER_INPUT_TOKENS": "2000",
            "MAX_AGENT_STEPS": "3" if args.demo else "6",
            "TOOL_RESULT_MAX_TOKENS": "1200",
            "RERANK_TOP_K": "5",
            "TOKEN_SAFETY_MARGIN": "250",
            "TOKEN_CHARS_PER_TOKEN": "2",
            "SYSTEM_TOOLS_TOKENS": "1000",
            "EVIDENCE_TOKENS_PER_RESULT": "100",
            "SUMMARY_INJECTION_TOKENS": "200",
            "HISTORY_TARGET_ROUNDS": "40",
            "STEADY_ROUND_TOKENS": "1000",
            "CHECKPOINT_PATH": str(ROOT / f"data/ch07-{mode}.sqlite"),
        },
    }
    environment["CH07_OFFLINE_MODE"] = mode
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "scripts.ch07_offline:create_offline_app"
            if args.offline
            else "support_agent.app:create_app",
            "--factory",
            "--app-dir",
            "." if args.offline else "src",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port or (8007 if args.demo else 8008)),
            "--workers",
            "1",
            "--log-config",
            "deploy/logging.json",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
