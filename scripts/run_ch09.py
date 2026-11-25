"""启动第九章真实客服、Langfuse 与飞轮后台演示，复用本地数据库和向量库。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    """显式使用同一模型的非思考模式，避免小输出预算被思考耗尽；不切换提供方。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    environment = {**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8"}
    environment.setdefault("MODEL_THINKING", "disabled")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "support_agent.app:create_app",
            "--factory",
            "--app-dir",
            "src",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--workers",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
