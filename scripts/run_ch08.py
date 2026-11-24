"""启动第八章客服服务；两个 MCP Server 应在另外两个终端独立启动。"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit


def main():
    """选择真实模型或离线模型演示，维持相同客服图与 MCP 客户端实现。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="仅用模型替身，MCP 和数据库实际运行")
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": "src", "PYTHONIOENCODING": "utf-8"}
    if not args.offline:
        sys.path.insert(0, str(root / "src"))
        from support_agent.config import Settings

        settings = Settings()
        if (
            settings.model_provider == "deepseek"
            or urlsplit(settings.openai_base_url).hostname == "api.deepseek.com"
        ):
            environment.setdefault(
                "MODEL_TEST_BUDGET_PATH", str(root / "data/ch08-real-budget.sqlite")
            )
    environment.setdefault(
        "MCP_SERVERS",
        json.dumps(
            {
                "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:8101/mcp"},
                "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:8102/mcp"},
            }
        ),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "scripts.ch08_offline:create_offline_app"
            if args.offline
            else "support_agent.app:create_app",
            "--factory",
            "--app-dir",
            "." if args.offline else "src",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--workers",
            "1",
        ],
        cwd=root,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
