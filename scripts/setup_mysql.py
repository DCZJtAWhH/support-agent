"""复用本机 Docker MySQL，为本项目建立独立数据库和仅访问该库的演示账号。"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    """从容器环境读取管理凭据但不输出，并仅更新本地 .env 的 DATABASE_URL。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="mysql-container")
    parser.add_argument("--port", type=int, default=3306)
    args = parser.parse_args()
    try:
        result = subprocess.run(
            ["docker", "inspect", args.container],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        container = json.loads(result.stdout)[0]
        environment = dict(item.split("=", 1) for item in container["Config"]["Env"] if "=" in item)
        password = environment.get("MYSQL_ROOT_PASSWORD", "")
        sql = """
CREATE DATABASE IF NOT EXISTS support_agent_ch02 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'support_agent_ch02'@'%' IDENTIFIED BY 'support_agent_demo';
GRANT ALL PRIVILEGES ON support_agent_ch02.* TO 'support_agent_ch02'@'%';
"""
        subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "--env",
                f"MYSQL_PWD={password}",
                args.container,
                "mysql",
                "-uroot",
                "--default-character-set=utf8mb4",
            ],
            input=sql,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (subprocess.SubprocessError, KeyError, ValueError):
        raise SystemExit(
            "无法初始化 MySQL；请检查容器名称和管理凭据配置。未输出任何管理凭据。"
        ) from None
    url = (
        "DATABASE_URL=mysql+aiomysql://support_agent_ch02:support_agent_demo"
        f"@127.0.0.1:{args.port}/support_agent_ch02?charset=utf8mb4"
    )
    path = Path(__file__).resolve().parents[1] / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    lines = [line for line in lines if not line.startswith("DATABASE_URL=")]
    path.write_text("\n".join([*lines, url, ""]), encoding="utf-8")
    print("项目数据库和演示账号已就绪，DATABASE_URL 已写入本地 .env。")


if __name__ == "__main__":
    main()
