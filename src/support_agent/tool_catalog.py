"""维护进程级工具表；每轮绑定可信会话并从 MCP Server 重新发现工具。"""

import asyncio
import json
import logging
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from support_agent.tools import ToolRegistry, build_tools

logger = logging.getLogger(__name__)


class ToolCatalog:
    def __init__(self, *, servers=None, policy_path=None, timeout=5, retries=1):
        """登记启动时内置工具；Server 地址可信，外部工具权限只从本地文件读取。"""
        self.entries = ToolRegistry(build_tools(None, "", "", ""))
        self.builtin_names = set(self.entries.tools)
        self.servers = servers or {}
        self.client = MultiServerMCPClient(self.servers)
        self.policy_path = Path(policy_path) if policy_path else None
        self.timeout, self.retries = timeout, retries
        self.discovery_errors = {}
        self.lock = asyncio.Lock()

    def register(self, tool, *, permission="read", result_fields=None):
        """热登记本系统可信插件，不修改 Agent 核心代码；写插件仍受引擎固定规则限制。"""
        self.entries.register(tool, permission=permission, result_fields=result_fields)

    def read_policy(self):
        """每次绑定重新读取本地授权；文件缺失或损坏时外部工具全部拒绝。"""
        if self.policy_path:
            try:
                data = json.loads(self.policy_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError):
                logger.warning("MCP 权限文件不可读，外部工具默认拒绝")
        return {}

    async def refresh(self):
        """现问现取清单；逐 Server 隔离发现故障并替换旧快照，不缓存失效工具。"""
        if not self.servers:
            return
        async with self.lock:

            async def discover(name):
                """限定单个 Server 的发现耗时，故障仅影响该 Server 的工具。"""
                try:
                    async with asyncio.timeout(self.timeout):
                        return name, await self.client.get_tools(server_name=name)
                except Exception:
                    return name, None

            discovered = await asyncio.gather(*(discover(name) for name in self.servers))
            fresh = ToolRegistry([])
            for name, item in self.entries.tools.items():
                if self.entries.sources[name] == "builtin":
                    fresh.register(
                        item,
                        permission=self.entries.permissions[name],
                        result_fields=self.entries.result_fields[name],
                    )
            self.discovery_errors = {}
            for server, items in discovered:
                if items is None:
                    self.discovery_errors[server] = "工具清单发现失败，暂时无法查询该系统。"
                    continue
                for item in items:
                    try:
                        fresh.register(item, source=f"mcp:{server}", permission="deny")
                    except (TypeError, ValueError):
                        logger.warning(
                            "拒绝 MCP 无效或重名工具 server=%s tool=%s", server, item.name
                        )
            self.entries = fresh

    def bind(self, repo, conversation_id, request_id, user_text, *, category=None):
        """生成调用快照；内置工具绑定可信身份，MCP 权限由本地按来源和名称精确授权。"""
        registry = ToolRegistry(
            [],
            repo=repo,
            conversation_id=conversation_id,
            timeout=self.timeout,
            retries=self.retries,
        )
        builtin = {
            t.name: t
            for t in build_tools(
                repo,
                conversation_id,
                request_id,
                user_text,
                category=category,
            )
        }
        policy = self.read_policy()
        for name, item in self.entries.tools.items():
            source = self.entries.sources[name]
            permission = self.entries.permissions[name]
            fields = self.entries.result_fields[name]
            if source.startswith("mcp:"):
                rules = policy.get(source[4:], {})
                rule = rules.get(name, "deny") if isinstance(rules, dict) else "deny"
                permission = rule.get("access", "deny") if isinstance(rule, dict) else rule
                fields = rule.get("result_fields") if isinstance(rule, dict) else None
                # 外部写权限即便被误配，也不支持；模型/Server 的描述与 annotation 不参与授权。
                if permission != "read":
                    permission = "deny"
            registry.register(
                builtin[name] if name in self.builtin_names else item,
                source=source,
                permission=permission,
                result_fields=fields,
            )
        return registry
