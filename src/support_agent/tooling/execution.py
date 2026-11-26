"""注册 LangChain 业务工具，集中校验参数并处理超时、重试与脱敏错误。"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

import httpx
from jsonschema.validators import validator_for
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)


class ToolArgumentsError(ValueError):
    """表示通过 Schema 校验后仍违反业务参数约束。"""


class ToolNotFoundError(ValueError):
    """表示查询正常完成但没有业务数据，不能用重试伪装故障。"""


def transient_error(exc):
    """只识别网络中断、超时及明确可恢复数据库故障，保留异常组中的真实原因。"""
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(transient_error(e) for e in exc.exceptions)
    if isinstance(
        exc,
        (
            TimeoutError,
            httpx.NetworkError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            ConnectionError,
        ),
    ):
        return True
    if isinstance(exc, OperationalError):
        return exc.connection_invalidated or getattr(exc.orig, "args", [None])[0] in {
            1205,
            1213,
            2006,
            2013,
        }
    return False


def timeout_error(exc):
    """识别适配器任务组包装后的网络超时，用于最终审计状态。"""
    if isinstance(exc, BaseExceptionGroup):
        return any(timeout_error(e) for e in exc.exceptions)
    return isinstance(exc, (TimeoutError, httpx.TimeoutException))


def format_result(data, fields=None, *, field=None):
    """抽取本地白名单字段并将内部状态码转换为中文，不把内部备注交给模型。"""
    labels = {
        "pending": "待处理",
        "in_transit": "运输中",
        "delivered": "已签收",
        "in_warranty": "在保",
        "expired": "已过保",
        "processing": "处理中",
        "received": "已收到退货",
        "refunded": "已退款",
        "CNY": "人民币",
    }
    if isinstance(data, dict):
        return {
            key: format_result(value, field=key)
            for key, value in data.items()
            if (fields is None or key in fields) and not key.startswith("internal_")
        }
    if isinstance(data, list):
        return [format_result(item, fields, field=field) for item in data]
    return (
        labels.get(data, data)
        if isinstance(data, str) and field in {"status", "currency"}
        else data
    )


@dataclass(frozen=True)
class ConfirmedWrite:
    """由本地装配指定可确认写能力及参数验证，不允许外部工具自行声明授权。"""

    name: str
    result_fields: list[str]
    validate: Callable[[dict, list[str]], bool]
    invalid_message: str


class ToolRegistry:
    def __init__(
        self,
        tools: list[BaseTool],
        *,
        timeout: float = 5,
        retries: int = 1,
        retry_delay: float = 0.1,
        repo=None,
        conversation_id: str = "",
        write_requested: bool = False,
        write_policy: ConfirmedWrite | None = None,
        confirmation=None,
        audit_timeout: float = 2,
    ):
        """注册唯一命名的工具，配置单次超时和最多额外重试次数。"""
        self.write_policy = write_policy
        self.tools = {}
        self.schemas = {}
        self.sources = {}
        self.permissions = {}
        self.result_fields = {}
        for item in tools:
            self.register(item)
        self.timeout, self.retries, self.retry_delay = timeout, retries, retry_delay
        self.repo, self.conversation_id = repo, conversation_id
        self.write_requested, self.confirmation = write_requested, confirmation
        self.input_sources = None
        self.audit_timeout = audit_timeout

    def register(self, item: BaseTool, *, source="builtin", permission="read", result_fields=None):
        """校验工具三要素并即时登记，拒绝无描述、非法 Schema 或重复名称。"""
        if permission not in {"read", "write", "deny"}:
            raise ValueError("工具权限只接受 read、write 或 deny。")
        if not item.name or not item.description or item.name in self.tools:
            raise ValueError("工具名称必须唯一且用途描述不能为空。")
        schema = item.args_schema
        schema = (
            schema
            if isinstance(schema, dict)
            else convert_to_openai_tool(item)["function"]["parameters"]
        )
        validator_for(schema).check_schema(schema)
        self.tools[item.name] = item
        self.schemas[item.name] = schema
        self.sources[item.name] = source
        self.permissions[item.name] = (
            "write" if self.write_policy and item.name == self.write_policy.name else permission
        )
        default_fields = (
            self.write_policy.result_fields
            if self.write_policy and item.name == self.write_policy.name
            else None
        )
        self.result_fields[item.name] = (
            result_fields if result_fields is not None else default_fields
        )

    def validation_error(self, name, arguments):
        """严格按照工具声明的 JSON Schema 校验并返回字段路径，合法时返回空值。"""
        schema = self.schemas.get(name)
        if schema is None:
            return None
        errors = list(validator_for(schema)(schema).iter_errors(arguments))
        if errors:
            detail = "; ".join(
                f"{'.'.join(map(str, error.absolute_path)) or '$'}: {error.message}"
                for error in errors[:8]
            )
            return self.failure("invalid_arguments", "参数校验未通过：" + detail, 0)
        return None

    async def execute(
        self, name: str, arguments: dict, *, tool_call_id: str = "", confirmed: bool = False
    ) -> dict:
        """统一执行并尽力记录一次审计；审计写入失败不会改变工具结果或阻止执行。"""
        started = perf_counter()
        cancelled = None
        progress = {"attempts": 0}
        try:
            result = await self._execute(
                name,
                arguments,
                confirmed=confirmed,
                tool_call_id=tool_call_id,
                progress=progress,
            )
        except asyncio.CancelledError as exc:
            cancelled = exc
            result = self.failure(
                "cancelled",
                "工具调用已取消；写操作的执行结果可能尚不确定。",
                progress["attempts"],
            )
        if self.repo:
            code = result.get("error", {}).get("code")
            status = {
                "invalid_arguments": "校验拦下",
                "permission_denied": "权限拒绝",
                "timeout": "超时",
            }.get(code, "成功" if result["ok"] else "失败")
            try:
                async with asyncio.timeout(self.audit_timeout):
                    await self.repo.record_tool_audit(
                        conversation_id=self.conversation_id,
                        tool_call_id=tool_call_id or uuid4().hex,
                        tool_name=name,
                        source=self.sources.get(name, "unknown"),
                        arguments=arguments,
                        result_summary=json.dumps(result, ensure_ascii=False)[:2000],
                        status=status,
                        error=result.get("error", {}).get("message"),
                        retry_count=max(0, result["attempts"] - 1),
                        duration_ms=max(0, round((perf_counter() - started) * 1000)),
                    )
            except Exception:
                logger.warning("工具审计写入失败 tool=%s call=%s", name, tool_call_id)
        if cancelled is not None:
            raise cancelled
        return result

    async def _execute(self, name, arguments, *, confirmed=False, tool_call_id="", progress=None):
        """执行工具并返回统一结果；仅超时和数据库瞬时故障可重试，取消信号直接传播。"""
        selected = self.tools.get(name)
        if not selected:
            return self.failure("unknown_tool", "模型申请了未注册的工具。", 0)
        invalid = self.validation_error(name, arguments)
        if invalid:
            return invalid
        permission = self.permissions[name]
        approved_write = self.write_policy and name == self.write_policy.name
        if (
            approved_write
            and self.write_requested
            and self.input_sources is not None
            and not self.write_policy.validate(arguments, self.input_sources)
        ):
            return self.failure("invalid_arguments", self.write_policy.invalid_message, 0)
        if (
            approved_write
            and self.sources[name] == "builtin"
            and not confirmed
            and self.write_requested
            and self.confirmation is not None
        ):
            # 暂停在超时与重试之外，恢复时只接受前端明确布尔确认。
            confirmed = self.confirmation({**arguments, "tool_call_id": tool_call_id}) is True
        if permission == "deny" or (
            permission == "write"
            and (not approved_write or self.sources[name] != "builtin" or not confirmed)
        ):
            return self.failure(
                "permission_denied", "本系统未授权该工具，或建工单尚未获得前端确认。", 0
            )
        retries = 0 if permission == "write" else self.retries
        for attempt in range(1, retries + 2):
            if progress is not None:
                progress["attempts"] = attempt
            try:
                async with asyncio.timeout(self.timeout):
                    raw = await selected.ainvoke(
                        {
                            "type": "tool_call",
                            "name": name,
                            "args": arguments,
                            "id": uuid4().hex,
                        }
                    )
                if isinstance(raw, ToolMessage):
                    artifact = raw.artifact or {}
                    if "structured_content" in artifact:
                        data = artifact["structured_content"]
                    else:
                        content = raw.content
                        if isinstance(content, list):
                            content = "\n".join(
                                block.get("text", "")
                                for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                            )
                        try:
                            data = json.loads(content)
                        except (ValueError, TypeError):
                            data = content
                else:
                    data = raw
                if (
                    data is None
                    or data == []
                    or data == {}
                    or (isinstance(data, dict) and data.get("found") is False)
                ):
                    raise ToolNotFoundError("查询已完成，但未找到匹配记录，请核对编号或补充信息。")
                return {
                    "ok": True,
                    "data": format_result(data, self.result_fields[name]),
                    "attempts": attempt,
                }
            except ToolArgumentsError as exc:
                return self.failure("invalid_arguments", str(exc), attempt)
            except ToolNotFoundError as exc:
                return self.failure("not_found", str(exc), attempt)
            except ValidationError:
                return self.failure("invalid_arguments", "工具参数不符合 Schema 约束。", attempt)
            except Exception as exc:
                if transient_error(exc) and attempt <= retries:
                    await asyncio.sleep(self.retry_delay)
                    continue
                code = (
                    "timeout"
                    if timeout_error(exc)
                    else ("unavailable" if transient_error(exc) else "execution_failed")
                )
                message = "工具执行失败，请稍后重试或联系人工。"
                if permission == "write" and code == "timeout":
                    message = (
                        "写操作超时，执行结果尚不确定，未自动重试；请先核对工单记录，避免重复提交。"
                    )
                return self.failure(code, message, attempt)

    @staticmethod
    def failure(code: str, message: str, attempts: int) -> dict:
        """构造不含异常原文、数据库地址或密钥的失败结果，供模型如实说明。"""
        category = {
            "invalid_arguments": "参数不合法",
            "not_found": "查询落空",
            "permission_denied": "权限拒绝",
        }.get(code, "真故障")
        return {
            "ok": False,
            "error": {"code": code, "category": category, "message": message},
            "attempts": attempts,
        }
