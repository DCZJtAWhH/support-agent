"""持久化真实模型测试预算；预留后才发送，未知扣费立即停止后续请求。"""

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import httpx

# 2026-11-24 核对官方人民币价目表；按高峰、无缓存费率保守结算，单位为元/百万tokens。
PRICE_VERIFIED_ON = date(2026, 11, 24)
PRICES = {
    "deepseek-flash": (Decimal("2"), Decimal("8")),
    "deepseek-v4-pro": (Decimal("9"), Decimal("27")),
}


class BudgetExceeded(RuntimeError):
    """表示额度不足、账目不确定或预算账本异常，禁止继续真实调用。"""


class BudgetLedger:
    def __init__(self, path, *, limit=Decimal("1.00")):
        """打开累计账本，重启不会清零；已有额度不能提高，绝不超过人民币1元。"""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        limit = min(Decimal(str(limit)), Decimal("1.00"))
        if not limit.is_finite() or limit <= 0:
            raise ValueError("预算必须大于0且不超过1元。")
        with self.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, cap TEXT, halted TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS calls "
                "(id TEXT PRIMARY KEY, model TEXT, reserved TEXT, charged TEXT, detail TEXT)"
            )
            conn.execute("INSERT OR IGNORE INTO budget VALUES (1, ?, '')", (str(limit),))
            current = Decimal(conn.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0])
            conn.execute("UPDATE budget SET cap=? WHERE id=1", (str(min(current, limit)),))

    @contextmanager
    def transaction(self):
        """使用进程间写锁完成预算比较与预留，防止并发和多脚本竞争超额。"""
        with closing(sqlite3.connect(self.path, timeout=10)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            yield conn

    @staticmethod
    def committed(conn):
        """按实际已结算费用加尚未释放的预留计算占用，金额使用十进制定点运算。"""
        return sum(
            (Decimal(c or r) for r, c in conn.execute("SELECT reserved, charged FROM calls")),
            Decimal("0.00"),
        )

    def reserve(self, model, maximum, *, detail=None):
        """发送前原子预留单次费用上界；上次请求未结算或余额不足时直接拒绝。"""
        maximum = Decimal(str(maximum))
        if not maximum.is_finite() or maximum <= 0:
            raise BudgetExceeded("无法确定本次请求费用上界，停止真实调用。")
        with self.transaction() as conn:
            cap, halted = conn.execute("SELECT cap, halted FROM budget WHERE id=1").fetchone()
            pending = conn.execute("SELECT 1 FROM calls WHERE charged IS NULL LIMIT 1").fetchone()
            if halted or pending:
                raise BudgetExceeded("预算存在未确认扣费，停止真实调用；请先核对账本。")
            if self.committed(conn) + maximum > Decimal(cap):
                raise BudgetExceeded("本次请求可能超过累计1元预算，已在发送前拦截。")
            identifier = uuid4().hex
            conn.execute(
                "INSERT INTO calls VALUES (?, ?, ?, NULL, ?)",
                (identifier, model, str(maximum), json.dumps(detail or {}, ensure_ascii=False)),
            )
            return identifier

    def settle(self, identifier, actual, *, detail=None):
        """按可信用量结算；超过预留或出现重复结算时保留阻塞，不隐瞒异常费用。"""
        actual = Decimal(str(actual))
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT reserved, charged FROM calls WHERE id=?", (identifier,)
            ).fetchone()
            if not row or row[1] is not None:
                raise BudgetExceeded("预算调用不存在或已结算。")
            if not actual.is_finite() or actual < 0 or actual > Decimal(row[0]):
                conn.execute("UPDATE budget SET halted='实际用量超出预留或无效' WHERE id=1")
            else:
                conn.execute(
                    "UPDATE calls SET charged=?, detail=? WHERE id=?",
                    (str(actual), json.dumps(detail or {}, ensure_ascii=False), identifier),
                )

    def halt(self, reason):
        """记录用量未知或连接异常；保留原预留费用且不自动恢复额度。"""
        with self.transaction() as conn:
            conn.execute("UPDATE budget SET halted=? WHERE id=1", (reason,))

    def observe_usage(self, identifier, usage, done):
        """在校验前保存可核对的数字用量和结束标记，方便失败后离线核账。"""
        numeric = {key: value for key, value in usage.items() if type(value) is int}
        with self.transaction() as conn:
            row = conn.execute("SELECT detail FROM calls WHERE id=?", (identifier,)).fetchone()
            detail = {**json.loads(row[0]), "observed_usage": numeric, "stream_done": done}
            conn.execute(
                "UPDATE calls SET detail=? WHERE id=?",
                (json.dumps(detail, ensure_ascii=False), identifier),
            )

    def report(self):
        """读取可审计费用与预留记录，不保存密钥、模型正文或推理内容。"""
        with self.transaction() as conn:
            cap, halted = conn.execute("SELECT cap, halted FROM budget WHERE id=1").fetchone()
            return {
                "limit_cny": cap,
                "committed_cny": str(self.committed(conn)),
                "halted": halted,
                "calls": [
                    {
                        "id": i,
                        "model": m,
                        "reserved_cny": r,
                        "charged_cny": c,
                        "detail": json.loads(d),
                    }
                    for i, m, r, c, d in conn.execute("SELECT * FROM calls ORDER BY rowid")
                ],
            }


class BudgetTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner, ledger, *, pricing_verified_on=PRICE_VERIFIED_ON):
        """包装真实网络出口，所有模型名与所有脚本共享同一个账本；不修改请求正文。"""
        self.inner, self.ledger = inner, ledger
        self.pricing_verified_on = pricing_verified_on

    async def handle_async_request(self, request):
        """先校验计价与请求上界并预留，再发送；异常保留预留且停止后续调用。"""
        today = datetime.now(timezone(timedelta(hours=8))).date()
        if self.pricing_verified_on != today:
            raise BudgetExceeded("官方价格快照尚未核对到今天，禁止真实测试。")
        if request.url.host != "api.deepseek.com" or not request.url.path.endswith(
            "/chat/completions"
        ):
            raise BudgetExceeded("费用闸仅支持已核价的 DeepSeek 官方文本聊天接口。")
        payload = json.loads(await request.aread())
        model = payload.get("model")
        maximum = payload.get("max_tokens", payload.get("max_completion_tokens"))
        if model not in PRICES or type(maximum) is not int or not 0 < maximum <= 4096:
            raise BudgetExceeded("模型未核价或缺少明确的小额输出上限，拒绝发送。")
        messages = payload.get("messages", [])
        if (
            payload.get("n", 1) != 1
            or not messages
            or any(not isinstance(m.get("content"), (str, type(None))) for m in messages)
        ):
            raise BudgetExceeded("预算测试仅接受单份纯文本输出，不接受图像、文件或音频。")
        if any(t.get("type") != "function" for t in payload.get("tools", [])):
            raise BudgetExceeded("预算测试不启用额外按次收费的服务端工具。")
        # UTF-8字节的两倍加32768个封装token，覆盖文本、Schema及消息/工具包装；不按中文字符粗估。
        input_bound = len(request.content) * 2 + 32768
        input_price, output_price = PRICES[model]
        reserved = (input_bound * input_price + maximum * output_price) / Decimal(1_000_000)
        identifier = self.ledger.reserve(
            model,
            reserved,
            detail={
                "input_token_bound": input_bound,
                "max_output_tokens": maximum,
                "basis": "官方高峰、输入全部按未命中计价的保守上界",
            },
        )
        try:
            response = await self.inner.handle_async_request(request)
            if response.status_code != 200:
                await response.aclose()
                raise BudgetExceeded("模型返回非200且扣费未知，保留预留并停止真实调用。")
            stream = BudgetStream(
                response.stream,
                self.ledger,
                identifier,
                model,
                input_bound,
                maximum,
                streaming=payload.get("stream", False),
            )
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                stream=stream,
                extensions=response.extensions,
            )
        except BaseException:
            self.ledger.halt("请求失败或被取消，扣费尚未确认。")
            raise

    async def aclose(self):
        """关闭被包装的网络连接池。"""
        await self.inner.aclose()


class BudgetStream(httpx.AsyncByteStream):
    def __init__(self, inner, ledger, identifier, model, input_bound, maximum, *, streaming):
        """透传流式正文，同时只提取费用用量，不把正文或推理内容写入费用账本。"""
        self.inner, self.ledger, self.identifier = inner, ledger, identifier
        self.model, self.input_bound, self.maximum = model, input_bound, maximum
        self.streaming, self.buffer, self.usage, self.finished = streaming, b"", None, False
        self.done = not streaming

    async def __aiter__(self):
        """保持原始分片流速；只有完整消费且用量有效，才释放多余预留额度。"""
        try:
            async for chunk in self.inner:
                self.buffer += chunk
                if self.streaming:
                    while b"\n" in self.buffer:
                        line, self.buffer = self.buffer.split(b"\n", 1)
                        if line.startswith(b"data:"):
                            value = line[5:].strip()
                            if value == b"[DONE]":
                                self.done = True
                            else:
                                row = json.loads(value)
                                if row.get("usage") is not None:
                                    self.usage = row["usage"]
                yield chunk
            if not self.streaming:
                self.usage = json.loads(self.buffer).get("usage")
            self.finish()
        finally:
            if not self.finished:
                self.ledger.halt("响应中断或用量缺失，保留费用预留，停止真实调用。")

    def finish(self):
        """按实际输入/输出总用量保守结算，思考token已在输出总数中，不重复加计。"""
        usage = self.usage or {}
        self.ledger.observe_usage(self.identifier, usage, self.done)
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if (
            not self.done
            or type(prompt) is not int
            or type(completion) is not int
            or not 0 <= prompt <= self.input_bound
            or not 0 <= completion <= self.maximum
            or usage.get("total_tokens") != prompt + completion
        ):
            raise BudgetExceeded("用量缺失或超出预留上界，停止后续真实调用。")
        input_price, output_price = PRICES[self.model]
        actual = (prompt * input_price + completion * output_price) / Decimal(1_000_000)
        self.ledger.settle(
            self.identifier,
            actual,
            detail={
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "basis": "按实际用量与高峰未命中价保守计费，实际账单可能更低",
            },
        )
        self.finished = True

    async def aclose(self):
        """断流保留预留，关闭底层连接；不把取消视为没有扣费。"""
        await self.inner.aclose()
        if not self.finished:
            self.ledger.halt("响应未完整消费，扣费尚未确认。")
