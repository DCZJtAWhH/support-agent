# 改造前持久数据

`pre-modular.json.gz` 来自提交 `510e000d2ede7ad2ca2ab2749b0b626177667edd` 的真实 ASGI 入口。
它包含独立 SQLite 业务库和 checkpoint 的 SQL 转储、请求身份及已确认工单号，不包含凭据或
真实顾客数据。gzip 仅压缩文本，测试在临时目录恢复，验收只通过现有 HTTP/SSE 入口观察。

四种记录：投诉回复已完成、工单等待确认、工单确认成功但模型回复失败、订单等待点选。
夹具生成使用可控模型 HTTP 替身；网络失败仅模拟模型边界，实际存储和 LangGraph 未替换。

需要重建时，先把上述固定提交的 `src/`、`config/` 和 `tests/conftest.py` 导出至独立目录，运行：

```powershell
uv run --locked python tests/fixtures/generate_pre_modular.py <旧提交目录> tests/fixtures/pre-modular.json.gz
```

生成器只能使用旧版本源码；它不是测试的一部分，也不会在测试期间调用 git 或联网。
