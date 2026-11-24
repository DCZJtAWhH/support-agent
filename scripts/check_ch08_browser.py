"""用实际浏览器验收工单预览确认/取消、投诉按钮和 MCP 工具结果。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def send(page, text):
    """发送当前问题并等候完整 SSE 与页面渲染，返回本轮会话/请求标识。"""
    await page.locator("#message").fill(text)
    async with page.expect_response("**/chat", timeout=60000) as pending:
        await page.locator("#send").click()
    response = await pending.value
    body = await response.text()
    assert "event: error" not in body, body
    await page.wait_for_function("!document.querySelector('#send').textContent.includes('正在')")
    return json.loads(body.split("data: ", 1)[1].split("\n", 1)[0])


async def main():
    """在桌面与两种手机宽度实际操作，断言落库和拒绝审计并保存截图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8009")
    args = parser.parse_args()
    report, errors = [], []
    async with httpx.AsyncClient() as client:
        for _ in range(50):
            try:
                response = await client.get(args.base_url, timeout=1)
                if response.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.2)
        else:
            raise RuntimeError("客服服务尚未启动，请先运行 scripts/run_ch08.py --offline")
    async with async_playwright() as playwright:
        cached = sorted(
            (Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright").glob(
                "chromium-*/chrome-win*/chrome.exe",
            )
        )
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=str(cached[-1]) if cached else None,
        )
        try:
            for width, height in [(1440, 1000), (390, 844), (320, 740)]:
                page = await browser.new_page(viewport={"width": width, "height": height})
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(args.base_url)
                for confirmed in [True, False]:
                    await page.locator("#new-conversation").evaluate("button => button.click()")
                    await send(page, "帮我建个工单")
                    assert await page.locator(".ticket-preview").count() == 0
                    assert "请描述" in await page.locator(".reply").last.inner_text()
                    meta = await send(page, "水壶不加热，希望维修")
                    await page.locator(".ticket-preview").wait_for()
                    assert "工单类型 · 维修" in await page.locator(".ticket-preview").inner_text()
                    await page.locator(".ticket-preview").scroll_into_view_if_needed()
                    await page.screenshot(path=str(ROOT / f"dev-notes/ch08-preview-{width}.png"))
                    before = await (
                        await page.request.get(
                            args.base_url + "/conversations/" + meta["conversation_id"],
                        )
                    ).json()
                    assert not before["tickets"]
                    async with page.expect_response("**/chat", timeout=60000) as pending:
                        await page.get_by_role(
                            "button", name="确认提交" if confirmed else "取消", exact=True
                        ).click()
                    resumed = await pending.value
                    resumed_body = await resumed.text()
                    assert "event: error" not in resumed_body, resumed_body
                    assert (
                        '"ticket_confirmation":' + str(confirmed).lower()
                        in resumed.request.post_data
                    )
                    await page.wait_for_function(
                        "!document.querySelector('#send').textContent.includes('正在')"
                    )
                    after = await (
                        await page.request.get(
                            args.base_url + "/conversations/" + meta["conversation_id"],
                        )
                    ).json()
                    rows = [
                        r for r in after["tool_audit_logs"] if r["tool_name"] == "create_ticket"
                    ]
                    assert len(rows) == 1
                    assert rows[0]["status"] == ("成功" if confirmed else "权限拒绝")
                    assert len(after["tickets"]) == int(confirmed)
                    if confirmed:
                        assert (
                            after["tickets"][0]["ticket_no"]
                            in await page.locator(".reply").last.inner_text()
                        )
                    assert await page.locator("body").evaluate("el => el.scrollWidth <= innerWidth")
                    outcome = "confirmed" if confirmed else "cancelled"
                    await page.screenshot(path=str(ROOT / f"dev-notes/ch08-{outcome}-{width}.png"))
                    report.append({"width": width, "confirmed": confirmed, "passed": True})
                if width == 1440:
                    await page.locator("#new-conversation").click()
                    meta = await send(page, "查询订单1001物流轨迹")
                    assert "演示数据" in await page.locator(".reply").last.inner_text()
                    audit = await (
                        await page.request.get(
                            args.base_url + "/conversations/" + meta["conversation_id"],
                        )
                    ).json()
                    assert audit["tool_audit_logs"][0]["source"] == "mcp:logistics"
                    await page.screenshot(path=str(ROOT / "dev-notes/ch08-logistics-1440.png"))
                    await page.locator("#new-conversation").click()
                    meta = await send(page, "我要投诉")
                    page.once("dialog", lambda dialog: dialog.accept())
                    await page.get_by_role("button", name="建工单", exact=True).click()
                    await page.get_by_role("button", name="已建工单", exact=False).wait_for()
                    audit = await (
                        await page.request.get(
                            args.base_url + "/conversations/" + meta["conversation_id"],
                        )
                    ).json()
                    assert len(audit["tickets"]) == 1
                await page.close()
        finally:
            await browser.close()
    result = {"cases": report, "browser_errors": errors, "passed": not errors}
    (ROOT / "dev-notes/ch08-browser-results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    assert not errors


if __name__ == "__main__":
    asyncio.run(main())
