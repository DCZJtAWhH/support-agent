"""用真实浏览器演示缺订单退款、流内点选续跑、原因表单及多轮意图切换。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from check_ch05_browser import send_message
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    """在桌面和移动视口操作实际页面及后端，保存截图、SSE 以及审计结论。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8006")
    args = parser.parse_args()
    rows, errors = [], []
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
                busy_replies = 0

                async def transient_busy(route):
                    """桌面端额外模拟一次提交收尾忙状态，其余请求全部访问真实后端。"""
                    nonlocal busy_replies
                    if route.request.post_data_json.get("selected_order_id") and not busy_replies:
                        busy_replies += 1
                        await route.fulfill(
                            status=409,
                            content_type="application/json",
                            body=json.dumps(
                                {"detail": "本次会话或请求正在执行，请等待结束后重试。"}
                            ),
                        )
                    else:
                        await route.continue_()

                if width == 1440:
                    await page.route("**/chat", transient_busy)
                await page.goto(args.base_url)
                await page.locator("#message").fill("这个能退吗")
                async with page.expect_response("**/chat", timeout=180000) as pending:
                    await page.locator("#send").click()
                first = await pending.value
                first_sse = await first.text()
                await page.locator(".order-card").first.wait_for(timeout=180000)
                await page.wait_for_function(
                    "!document.querySelector('#send').textContent.includes('正在')"
                )
                await page.locator(".order-card").first.scroll_into_view_if_needed()
                await page.screenshot(path=str(ROOT / f"dev-notes/ch06-picker-{width}.png"))
                assert "order_required" in first_sse and '"node": "retrieve"' not in first_sse
                meta = json.loads(first_sse.split("data: ", 1)[1].split("\n", 1)[0])
                async with page.expect_response(
                    lambda response: response.url.endswith("/chat") and response.status == 200,
                    timeout=180000,
                ) as pending:
                    await page.get_by_role("button", name="BX-500 保温杯", exact=False).click()
                resumed = await pending.value
                resumed_sse = await resumed.text()
                await page.locator(".refund-form").wait_for(timeout=180000)
                await page.wait_for_function(
                    "!document.querySelector('#send').textContent.includes('正在')"
                )
                assert '"selected_order_id":"1001"' in resumed.request.post_data
                assert resumed_sse.index('"node": "load_order"') < resumed_sse.index(
                    '"node": "retrieve"'
                )
                assert "retrieval_queries" in resumed_sse and "refund_form" in resumed_sse
                assert '"node": "resolve"' not in resumed_sse
                await page.get_by_label("退款原因", exact=True).select_option(label="不想要了")
                await page.get_by_role("button", name="提交退款申请", exact=True).click()
                await page.get_by_role("button", name="已提交申请", exact=True).wait_for(
                    timeout=30000
                )
                await page.locator(".refund-form").scroll_into_view_if_needed()
                await page.screenshot(path=str(ROOT / f"dev-notes/ch06-refund-{width}.png"))
                audit = await (
                    await page.request.get(
                        args.base_url + "/conversations/" + meta["conversation_id"]
                    )
                ).json()
                assert len(audit["refunds"]) == 1
                assert audit["refunds"][0]["order_id"] == "1001"
                assert await page.locator("body").evaluate("el => el.scrollWidth <= innerWidth")
                row = {
                    "width": width,
                    "conversation_id": meta["conversation_id"],
                    "picker_then_resume": True,
                    "refund": audit["refunds"][0],
                }
                if width == 1440:
                    assert busy_replies == 1
                    row["transient_busy_retried"] = True
                if width == 1440:
                    await send_message(page, "先不退了，这一单的物流到哪了")
                    await send_message(page, "它还能退吗")
                    await send_message(page, "先不退了，它的物流到哪了")
                    audit = await (
                        await page.request.get(
                            args.base_url + "/conversations/" + meta["conversation_id"]
                        )
                    ).json()
                    turns = [
                        json.loads(m["content"])
                        for m in audit["messages"]
                        if m["role"] == "workflow"
                    ]
                    assert [turn["intent"] for turn in turns] == [
                        "退款退货",
                        "物流",
                        "退款退货",
                        "物流",
                    ]
                    assert all("1001" in turn["resolved_question"] for turn in turns[1:])
                    row["multi_turn"] = [
                        {k: turn[k] for k in ("intent", "resolved_question", "trace")}
                        for turn in turns
                    ]
                (ROOT / f"dev-notes/ch06-resumed-{width}.sse").write_text(
                    resumed_sse.rstrip("\r\n") + "\n", encoding="utf-8", newline="\n"
                )
                rows.append(row)
                (ROOT / "dev-notes/ch06-browser-results.json").write_text(
                    json.dumps({"cases": rows, "page_errors": errors}, ensure_ascii=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                print(f"browser {width}: PASS", flush=True)
                await page.close()
        finally:
            await browser.close()
    assert not errors, errors


if __name__ == "__main__":
    asyncio.run(main())
