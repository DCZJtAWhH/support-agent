"""用真实浏览器验证工具徽章、真实 SSE 文本与桌面及手机布局。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    """通过聊天页运行三个验收问题，并保存截图和浏览器结果供复核。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--chromium", default=None)
    args = parser.parse_args()
    executable = args.chromium
    if not executable and os.name == "nt":
        cached = list(
            (Path(os.environ["LOCALAPPDATA"]) / "ms-playwright").glob(
                "chromium-*/chrome-win*/chrome.exe"
            )
        )
        if cached:
            executable = str(sorted(cached)[-1])
    report = []
    errors = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, executable_path=executable)
        try:
            page = await browser.new_page(
                viewport={"width": 1440, "height": 1000}, reduced_motion="reduce"
            )
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            for index, (message, tool_name) in enumerate(
                [
                    ("订单 1001 的物流到哪了", "query_logistics"),
                    ("退货政策是什么", "query_faq"),
                    ("邮费是多少", "query_faq"),
                ]
            ):
                await page.goto(args.base_url)
                await page.get_by_label("输入您的问题").fill(message)
                await page.get_by_role("button", name="发送消息").click()
                await page.wait_for_function(
                    "document.querySelector('#status').textContent.includes('您可以继续追问')"
                    " || !document.querySelector('#notice').hidden",
                    timeout=180000,
                )
                assert await page.locator("#notice").is_hidden(), await page.locator(
                    "#notice-text"
                ).inner_text()
                badge = page.locator(".assistant .tool-badge")
                badges = await badge.all_text_contents()
                assert any(tool_name in text and "已完成" in text for text in badges)
                answer = await page.locator(".assistant .reply").inner_text()
                assert answer.strip()
                await page.screenshot(path=str(ROOT / f"dev-notes/ch02-browser-{index + 1}.png"))
                report.append({"message": message, "badges": badges, "answer": answer})
            await page.set_viewport_size({"width": 390, "height": 844})
            assert await page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            await page.screenshot(path=str(ROOT / "dev-notes/ch02-browser-mobile.png"))
            assert not errors, errors
        finally:
            await browser.close()
    (ROOT / "dev-notes/ch02-browser-results.json").write_text(
        json.dumps(
            {"cases": report, "page_errors": errors, "mobile_overflow": False},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("浏览器验收通过：三个真实聊天案例、工具徽章、手机无横向溢出、无页面异常。")


if __name__ == "__main__":
    asyncio.run(main())
