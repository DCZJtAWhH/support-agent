"""用真实浏览器验证引用、来源原文、反馈锁定和未知问题拒答。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    """在桌面和手机视口操作真实聊天服务，保存截图与交互结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    args = parser.parse_args()
    executable = None
    if os.name == "nt":
        cached = list(
            (Path(os.environ["LOCALAPPDATA"]) / "ms-playwright").glob(
                "chromium-*/chrome-win*/chrome.exe",
            )
        )
        if cached:
            executable = str(sorted(cached)[-1])
    report, errors = [], []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, executable_path=executable)
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            for width, height in [(1440, 1000), (390, 844)]:
                await page.set_viewport_size({"width": width, "height": height})
                await page.goto(args.base_url)
                await page.locator("#message").fill("BX-500容量是多少")
                await page.locator("#send").click()
                feedback = page.locator(".assistant .answer-feedback").last
                await feedback.wait_for(timeout=180000)
                assert await page.locator("#notice").is_hidden(), await page.locator(
                    "#notice-text"
                ).inner_text()
                answer = await page.locator(".assistant .reply").last.inner_text()
                assert "500" in answer
                citation = page.locator(".assistant .citation").first
                await citation.click()
                dialog = page.locator("#source-dialog")
                assert await dialog.is_visible()
                assert "BX-500" in await dialog.locator(".source-path").inner_text()
                assert "500毫升" in await dialog.locator(".source-text").inner_text()
                async with page.expect_popup() as popup_event:
                    await dialog.get_by_role("link", name="查看原文与所在章节").click()
                original = await popup_event.value
                await original.wait_for_load_state()
                assert "500毫升" in await original.locator("body").inner_text()
                assert await original.locator(":target").count() == 1
                await original.close()
                await page.screenshot(path=str(ROOT / f"dev-notes/ch04-source-{width}.png"))
                await dialog.get_by_role("button", name="关闭").click()
                await feedback.get_by_role("button", name="满意", exact=True).click()
                assert await feedback.get_by_text("已反馈", exact=True).is_visible()
                assert (
                    await feedback.get_by_role("button", name="满意", exact=True).get_attribute(
                        "aria-pressed"
                    )
                    == "true"
                )
                assert await feedback.get_by_role("button", name="满意", exact=True).is_disabled()
                assert await feedback.get_by_role("button", name="不满意", exact=True).is_disabled()
                stored = await page.evaluate(
                    "Object.keys(localStorage)"
                    ".filter(k => k.startsWith('support-agent:feedback:'))"
                    ".map(k => JSON.parse(localStorage[k]))"
                )
                assert stored and stored[-1]["value"] == "up"
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                await page.screenshot(path=str(ROOT / f"dev-notes/ch04-chat-{width}.png"))
                report.append(
                    {
                        "width": width,
                        "answer": answer,
                        "source_popup": True,
                        "feedback_locked": True,
                        "local_feedback": stored[-1],
                    }
                )
            await page.locator("#new-conversation").click()
            await page.locator("#message").fill("火星殖民飞船燃料补贴是多少")
            await page.locator("#send").click()
            await page.locator(".assistant .answer-feedback").wait_for(timeout=180000)
            assert "无法确认" in await page.locator(".assistant .reply").inner_text()
            assert await page.locator(".citation").count() == 0
            await (
                page.locator(".answer-feedback")
                .get_by_role("button", name="不满意", exact=True)
                .click()
            )
            assert await page.locator(".answer-feedback").get_by_text("已反馈").is_visible()
            report.append(
                {"unknown_refused": True, "no_fabricated_citations": True, "down_feedback": True}
            )
            assert not errors, errors
        finally:
            await browser.close()
    (ROOT / "dev-notes/ch04-browser-results.json").write_text(
        json.dumps({"cases": report, "page_errors": errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("浏览器验收通过：桌面/手机引用、原文定位、满意度锁定、本地采集、未知问题拒答。")


if __name__ == "__main__":
    asyncio.run(main())
