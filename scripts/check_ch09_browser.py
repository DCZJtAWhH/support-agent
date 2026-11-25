"""在真实服务上验证反馈、审核、知识再答以及桌面和移动端布局。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "CH09演示杯支持激光刻字服务吗"
APPROVED = "仅供第九章功能验收：CH09演示杯支持激光刻字，最多八个汉字；此规则不适用于真实商品。"


async def check(args):
    """使用浏览器实际点击反馈与审核，所有业务数据写入现有演示数据库。"""
    results, errors = {}, []
    async with async_playwright() as browser_tools:
        cached = sorted(
            (Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright").glob(
                "chromium-*/chrome-win*/chrome.exe",
            )
        )
        browser = await browser_tools.chromium.launch(
            headless=True,
            executable_path=str(cached[-1]) if cached else None,
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(args.base_url)
        await page.locator("#message").fill(QUESTION)
        await page.get_by_role("button", name="发送消息").click()
        await expect(page.locator(".answer-feedback").last).to_be_visible(timeout=120000)
        await expect(page.locator(".message.assistant .reply").last).to_contain_text("无法确认")
        await page.get_by_role("button", name="不满意", exact=True).last.click()
        await expect(page.locator(".answer-feedback").last).to_contain_text("已反馈", timeout=30000)
        await page.screenshot(path=str(ROOT / "dev-notes/ch09-feedback-1440.png"))
        results["feedback_posted_in_browser"] = True
        admin = await browser.new_page(viewport={"width": 1440, "height": 1000})
        admin.on("pageerror", lambda error: errors.append(str(error)))
        await admin.goto(args.base_url + "/reviews")
        for _ in range(60):
            response = await admin.request.get(args.base_url + "/api/review-queue")
            rows = await response.json()
            row = next((r for r in rows if "CH09" in r["normalized_question"]), None)
            if row:
                break
            await asyncio.sleep(1)
        assert row is not None, "问题未自动进入待审队列"
        results["review_id"] = row["id"]
        await admin.get_by_role("button", name="刷新数据").click()
        item = admin.locator(f".review-row[data-review-id='{row['id']}']")
        await item.locator(":scope > summary").click()
        await expect(item.locator(".origin").first).to_be_visible()
        await expect(item.locator(".origins")).to_contain_text(QUESTION)
        snapshot = item.locator(".snapshot").first
        await expect(snapshot).to_be_visible()
        await snapshot.locator("summary").click()
        await expect(snapshot.locator("pre")).not_to_be_empty()
        results["original_question_and_snapshot_visible"] = True
        for width, height in [(1440, 1000), (1024, 900), (768, 900), (390, 844), (320, 740)]:
            await admin.set_viewport_size({"width": width, "height": height})
            assert await admin.evaluate("document.documentElement.scrollWidth <= innerWidth")
            await admin.screenshot(
                path=str(ROOT / f"dev-notes/ch09-review-{width}.png"), full_page=True
            )
        results["responsive_widths"] = [1440, 1024, 768, 390, 320]
        if args.approve:
            # 仅发布带有显式验收标记的演示答案，不编造真实商品政策。
            await admin.set_viewport_size({"width": 1440, "height": 1000})
            await item.locator("textarea").fill(APPROVED)
            await item.get_by_role("button", name="通过并写入知识库").click()
            await expect(admin.locator("#review-status")).to_contain_text(
                "知识已入库", timeout=120000
            )
            await page.locator("#message").fill(QUESTION)
            await page.get_by_role("button", name="发送消息").click()
            await expect(page.locator(".answer-feedback")).to_have_count(2, timeout=120000)
            await expect(page.locator(".message.assistant .reply").last).to_contain_text("八个汉字")
            await page.screenshot(path=str(ROOT / "dev-notes/ch09-approved-answer-1440.png"))
            results["approved_knowledge_answered_same_question"] = True
        for width in (390, 320):
            await page.set_viewport_size({"width": width, "height": 844})
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            await page.screenshot(path=str(ROOT / f"dev-notes/ch09-chat-{width}.png"))
        assert not errors, errors
        results["page_errors"] = errors
        await browser.close()
    (ROOT / "dev-notes/ch09-browser-results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    """选择本地端口，显式 --approve 进行完整的演示知识发布验收。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--approve", action="store_true")
    asyncio.run(check(parser.parse_args()))


if __name__ == "__main__":
    main()
