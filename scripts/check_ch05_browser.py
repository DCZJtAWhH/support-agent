"""通过真实浏览器验证两个选项独立、确认建单、取消确认和继续聊天。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def send_message(page, text):
    """发送真实聊天并等待本轮完成，返回 SSE 的会话与请求标识。"""
    completed = await page.locator(".answer-feedback").count()
    await page.locator("#message").fill(text)
    async with page.expect_response("**/chat", timeout=180000) as pending:
        await page.locator("#send").click()
    response = await pending.value
    body = await response.text()
    await page.wait_for_function(
        "count => document.querySelectorAll('.answer-feedback').length > count",
        arg=completed,
        timeout=180000,
    )
    return json.loads(body.split("data: ", 1)[1].split("\n", 1)[0])


async def audit(page, base_url, cid):
    """通过公开会话接口读取工单，不直连数据库或依赖页面内部变量。"""
    response = await page.request.get(base_url + "/conversations/" + cid)
    assert response.ok
    return await response.json()


async def main():
    """在桌面及移动视口点击真实页面，保存可复核截图和行为报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8005")
    args = parser.parse_args()
    executable = None
    if os.name == "nt":
        cached = list(
            (Path(os.environ["LOCALAPPDATA"]) / "ms-playwright").glob(
                "chromium-*/chrome-win*/chrome.exe"
            )
        )
        if cached:
            executable = str(sorted(cached)[-1])
    report, errors = [], []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, executable_path=executable)
        try:
            for width, height in [(1440, 1000), (390, 844), (320, 740)]:
                page = await browser.new_page(viewport={"width": width, "height": height})
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(args.base_url)
                meta = await send_message(page, "我要投诉")
                cid = meta["conversation_id"]
                handoff = page.get_by_role("button", name="转人工", exact=True)
                ticket = page.get_by_role("button", name="建工单", exact=True)
                assert await handoff.count() == 1 and await ticket.count() == 1
                await page.screenshot(path=str(ROOT / f"dev-notes/ch05-choices-{width}.png"))
                assert not (await audit(page, args.base_url, cid))["tickets"]
                # 两者都不点击仍可正常继续，不会产生工单。
                await send_message(page, "谢谢")
                assert not (await audit(page, args.base_url, cid))["tickets"]
                await handoff.click()
                assert await page.get_by_text("已转接人工客服", exact=True).is_visible()
                assert await page.get_by_text(
                    "您好，我是客服小猫，请问有什么可以帮您的", exact=True
                ).is_visible()
                assert not (await audit(page, args.base_url, cid))["tickets"]
                assert await ticket.is_enabled()

                async def dismiss(dialog):
                    """第一次取消建单确认，确保没有后端副作用。"""
                    await dialog.dismiss()

                page.once("dialog", dismiss)
                await ticket.click()
                assert not (await audit(page, args.base_url, cid))["tickets"]

                async def accept(dialog):
                    """第二次明确确认建单，允许执行用户点击授权的工具。"""
                    await dialog.accept()

                page.once("dialog", accept)
                await ticket.click()
                await page.get_by_role("button", name="已建工单", exact=True).wait_for()
                saved = await audit(page, args.base_url, cid)
                assert len(saved["tickets"]) == 1 and saved["status"] == "active"
                assert await page.locator("body").evaluate(
                    "el => el.scrollWidth <= window.innerWidth"
                )
                await page.locator(".message.assistant").last.scroll_into_view_if_needed()
                await page.screenshot(path=str(ROOT / f"dev-notes/ch05-actions-{width}.png"))
                report.append(
                    {
                        "width": width,
                        "conversation_id": cid,
                        "independent_actions": True,
                        "ignore_and_continue": True,
                        "cancel_no_ticket": True,
                        "ticket_count": len(saved["tickets"]),
                    }
                )
                # 新会话只点建单，不应出现转人工消息，且转人工按钮保持可用。
                await page.locator("#new-conversation").click()
                second = await send_message(page, "我要投诉")
                page.once("dialog", accept)
                await page.get_by_role("button", name="建工单", exact=True).click()
                await page.get_by_role("button", name="已建工单", exact=True).wait_for()
                assert await page.get_by_text("已转接人工客服", exact=True).count() == 0
                assert await page.get_by_role("button", name="转人工", exact=True).is_enabled()
                assert (
                    len((await audit(page, args.base_url, second["conversation_id"]))["tickets"])
                    == 1
                )
                # 知识回答仍保留原页面的可点击引用，防止新 Prompt 破坏旧能力。
                await page.locator("#new-conversation").click()
                await send_message(page, "BX-500容量是多少")
                await page.locator(".citation").first.click()
                assert "500" in await page.locator(".source-text").inner_text()
                await page.get_by_role("button", name="关闭 ×", exact=True).click()
                await page.close()
        finally:
            await browser.close()
    (ROOT / "dev-notes/ch05-browser-results.json").write_text(
        json.dumps({"cases": report, "page_errors": errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    assert not errors, errors
    print(json.dumps({"viewports": len(report), "passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
