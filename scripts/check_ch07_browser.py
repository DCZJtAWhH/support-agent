"""在离线模型服务上切换默认和已摘要会话，验证原文回载、移动抽屉及续聊。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    """使用验收报告中的会话，不创建工单/退款；只在桌面旧会话追加一轮白名单感谢。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8007")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    demo = json.loads(
        (ROOT / "dev-notes/ch07-offline-demo-acceptance.json").read_text(encoding="utf-8")
    )
    default = json.loads(
        (ROOT / "dev-notes/ch07-offline-default-acceptance.json").read_text(encoding="utf-8")
    )
    complete = demo["passed"] and default["passed"]
    assert complete or args.allow_partial, (
        "先完成23轮验收；额度受阻时可显式 --allow-partial 验证界面"
    )
    rows, errors = [], []
    async with (
        httpx.AsyncClient(base_url=args.base_url) as client,
        async_playwright() as playwright,
    ):
        status = await client.get("/_offline/status")
        assert status.status_code == 200 and status.json().get("offline"), "仅允许离线服务"
        cached = sorted(
            (Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright").glob(
                "chromium-*/chrome-win*/chrome.exe"
            )
        )
        browser = await playwright.chromium.launch(
            executable_path=str(cached[-1]) if cached else None
        )
        try:
            for width in (1440, 1024, 768, 390, 320):
                page = await browser.new_page(viewport={"width": width, "height": 900})
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(args.base_url)
                for report in (demo, default, demo):
                    cid = report["conversation_id"]
                    original = (await client.get(f"/api/conversations/{cid}/messages")).json()
                    if width < 700:
                        await page.locator("#conversation-toggle").click()
                    button = page.locator(f'#conversation-list [data-conversation-id="{cid}"]')
                    await button.wait_for()
                    await button.click()
                    await page.wait_for_function(
                        "count => document.querySelectorAll('.message .reply').length === count",
                        arg=len(original),
                    )
                    assert await page.locator(".message .reply").all_text_contents() == [
                        m["content"] for m in original
                    ]
                    assert await page.locator("#notice").is_hidden()
                    if report is demo:
                        assert await button.locator(".summary-badge").count() == 1
                # 桌面接着聊一次，其他宽度只读验证同一份完整原文。
                if width == 1440:
                    if not complete:
                        # 独立白名单会话零模型调用，不改变待恢复的业务会话 checkpoint。
                        response = await client.post("/chat", json={"message": "你好"})
                        frame = next(
                            s for s in response.text.split("\n\n") if s.startswith("event: meta")
                        )
                        cid = json.loads(frame.split("data: ", 1)[1])["conversation_id"]
                        await page.reload()
                        button = page.locator(f'#conversation-list [data-conversation-id="{cid}"]')
                        await button.click()
                        await page.wait_for_function(
                            "document.querySelectorAll('.message .reply').length === 2"
                        )
                    else:
                        cid = demo["conversation_id"]
                    await page.locator("#message").fill("谢谢")
                    async with page.expect_response(
                        lambda response: response.url.endswith("/chat") and response.status == 200,
                    ) as pending:
                        await page.locator("#send").click()
                    response = await pending.value
                    sse = await response.text()
                    assert cid in sse and "event: done" in sse
                    await page.wait_for_function(
                        "!document.querySelector('#send').textContent.includes('正在')"
                    )
                    assert await page.locator("#notice").is_hidden()
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                await page.screenshot(path=str(ROOT / f"dev-notes/ch07-offline-chat-{width}.png"))
                if width < 700:
                    await page.locator("#conversation-toggle").click()
                    await page.wait_for_function(
                        "getComputedStyle(document.querySelector('.sidebar')).transform === 'none'"
                    )
                    await page.screenshot(
                        path=str(ROOT / f"dev-notes/ch07-offline-sidebar-{width}.png")
                    )
                await page.locator("#new-conversation").click()
                assert await page.locator("#welcome").is_visible()
                assert await page.locator("#conversation-list button").count() >= 2
                rows.append(
                    {
                        "width": width,
                        "originals_match": True,
                        "summary_badge": True,
                        "no_overflow": True,
                        "new_keeps_old": True,
                    }
                )
                await page.close()
        finally:
            await browser.close()
    output = {
        "passed": not errors,
        "model_mode": "offline",
        "full_model_acceptance_complete": complete,
        "errors": errors,
        "views": rows,
    }
    (ROOT / "dev-notes/ch07-browser-results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    assert not errors
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
