import asyncio
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


def test_sidebar_switches_reload_originals_and_silently_survives_list_failure():
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.path = self.path.removeprefix("/static")
            super().do_GET()

        def log_message(self, *args):
            pass

    async def scenario(base):
        (ROOT / "data/reports").mkdir(parents=True, exist_ok=True)
        async with async_playwright() as p:
            cached = sorted(
                (Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright").glob(
                    "chromium-*/chrome-win*/chrome.exe"
                )
            )
            browser = await p.chromium.launch(executable_path=str(cached[-1]) if cached else None)
            try:
                for width in (1440, 390, 320):
                    page = await browser.new_page(viewport={"width": width, "height": 900})
                    sent, errors = [], []
                    page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))

                    async def listed(route):
                        await route.fulfill(
                            json=[
                                {"id": "new-id", "preview": "第二个问题", "has_summary": False},
                                {
                                    "id": "old-id",
                                    "preview": "订单1001要求换货",
                                    "has_summary": True,
                                },
                            ]
                        )

                    async def history(route):
                        old = "old-id" in route.request.url
                        await route.fulfill(
                            json=[
                                {
                                    "role": "user",
                                    "content": "订单1001要求换货\n原话一字不改"
                                    if old
                                    else "第二个问题",
                                    "state": "completed",
                                },
                                {
                                    "role": "assistant",
                                    "content": "完整回复" + "甲" * 100,
                                    "state": "completed",
                                },
                            ]
                        )

                    async def chat(route, request, sent=sent):
                        sent.append(route.request.post_data_json)
                        await route.fulfill(
                            content_type="text/event-stream",
                            body=(
                                'event: meta\ndata: {"conversation_id":"new-id",'
                                '"request_id":"r"}\n\n'
                                'event: token\ndata: {"text":"接着聊"}\n\n'
                                'event: done\ndata: {"conversation_id":"new-id",'
                                '"request_id":"r"}\n\n'
                            ),
                        )

                    await page.route("**/api/conversations", listed)
                    await page.route("**/api/conversations/*/messages", history)
                    await page.route("**/chat", chat)
                    await page.goto(base)
                    await page.locator("#conversation-list button").first.wait_for(
                        state="attached", timeout=3000
                    )
                    if width < 700:
                        await page.get_by_role("button", name="历史会话", exact=True).click()
                    await page.get_by_role("button", name="订单1001要求换货", exact=False).click()
                    assert (
                        await page.locator(".message.user .reply").inner_text()
                        == "订单1001要求换货\n原话一字不改"
                    )
                    assert len(await page.locator(".message.assistant .reply").inner_text()) == 104
                    await page.locator(
                        "#mobile-new" if width < 700 else "#new-conversation"
                    ).click()
                    assert await page.locator("#conversation-list button").count() == 2
                    if width < 700:
                        await page.get_by_role("button", name="历史会话", exact=True).click()
                    await page.get_by_role("button", name="第二个问题", exact=True).click()
                    await (
                        page.locator(".message.user .reply")
                        .filter(has_text="第二个问题")
                        .wait_for()
                    )
                    await page.locator("#message").fill("继续")
                    await page.locator("#send").click()
                    await page.wait_for_function(
                        "!document.querySelector('#send').textContent.includes('正在')"
                    )
                    assert sent[-1]["conversation_id"] == "new-id"
                    assert await page.locator("#notice").is_hidden()
                    assert (
                        await page.locator(".message.assistant .reply").last.inner_text()
                        == "接着聊"
                    )
                    assert not errors
                    assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    await page.screenshot(
                        path=str(ROOT / f"data/reports/context-sidebar-{width}.png")
                    )
                    await page.unroute("**/api/conversations", listed)
                    await page.route("**/api/conversations", lambda route: route.abort())
                    await page.reload()
                    await page.locator("#message").fill("你好")
                    assert await page.locator("#send").is_enabled()
                    assert await page.locator("#notice").is_hidden()
                    await page.close()
            finally:
                await browser.close()

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "src/support_agent/web"))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        asyncio.run(scenario(f"http://127.0.0.1:{server.server_port}"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
