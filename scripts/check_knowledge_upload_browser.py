"""在真实浏览器验证知识上传、多文件、反馈状态和导入后引用问答。"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


async def main(args):
    """桌面和手机使用真实服务导入 Markdown/TXT，再验证失败重试与来源原文。"""
    executable = None
    if os.name == "nt":
        cached = sorted(
            (Path(os.environ["LOCALAPPDATA"]) / "ms-playwright").glob(
                "chromium-*/chrome-win*/chrome.exe",
            )
        )
        if cached:
            executable = str(cached[-1])
    report, errors = [], []
    files = [
        {
            "name": "上传演示手册.md",
            "mimeType": "text/markdown",
            "buffer": (
                "# 上传验收（演示）\n## UP-680容量是多少\nUP-680演示水杯的标称容量为680毫升。"
                "\n## UP-680清洗方式\n请用温水手洗。"
            ).encode(),
        },
        {
            "name": "上传文本演示.txt",
            "mimeType": "text/plain",
            "buffer": "演示商品LC-208的包装包含一份说明书。".encode(),
        },
    ]
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, executable_path=executable)
        try:
            page = await browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            for width, height in [(1440, 1000), (390, 844)]:
                await page.set_viewport_size({"width": width, "height": height})
                await page.goto(args.base_url)
                await page.get_by_role("button", name="上传知识").click()
                dialog = page.locator("#knowledge-upload-dialog")
                await expect(dialog).to_be_visible()
                await page.get_by_label("选择知识文件").set_input_files(files)
                await page.get_by_role("button", name="开始导入", exact=True).click()
                await expect(page.locator("#knowledge-upload-close")).to_be_disabled()
                await expect(
                    page.locator("#knowledge-upload-list li[data-state=ready]")
                ).to_have_count(
                    2,
                    timeout=180000,
                )
                await expect(page.locator("#knowledge-upload-summary")).to_contain_text(
                    "2 个文件已加入知识库"
                )
                assert await dialog.evaluate(
                    "element => element.scrollWidth <= element.clientWidth"
                )
                await page.screenshot(path=str(ROOT / f"dev-notes/ch04-upload-{width}.png"))
                await page.get_by_role("button", name="返回聊天", exact=True).click()
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                report.append(
                    {
                        "width": width,
                        "multi_file_import": True,
                        "markdown_and_txt": True,
                        "no_horizontal_overflow": True,
                    }
                )

            await page.get_by_role("button", name="上传知识").click()
            await page.get_by_role("button", name="清空列表").click()
            await page.get_by_label("选择知识文件").set_input_files(
                [
                    {
                        "name": "unsupported.pdf",
                        "mimeType": "application/pdf",
                        "buffer": b"%PDF-test",
                    },
                ]
            )
            await expect(
                page.locator("#knowledge-upload-list li[data-state=invalid]")
            ).to_have_count(1)
            await expect(page.get_by_role("button", name="开始导入", exact=True)).to_be_disabled()
            report.append({"unsupported_file_rejected_in_browser": True})
            await page.get_by_role("button", name="清空列表").click()
            await page.get_by_label("选择知识文件").set_input_files(
                [
                    {"name": "bad-encoding.md", "mimeType": "text/markdown", "buffer": b"\xff\xfe"},
                ]
            )
            await page.get_by_role("button", name="开始导入", exact=True).click()
            await expect(page.locator("#knowledge-upload-list")).to_contain_text(
                "UTF-8", timeout=30000
            )
            report.append({"invalid_encoding_rejected_by_real_server": True})
            await page.get_by_role("button", name="清空列表").click()

            attempts = 0

            async def fail_first_request(route):
                """仅首个请求模拟503以验证UI重试，后续请求放行到真实服务。"""
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    await route.fulfill(
                        status=503,
                        json={
                            "detail": {
                                "code": "index_pending",
                                "message": "知识原文已保存，索引尚未完成，请重试。",
                            }
                        },
                    )
                else:
                    await route.continue_()

            await page.route("**/knowledge/upload", fail_first_request)
            await page.get_by_label("选择知识文件").set_input_files([files[0]])
            await page.get_by_role("button", name="开始导入", exact=True).click()
            await expect(
                page.locator("#knowledge-upload-list li[data-state=failed]")
            ).to_have_count(1)
            await page.get_by_role("button", name="重试导入", exact=True).click()
            await expect(page.locator("#knowledge-upload-list li[data-state=ready]")).to_have_count(
                1,
                timeout=180000,
            )
            await page.unroute("**/knowledge/upload", fail_first_request)
            report.append({"simulated_503_then_real_retry": True, "attempts": attempts})
            await page.get_by_role("button", name="返回聊天", exact=True).click()
            await page.locator("#message").fill("UP-680容量是多少")
            await page.locator("#send").click()
            await page.locator(".assistant .answer-feedback").wait_for(timeout=180000)
            answer = await page.locator(".assistant .reply").inner_text()
            assert "680" in answer, answer
            await page.locator(".assistant .citation").first.click()
            await expect(page.locator("#source-dialog .source-path")).to_contain_text("UP-680")
            await expect(page.locator("#source-dialog .source-origin").first).to_contain_text(
                "upload:"
            )
            await expect(page.locator("#source-dialog .source-text")).to_contain_text("680毫升")
            await page.screenshot(path=str(ROOT / "dev-notes/ch04-upload-citation.png"))
            report.append(
                {"uploaded_knowledge_answer": answer, "citation_matches_uploaded_source": True}
            )
            assert not errors, errors
        finally:
            await browser.close()
    (ROOT / "dev-notes/ch04-upload-browser-results.json").write_text(
        json.dumps({"cases": report, "page_errors": errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("知识上传浏览器验收通过：桌面/手机、多文件、格式与编码、失败重试、真实知识问答和引用。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    asyncio.run(main(parser.parse_args()))
