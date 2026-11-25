"""验证真实两轮评估在后台画出趋势和逐轮表格，不使用模拟分数。"""

import asyncio
import json
import os
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


async def main():
    """读取真实接口核对时间顺序、三项指标、折线与移动布局，保存截图。"""
    async with async_playwright() as driver:
        cached = sorted(
            (Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright").glob(
                "chromium-*/chrome-win*/chrome.exe"
            )
        )
        browser = await driver.chromium.launch(
            headless=True, executable_path=str(cached[-1]) if cached else None
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = await page.request.get("http://127.0.0.1:8010/api/evaluations/runs")
        rows = await response.json()
        assert len(rows) >= 2
        assert len({r["dataset_hash"] for r in rows}) == 1
        assert len({r["corpus_hash"] for r in rows}) == 1
        await page.goto("http://127.0.0.1:8010/reviews")
        await expect(page.locator("#run-list tbody tr")).to_have_count(len(rows))
        await expect(page.locator(".trend-svg")).to_be_visible()
        for index, row in enumerate(rows):
            cells = page.locator("#run-list tbody tr").nth(index).locator("td")
            for cell_index, key in enumerate(("recall@5", "mrr", "faithfulness"), 2):
                value = row["metrics"]["hybrid_rerank"][key]
                await expect(cells.nth(cell_index)).to_have_text(
                    "N/A" if value is None else f"{value:.4f}"
                )
                if index:
                    previous = rows[index - 1]["metrics"]["hybrid_rerank"][key]
                    if value is not None and previous is not None and value < previous:
                        await expect(cells.nth(cell_index)).to_have_class("decline")
        for width, height in [(1440, 1000), (390, 844), (320, 740)]:
            await page.set_viewport_size({"width": width, "height": height})
            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            await page.locator("#trend-title").scroll_into_view_if_needed()
            await page.screenshot(
                path=str(ROOT / f"dev-notes/ch09-trends-{width}.png"), full_page=True
            )
        assert not errors, errors
        (ROOT / "dev-notes/ch09-trends-verification.json").write_text(
            json.dumps(
                {
                    "real_runs": len(rows),
                    "metrics_match_api": True,
                    "responsive_widths": [1440, 390, 320],
                    "page_errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
