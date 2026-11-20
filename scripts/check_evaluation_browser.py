"""验证RAG评估入口、真实报告、筛选、证据与可选的完整真实重新评估。"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from playwright.async_api import async_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


async def main(args):
    """在桌面与手机核验实际页面；--run-evaluation 会启动完整80条真实评估。"""
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
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, executable_path=executable)
        try:
            chat = await browser.new_page(viewport={"width": 1440, "height": 1050})
            await chat.goto(args.base_url)
            await chat.locator("#message").fill("保留尚未发送的聊天草稿")
            async with chat.expect_popup() as opened:
                await chat.get_by_role("link", name="RAG 评估").click()
            page = await opened.value
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            await expect(page.locator("#strategy-rows tr")).to_have_count(4)
            assert await chat.locator("#message").input_value() == "保留尚未发送的聊天草稿"
            source_report = (await page.request.get(args.base_url + "/api/evaluations")).json()
            source_report = await source_report
            assert source_report["report"]["row_count"] == 80
            await expect(page.locator("#report-cases")).to_have_text("20 题")
            expected_mrr = f"{source_report['strategies']['hybrid_rerank']['mrr']:.4f}"
            await expect(
                page.locator("tr[data-strategy=hybrid_rerank] td[data-metric=mrr] .metric-value")
            ).to_have_text(expected_mrr)
            if source_report["stale"]:
                await expect(page.locator("#freshness")).to_contain_text("历史报告")
            report.append(
                {
                    "entry_preserves_chat": True,
                    "saved_report": source_report["report"],
                    "historical_snapshot_marked": source_report["stale"],
                }
            )

            await page.get_by_label("问题类型", exact=True).select_option("model")
            await page.get_by_label("难度", exact=True).select_option("hard")
            await expect(page.locator("#selection-count")).to_contain_text("1 道题")
            await page.get_by_label("召回深度 K").select_option("1")
            await expect(page.locator("#recall-heading")).to_have_text("Recall@1")
            card = page.locator(".case-card[data-strategy=hybrid_rerank]")
            await card.locator(".citation").first.click()
            await expect(page.locator("#evaluation-source")).to_be_visible()
            await expect(page.locator("#source-text")).not_to_be_empty()
            await page.get_by_role("button", name="关闭 ×", exact=True).click()
            await card.locator("details").first.locator("summary").click()
            await expect(card.locator(".rank-list button").first).to_be_visible()
            await card.locator(".rank-list button").first.click()
            await expect(page.locator("#source-origin")).not_to_be_empty()
            await page.screenshot(path=str(ROOT / "dev-notes/ch04-evaluation-source.png"))
            await page.get_by_role("button", name="关闭 ×", exact=True).click()
            await card.locator("details").last.locator("summary").click()
            assert await card.locator(".claim-list li").count() > 0
            report.append(
                {
                    "combined_filters": True,
                    "recall_selector": True,
                    "answer_citation_and_ranked_snapshot": True,
                    "claim_reasons": True,
                }
            )

            await page.get_by_label("问题类型", exact=True).select_option("unknown")
            await page.get_by_label("难度", exact=True).select_option("")
            await page.get_by_label("评估题目", exact=True).select_option("unknown-subsidy")
            await expect(page.locator(".case-card .tag.refused")).to_have_count(4)
            for value in await page.locator(
                "td[data-metric=faithfulness] .metric-value"
            ).all_text_contents():
                assert value == "N/A"
            await page.get_by_label("问题类型", exact=True).select_option("direct")
            await page.get_by_label("难度", exact=True).select_option("hard")
            await expect(page.locator("#selection-count")).to_contain_text("0 道题")
            await expect(page.locator("#case-reference")).to_contain_text("没有匹配题目")
            await page.get_by_role("button", name="清除筛选").click()
            await expect(page.locator("#selection-count")).to_contain_text("20 道题")
            report.append({"refusal_faithfulness_is_na": True, "empty_bucket": True})
            for width, height in [(1440, 1050), (390, 844), (320, 740)]:
                await page.set_viewport_size({"width": width, "height": height})
                assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                await page.screenshot(
                    path=str(ROOT / f"dev-notes/ch04-evaluation-{width}.png"), full_page=True
                )
                report.append({"width": width, "no_horizontal_page_overflow": True})

            if args.run_evaluation:
                await page.set_viewport_size({"width": 1440, "height": 1050})
                await page.get_by_role("button", name="重新评估", exact=True).click()
                await expect(page.locator("#run-evaluation")).to_be_disabled()
                await expect(page.locator("#job-title")).to_have_text("评估进行中")
                duplicate = await page.request.post(args.base_url + "/api/evaluations/run")
                assert duplicate.status == 409
                await page.reload()
                await expect(page.locator("#job-title")).to_have_text("评估进行中")
                deadline = time.monotonic() + 900
                seen_progress = False
                while time.monotonic() < deadline:
                    state = await page.request.get(args.base_url + "/api/evaluations/status")
                    job = (await state.json())["job"]
                    if 0 < job["completed"] < job["total"] and not seen_progress:
                        seen_progress = True
                        await page.screenshot(
                            path=str(ROOT / "dev-notes/ch04-evaluation-progress.png")
                        )
                    if job["status"] != "running":
                        assert job["status"] == "completed", job
                        break
                    await asyncio.sleep(2)
                else:
                    raise AssertionError("真实评估超时")
                await expect(page.locator("#job-title")).to_have_text("评估已完成", timeout=15000)
                await expect(page.locator("#freshness")).to_contain_text(
                    "与当前知识库、标注集一致", timeout=15000
                )
                latest_response = await page.request.get(args.base_url + "/api/evaluations")
                latest = await latest_response.json()
                assert latest["report"]["id"] == job["id"]
                assert latest["report"]["row_count"] == job["total"]
                assert seen_progress
                report.append(
                    {
                        "real_rerun": True,
                        "job": job,
                        "refresh_preserves_running_job": True,
                        "duplicate_run_rejected": True,
                        "new_report": latest["report"],
                        "strategies": latest["strategies"],
                    }
                )
                await page.screenshot(
                    path=str(ROOT / "dev-notes/ch04-evaluation-completed.png"), full_page=True
                )
            assert not errors, errors
        finally:
            await browser.close()
    (ROOT / "dev-notes/ch04-evaluation-browser-results.json").write_text(
        json.dumps({"cases": report, "page_errors": errors}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("RAG评估浏览器验收通过：入口、真实指标、筛选、证据、桌面/手机及所选重新评估流程。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8004")
    parser.add_argument(
        "--run-evaluation", action="store_true", help="会调用真实模型，完整运行四策略"
    )
    asyncio.run(main(parser.parse_args()))
