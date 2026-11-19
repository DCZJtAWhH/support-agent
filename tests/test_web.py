import asyncio


def test_chat_page_and_its_assets_are_served_without_calling_the_model(harness):
    async def scenario():
        async with harness() as (client, calls, _):
            page = await client.get("/")
            assert page.status_code == 200
            assert page.headers["content-type"].startswith("text/html")
            assert '<html lang="zh-CN">' in page.text
            assert "客服对话" in page.text
            assert "/static/chat.mjs" in page.text
            for path, content_type in (
                ("/static/chat.css", "text/css"),
                ("/static/chat.mjs", "javascript"),
                ("/static/stream.mjs", "javascript"),
            ):
                asset = await client.get(path)
                assert asset.status_code == 200
                assert content_type in asset.headers["content-type"]
            assert calls == []
            assert (await client.get("/docs")).status_code == 200
            assert (await client.get("/static/../config.py")).status_code == 404

    asyncio.run(scenario())
