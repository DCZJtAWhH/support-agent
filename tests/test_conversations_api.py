import asyncio
from uuid import uuid4

from conftest import sse_events


def test_conversations_are_newest_first_and_reload_verbatim_for_continuation(harness):
    async def scenario():
        async with harness() as (client, _, app):
            first = await client.post("/chat", json={"message": "你好"})
            cid = sse_events(first)[0][1]["conversation_id"]
            second = await client.post("/chat", json={"message": "谢谢"})
            second_cid = sse_events(second)[0][1]["conversation_id"]
            repo = app.state.repository
            # 外部用户会话不允许通过只读接口访问或续聊。
            from support_agent.models import Conversation

            private = str(uuid4())
            async with repo.database.sessions.begin() as session:
                session.add(Conversation(id=private, user_id="another-user"))
            listed = await client.get("/api/conversations")
            assert listed.status_code == 200
            assert [c["id"] for c in listed.json()] == [second_cid, cid]
            assert listed.json()[1]["preview"] == "你好"
            assert listed.json()[1]["has_summary"] is False
            history = await client.get(f"/api/conversations/{cid}/messages")
            assert [m["content"] for m in history.json()] == [
                "你好",
                "您好，我是本店智能客服，可以帮您查询订单、物流、商品信息和售后政策。",
            ]
            assert (await client.get(f"/api/conversations/{private}/messages")).status_code == 404
            assert (
                await client.post("/chat", json={"message": "你好", "conversation_id": private})
            ).status_code == 404
            follow = await client.post("/chat", json={"message": "谢谢", "conversation_id": cid})
            assert sse_events(follow)[-1][0] == "done"
            assert len((await client.get(f"/api/conversations/{cid}/messages")).json()) == 4

    asyncio.run(scenario())
