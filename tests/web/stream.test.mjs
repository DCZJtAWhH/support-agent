import assert from "node:assert/strict";
import { test } from "node:test";
import { createEventParser, readChatStream } from "../../src/support_agent/web/stream.mjs";

const id = "ff1d90ea-5b45-423e-b02b-e0d375d85ce3";
const meta = `event: meta\ndata: {"conversation_id":"${id}"}\n\n`;
const token = `event: token\ndata: ${JSON.stringify({ text: "您好🙂\n订单 A123" })}\n\n`;
const done = `event: done\ndata: {"conversation_id":"${id}"}\n\n`;

function responseFor(text, size = 1) {
  const bytes = new TextEncoder().encode(text);
  return new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < bytes.length; i += size) controller.enqueue(bytes.slice(i, i + size));
      controller.close();
    },
  }), { headers: { "Content-Type": "text/event-stream; charset=utf-8" } });
}

for (const ending of ["\n", "\r\n", "\r"]) {
  test(`decodes split UTF-8 and SSE boundaries with ${JSON.stringify(ending)}`, async () => {
    const events = [];
    const result = await readChatStream(
      responseFor((meta + token + done).replaceAll("\n", ending)),
      (...event) => events.push(event),
    );
    assert.equal(result, id);
    assert.deepEqual(events.map(([event]) => event), ["meta", "token", "done"]);
    assert.equal(events[1][1].text, "您好🙂\n订单 A123");
  });
}

test("parses merged events, comments and multiline data", async () => {
  const events = [];
  const parser = createEventParser((...event) => events.push(event));
  parser.push(': heartbeat\r\nevent: token\r\ndata: {\r\ndata: "text": "中文"}\r\n\r');
  parser.push('\nevent: done\ndata: {}\n\n');
  assert.deepEqual(events, [["token", { text: "中文" }], ["done", {}]]);
  const merged = [];
  await readChatStream(responseFor(meta + token + token + done, 4096), (...event) => merged.push(event));
  assert.equal(merged.length, 4);
});

test("delivers first token before the stream ends", async () => {
  let controller;
  const events = [];
  const response = new Response(new ReadableStream({ start(value) { controller = value; } }), {
    headers: { "Content-Type": "text/event-stream" },
  });
  const reading = readChatStream(response, (...event) => events.push(event));
  const encoder = new TextEncoder();
  controller.enqueue(encoder.encode(meta + token));
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(events.map(([event]) => event), ["meta", "token"]);
  controller.enqueue(encoder.encode(done));
  controller.close();
  assert.equal(await reading, id);
});

for (const [name, body] of [
  ["EOF without done", meta + token],
  ["unterminated done event", meta + token + done.trimEnd()],
  ["empty reply", meta + done],
  ["token before meta", token + done],
  ["mismatched conversation", meta + token + done.replace(id, "other")],
  ["invalid JSON", meta + "event: token\ndata: broken\n\n"],
  ["invalid token", meta + "event: token\ndata: {}\n\n"],
  ["duplicate meta", meta + meta + token + done],
  ["events after done", meta + token + done + token],
  ["upstream error", meta + token + 'event: error\ndata: {"message":"模型暂不可用"}\n\n'],
]) {
  test(`rejects ${name} and releases the reader`, async () => {
    const response = responseFor(body);
    await assert.rejects(readChatStream(response, () => {}));
    assert.equal(response.body.locked, false);
  });
}

for (const status of [404, 409, 413, 422, 500]) {
  test(`reports HTTP ${status} as an error`, async () => {
    await assert.rejects(readChatStream(new Response("failure", { status }), () => {}), (error) => error.status === status);
  });
}

test("rejects non-SSE responses", async () => {
  await assert.rejects(readChatStream(new Response("a proxy HTML error"), () => {}));
});

test("aborted transport rejects an incomplete turn", async () => {
  const response = new Response(new ReadableStream({
    start(controller) { controller.error(new Error("disconnected")); },
  }), { headers: { "Content-Type": "text/event-stream" } });
  await assert.rejects(readChatStream(response, () => {}), /disconnected/);
  assert.equal(response.body.locked, false);
});
test("actions remain separate and are delivered before completion", async () => {
  const events = [];
  const body = 'event: meta\ndata: {"conversation_id":"c"}\n\n' +
    'event: token\ndata: {"text":"抱歉"}\n\n' +
    'event: actions\ndata: {"actions":["handoff","create_ticket"]}\n\n' +
    'event: done\ndata: {"conversation_id":"c"}\n\n';
  await readChatStream(new Response(body, {headers: {"content-type": "text/event-stream"}}),
    (name, data) => events.push([name, data]));
  assert.deepEqual(events.find(([name]) => name === "actions")[1].actions,
    ["handoff", "create_ticket"]);
});
