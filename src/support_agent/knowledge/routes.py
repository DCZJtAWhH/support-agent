"""knowledge 的 HTTP 入口；协议转换与业务能力分离。"""

from html import escape

from fastapi import HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .uploads import MAX_UPLOAD_BYTES, UploadIndexPending, UploadRejected


def mount_routes(app, web_directory):
    """挂载既有路径和响应契约，页面资源路径由启动装配传入。"""

    @app.get("/knowledge/chunks/{chunk_id}")
    async def knowledge_chunk(chunk_id: int):
        """展示权威来源原文与章节路径；过期或失效块不再作为有效原文返回。"""
        row = await app.state.knowledge_store.get(chunk_id)
        if row is None or not row.is_active:
            raise HTTPException(404, detail="来源知识已失效或不存在。")
        return {
            "chunk_id": row.id,
            "source_ref": row.source_ref,
            "section_path": row.section_path or row.questions,
            "question": row.questions,
            "answer": row.answer,
            "content_hash": row.content_hash,
        }

    @app.post("/knowledge/upload", status_code=201)
    async def upload_knowledge(file: UploadFile):
        """接收知识文件并同步建立索引；完整处理后才报告可供检索。"""
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            return await app.state.knowledge_uploads.import_document(file.filename or "", content)
        except UploadRejected as exc:
            raise HTTPException(exc.status_code, detail=str(exc)) from exc
        except UploadIndexPending as exc:
            return JSONResponse(status_code=503, content={"detail": exc.result})
        finally:
            await file.close()

    @app.get("/knowledge/chunks/{chunk_id}/original", response_class=HTMLResponse)
    async def original_document(chunk_id: int, version: str = ""):
        """按来源重组已入库原文并定位章节；旧引用链接明确提示内容版本变化。"""
        store = app.state.knowledge_store
        selected = await store.get(chunk_id)
        if selected is None or not selected.is_active:
            raise HTTPException(404, detail="来源知识已失效或不存在；请在聊天引用中查看原快照。")
        chunks = [r for r in await store.active() if r.source_ref == selected.source_ref]
        sections = "".join(
            f'<section id="chunk-{row.id}"><h2>{escape(row.section_path or row.questions)}</h2>'
            f"<pre>{escape(row.answer)}</pre></section>"
            for row in chunks
        )
        note = (
            "版本已更新，下方为当前入库原文；回答时的原文保留在聊天引用面板。"
            if version and version != selected.content_hash
            else "当前入库原文 · 按知识块展示所在章节"
        )
        return HTMLResponse(
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>知识来源原文</title><style>"
            "body{font:16px/1.9 system-ui;max-width:860px;margin:40px auto;padding:0 24px;"
            "color:#22332e}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}"
            "h1{font-size:24px;overflow-wrap:anywhere}h2{font-size:19px}"
            "section{padding:24px;border-bottom:1px solid #ddd;scroll-margin-top:24px}"
            "section:target{background:#edf5f0}p{color:#64726e}</style>"
            f"<h1>{escape(selected.source_ref)}</h1><p>{note}</p>{sections}</html>"
        )
