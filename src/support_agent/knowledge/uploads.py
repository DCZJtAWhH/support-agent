"""把用户上传的文本复用到权威知识分块与混合索引链路。"""

import asyncio
from pathlib import PurePath

from support_agent.knowledge.chunks import split_markdown
from support_agent.knowledge.store import synchronize

MAX_UPLOAD_BYTES = 1024 * 1024
MAX_UPLOAD_CHUNKS = 200


class UploadRejected(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        """携带可公开的校验提示与 HTTP 状态，保证拒绝发生在原文写入之前。"""
        super().__init__(message)
        self.status_code = status_code


class UploadIndexPending(RuntimeError):
    def __init__(self, result: dict):
        """保留已提交原文的身份与知识主键，允许重试且不暴露提供方异常。"""
        super().__init__("知识原文已保存，索引尚未完成。")
        self.result = result


def parse_upload(filename: str, content: bytes):
    """校验名称、大小、UTF-8 和完整分块，拒绝不支持的内容后才允许入库。"""
    if (
        not filename
        or len(filename) > 180
        or any(char in "/\\" or ord(char) < 32 for char in filename)
    ):
        raise UploadRejected("文件名无效，请使用不含目录的文件名，且不超过180字。")
    if PurePath(filename).suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise UploadRejected("文件格式不支持，请上传 Markdown（.md/.markdown）或 TXT 文件。")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadRejected("每个文件最大为 1 MiB，请拆分后上传。", 413)
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UploadRejected("文件须为 UTF-8 编码，请另存为 UTF-8 后上传。") from exc
    if any((ord(char) < 32 and char not in "\r\n\t") or ord(char) == 127 for char in text):
        raise UploadRejected("文件包含非文本字符，请转换为 UTF-8 文本后上传。")
    try:
        drafts = split_markdown(text, title=PurePath(filename).stem)
    except ValueError as exc:
        raise UploadRejected(str(exc)) from exc
    if not drafts:
        raise UploadRejected("文件没有可导入的正文，请添加标题下的知识内容。")
    if len(drafts) > MAX_UPLOAD_CHUNKS:
        raise UploadRejected("单个文件最多导入200个知识块，请按主题拆分文件。")
    if any(len(d.category) > 255 or len(d.section_path) > 512 for d in drafts):
        raise UploadRejected("文档标题或标题层级过长，请缩短标题后上传。")
    return drafts


class KnowledgeUploader:
    def __init__(self, store, embeddings, vectors):
        """复用应用检索依赖；串行提交上传，避免同名文件的原文和索引交错。"""
        self.store, self.embeddings, self.vectors = store, embeddings, vectors
        self.lock = asyncio.Lock()

    async def import_document(self, filename: str, content: bytes) -> dict:
        """先解析完整正文，再提交原文并建立两路索引；同名上传复用知识主键。"""
        drafts = await asyncio.to_thread(parse_upload, filename, content)
        source = "upload:" + filename
        async with self.lock:
            ids = await self.store.replace_source(source, drafts)
            try:
                await synchronize(self.store, self.embeddings, self.vectors, source_ref=source)
            except Exception as exc:
                raise UploadIndexPending(
                    {
                        "code": "index_pending",
                        "status": "pending",
                        "filename": filename,
                        "source_ref": source,
                        "chunk_count": len(ids),
                        "chunk_ids": ids,
                        "message": "知识原文已保存，但检索索引尚未完成。"
                        "请稍后重试导入，无需重复添加文件。",
                    }
                ) from exc
        return {
            "filename": filename,
            "source_ref": source,
            "status": "ready",
            "chunk_count": len(ids),
            "chunk_ids": ids,
        }
