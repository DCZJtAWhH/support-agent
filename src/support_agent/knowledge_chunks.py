"""将 Markdown 按章节、段落、完整句和表格行切分成可检索知识。"""

import re
from dataclasses import dataclass

SENTENCE_BREAK = r"""(?<=[。！？!?])(?=[^”’"'])|(?<=[。！？!?][”’"'])|(?<=\.)(?=\s|$)"""


@dataclass
class KnowledgeDraft:
    """表示尚未持久化的知识块，元数据不参与嵌入。"""

    category: str
    questions: list[str]
    answer: str
    section_path: str = ""
    content_type: str = "policy"
    is_key_clause: bool = False

    def embedding_text(self) -> str:
        """仅拼接分类、问法与答案，保持建库和更新的嵌入格式一致。"""
        return (
            f"分类：{self.category}\n问题：" + "\n".join(self.questions) + f"\n答案：{self.answer}"
        )


def _units(text: str, limit: int, level: int = 0) -> list[str]:
    """递归拆解超长段落；最后一级仅允许完整句，不能满足预算时明确报错。"""
    separators = [r"\n\s*\n", SENTENCE_BREAK]
    if len(text) <= limit:
        return [text]
    if level >= len(separators):
        raise ValueError("完整句或完整行超过块大小，请修正文档或调大 max_chars。")
    parts = [p.strip() for p in re.split(separators[level], text) if p.strip()]
    return [unit for part in parts for unit in _units(part, limit, level + 1)]


def _overlap(text: str, target: int, capacity: int) -> str:
    """按目标选末尾完整句；目标落在句中时可保留整句，但不得超出新块容量。"""
    if target <= 0 or capacity <= 0:
        return ""
    sentences = [part for part in re.split(SENTENCE_BREAK, text) if part.strip()]
    result = ""
    for sentence in reversed(sentences):
        if len(sentence) + len(result) > target:
            if not result and len(sentence.strip()) <= capacity:
                result = sentence
            break
        result = sentence + result
    return result.strip()


def _pack(text: str, limit: int, overlap: int) -> list[str]:
    """将递归切出的完整单元装入块，并在可容纳时追加完整句重叠。"""
    result, current = [], ""
    for unit in _units(text.strip(), limit):
        separator = "\n" if current else ""
        if len(current) + len(separator) + len(unit) > limit:
            result.append(current)
            current = _overlap(current, overlap, limit - len(unit) - 1)
            separator = "\n" if current else ""
        current += separator + unit
    if current:
        result.append(current)
    return result


def _body_chunks(body: str, limit: int, overlap: int) -> list[tuple[str, bool]]:
    """分离普通文本与 Markdown 表格，表格按完整行装块并重复两行表头。"""
    lines, result, prose = body.splitlines(), [], []
    index = 0
    fenced = False
    while index < len(lines):
        line = lines[index]
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
        is_table = (
            not fenced
            and index + 1 < len(lines)
            and "|" in line
            and re.match(r"^\s*\|?\s*:?-{3,}", lines[index + 1])
        )
        if not is_table:
            prose.append(line)
            index += 1
            continue
        if "\n".join(prose).strip():
            result.extend((p, False) for p in _pack("\n".join(prose), limit, overlap))
        prose = []
        header = line + "\n" + lines[index + 1]
        index += 2
        table = header
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            row = lines[index]
            if len(header) + len(row) + 1 > limit:
                raise ValueError("表格表头与完整行超过块大小，请修正文档或调大 max_chars。")
            if len(table) + len(row) + 1 > limit:
                result.append((table, True))
                previous_row = table.splitlines()[-1]
                table = header
                if overlap and len(previous_row) <= overlap:
                    if len(header) + len(previous_row) + len(row) + 2 <= limit:
                        table += "\n" + previous_row
            table += "\n" + row
            index += 1
        if len(table) > limit:
            raise ValueError("表格表头超过块大小。")
        result.append((table, True))
    if "\n".join(prose).strip():
        result.extend((p, False) for p in _pack("\n".join(prose), limit, overlap))
    return result


def split_markdown(
    text: str,
    *,
    title: str = "知识文档",
    max_chars: int = 900,
    overlap_chars: int = 100,
) -> list[KnowledgeDraft]:
    """按 ATX/Setext 标题层级切分；显式注释声明 FAQ 与关键条款，超长原子内容报错。"""
    if max_chars < 20 or overlap_chars < 0:
        raise ValueError("max_chars 至少为20，overlap_chars 不能为负。")
    content_type = "faq" if re.search(r"<!--\s*content_type:\s*faq\s*-->", text) else "policy"
    sections, headings, body = [], [], []
    fenced = False
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
        heading = None if fenced else re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        setext = (
            not fenced
            and line.strip()
            and index + 1 < len(lines)
            and re.fullmatch(r"\s*(=+|-+)\s*", lines[index + 1])
        )
        if heading or setext:
            if "\n".join(body).strip():
                sections.append(([h[1] for h in headings], "\n".join(body)))
            body = []
            level = len(heading[1]) if heading else (1 if "=" in lines[index + 1] else 2)
            label = heading[2] if heading else line.strip()
            headings = [h for h in headings if h[0] < level] + [(level, label)]
            index += 1 if heading else 2
        else:
            body.append(line)
            index += 1
    if "\n".join(body).strip():
        sections.append(([h[1] for h in headings], "\n".join(body)))
    result = []
    for path, raw in sections:
        key = bool(re.search(r"<!--\s*key_clause\s*-->", raw))
        clean = re.sub(r"<!--.*?-->", "", raw, flags=re.S).strip()
        path = path or [title]
        for answer, table in _body_chunks(clean, max_chars, overlap_chars):
            result.append(
                KnowledgeDraft(
                    category=" / ".join(path[:-1]) or title,
                    questions=[path[-1]],
                    answer=answer,
                    section_path=" / ".join(path),
                    content_type="table" if table else content_type,
                    is_key_clause=key,
                )
            )
    return result
