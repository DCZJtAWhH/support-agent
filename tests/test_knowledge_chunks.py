import pytest

from support_agent.knowledge.chunks import split_markdown


def test_overlap_uses_whole_sentence_when_target_falls_inside_it():
    chunks = split_markdown(
        "# 政策\n" + "甲" * 29 + "。" + "乙" * 29 + "。" + "丙" * 29 + "。",
        max_chars=70,
        overlap_chars=10,
    )
    assert chunks[1].answer.startswith("乙" * 29 + "。")
    assert len(chunks[1].answer) <= 70


def test_policy_chunks_keep_heading_paths_complete_sentences_and_overlap():
    text = "# 售后手册\n## 退货政策\n" + "商品应保持完好。退货需保留包装。签收后七天内申请。" * 8
    chunks = split_markdown(text, max_chars=80, overlap_chars=20)
    assert len(chunks) > 2
    assert all(c.category == "售后手册" for c in chunks)
    assert all(c.questions == ["退货政策"] for c in chunks)
    assert all(c.section_path == "售后手册 / 退货政策" for c in chunks)
    assert all(len(c.answer) <= 80 and c.answer.endswith("。") for c in chunks)
    assert chunks[0].answer.endswith("商品应保持完好。\n退货需保留包装。")
    assert chunks[1].answer.startswith("商品应保持完好。\n退货需保留包装。\n签收后七天内申请。")
    assert "".join(c.answer for c in chunks).count("签收后七天内申请。") >= 8


def test_table_chunks_copy_headers_and_never_lose_rows():
    rows = [f"| 地区{i} | 运费{i}元 |" for i in range(12)]
    chunks = split_markdown(
        "# 运费\n## 地区标准\n| 地区 | 费用 |\n| --- | --- |\n" + "\n".join(rows),
        max_chars=90,
        overlap_chars=0,
    )
    assert len(chunks) > 2
    assert all(c.content_type == "table" for c in chunks)
    assert all(c.answer.startswith("| 地区 | 费用 |\n| --- | --- |\n") for c in chunks)
    assert all(len(c.answer) <= 90 for c in chunks)
    for row in rows:
        assert sum(row in c.answer.splitlines() for c in chunks) == 1


def test_faq_headings_are_real_questions_and_key_clauses_are_explicit():
    chunks = split_markdown(
        "<!-- content_type: faq -->\n# 耳机 FAQ\n## 可以淋雨吗？\n"
        "<!-- key_clause -->\n耳机不防水，请避免淋雨。\n"
        "```markdown\n# 这是代码内容\n```\n"
    )
    assert len(chunks) == 1
    assert chunks[0].questions == ["可以淋雨吗？"]
    assert chunks[0].is_key_clause
    assert chunks[0].content_type == "faq"
    assert "# 这是代码内容" in chunks[0].answer


def test_impossibly_long_sentence_reports_input_instead_of_truncating():
    with pytest.raises(ValueError, match="完整句|完整行"):
        split_markdown("# 政策\n" + "长" * 100 + "。", max_chars=40)


def test_wrapped_sentence_is_not_cut_at_newline_and_decimal_price_is_preserved():
    with pytest.raises(ValueError, match="完整句|完整行"):
        split_markdown("# 费用\n" + ("同一句中的文字\n" * 10) + "。", max_chars=40)
    chunks = split_markdown(
        "# 费用\n" + "配件价格为19.99元。配送费用为8.50元。" * 8,
        max_chars=60,
        overlap_chars=15,
    )
    assert all(c.answer.endswith("。") for c in chunks)
    assert all("19.\n" not in c.answer and "8.\n" not in c.answer for c in chunks)
