"""检查 issue #8 的模块边界；业务行为继续由现有入口回归验证。"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src/support_agent"
DEPENDENCIES = {
    "conversation": set(),
    "commerce": set(),
    "knowledge": set(),
    "tooling": set(),
    "improvement": {"conversation", "knowledge"},
    "evaluation": {"knowledge"},
    "dialogue": {"conversation", "commerce", "knowledge", "tooling", "improvement"},
    "platform": set(),
}

TABLE_OWNERS = {
    "conversation": {"conversations", "conversation_summaries", "messages"},
    "commerce": {"tickets", "refund_applications"},
    "knowledge": {
        "faq",
        "knowledge_chunks",
        "qa_extraction_staging",
        "knowledge_extraction_batches",
    },
    "tooling": {"tool_audit_logs"},
    "improvement": {
        "low_confidence_questions",
        "turn_retrievals",
        "review_queue",
        "answer_feedback",
    },
    "evaluation": {"eval_runs"},
    "platform": {"request_intents", "model_usage"},
}


def test_feature_modules_expose_public_boundaries_and_only_allowed_dependencies():
    violations = []
    for owner, allowed in DEPENDENCIES.items():
        assert (ROOT / owner / "__init__.py").is_file(), f"缺少模块 {owner}"
        for path in (ROOT / owner).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                elif isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                else:
                    continue
                for name in imports:
                    parts = name.split(".")
                    if len(parts) < 2 or parts[0] != "support_agent":
                        continue
                    target = parts[1]
                    if target == owner or target == "platform":
                        continue
                    if target not in allowed or len(parts) != 2:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "越过模块边界：\n" + "\n".join(violations)


def test_persistent_tables_belong_to_their_feature_module():
    actual = {owner: set() for owner in TABLE_OWNERS}
    for path in ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == "__tablename__"
                for target in node.targets
            ):
                continue
            owner = path.relative_to(ROOT).parts[0]
            assert owner in actual, f"业务表未归属功能模块：{path}"
            actual[owner].add(ast.literal_eval(node.value))
    assert actual == TABLE_OWNERS


def test_every_source_function_has_a_chinese_purpose_comment():
    violations = []
    for path in ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node) or ""
            preceding = lines[node.lineno - 2].strip() if node.lineno > 1 else ""
            comment = preceding if preceding.startswith("#") else ""
            if not any("\u4e00" <= char <= "\u9fff" for char in doc + comment):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.name}")
    assert not violations, "缺少中文用途注释：\n" + "\n".join(violations)
