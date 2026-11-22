"""Static guardrails against tests that silently prove nothing."""

import ast
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = PROJECT_ROOT / "tests"
TEST_ROOTS = (TESTS_ROOT,)
TEST_FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def iter_test_files():
    for root in TEST_ROOTS:
        yield from root.rglob("test_*.py")


def is_len_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
    )


def iter_test_functions(tree: ast.AST):
    return (
        node for node in ast.walk(tree)
        if isinstance(node, TEST_FUNCTION_TYPES) and node.name.startswith("test")
    )


def executable_body(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """Return a function body without its optional docstring expression."""
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    return body


def same_expression(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, include_attributes=False) == ast.dump(
        right,
        include_attributes=False,
    )


def none_get_guard(node: ast.AST):
    """Return ``(mapping, key)`` for ``mapping.get(key) is None``."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.ops[0], ast.Is):
        return None
    left, right = node.left, node.comparators[0]
    if not (isinstance(right, ast.Constant) and right.value is None):
        return None
    if not (
        isinstance(left, ast.Call)
        and isinstance(left.func, ast.Attribute)
        and left.func.attr == "get"
        and len(left.args) == 1
    ):
        return None
    return left.func.value, left.args[0]


def membership_guard(node: ast.AST):
    """Return ``(mapping, key)`` for ``key in mapping``."""
    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return None
    if not isinstance(node.ops[0], ast.In):
        return None
    return node.comparators[0], node.left


def is_none_or_membership_tautology(node: ast.AST) -> bool:
    """Detect ``mapping.get(key) is None or key in mapping``."""
    if not (
        isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.Or)
        and len(node.values) == 2
    ):
        return False
    for none_node, membership_node in (
        (node.values[0], node.values[1]),
        (node.values[1], node.values[0]),
    ):
        none_parts = none_get_guard(none_node)
        membership_parts = membership_guard(membership_node)
        if none_parts is None or membership_parts is None:
            continue
        none_mapping, none_key = none_parts
        member_mapping, member_key = membership_parts
        if same_expression(none_mapping, member_mapping) and same_expression(
            none_key,
            member_key,
        ):
            return True
    return False


def contains_verification(node: ast.AST) -> bool:
    """Recognize assertion syntax and common pytest/mock verification calls."""
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Assert):
            return True
        if not isinstance(candidate, ast.Call):
            continue
        if isinstance(candidate.func, ast.Attribute):
            if candidate.func.attr.startswith("assert_"):
                return True
            if (
                isinstance(candidate.func.value, ast.Name)
                and candidate.func.value.id == "pytest"
                and candidate.func.attr in {"raises", "warns", "fail"}
            ):
                return True
        elif isinstance(candidate.func, ast.Name) and candidate.func.id.startswith("assert_"):
            return True
    return False


def assigned_none_before(body: list[ast.stmt], guard: ast.If, name: str) -> bool:
    for statement in body:
        if statement is guard:
            return False
        for candidate in ast.walk(statement):
            if not isinstance(candidate, (ast.Assign, ast.AnnAssign)):
                continue
            targets = candidate.targets if isinstance(candidate, ast.Assign) else [candidate.target]
            value = candidate.value
            if (
                isinstance(value, ast.Constant)
                and value.value is None
                and any(isinstance(target, ast.Name) and target.id == name for target in targets)
            ):
                return True
    return False


def test_no_duplicate_test_definitions_in_the_same_scope():
    duplicates = []

    for path in iter_test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        scopes = [("<module>", tree.body)] + [
            (node.name, node.body)
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for scope_name, body in scopes:
            counts = Counter(
                node.name
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test")
            )
            duplicates.extend(
                f"{path.relative_to(PROJECT_ROOT)}::{scope_name}::{name}"
                for name, count in counts.items()
                if count > 1
            )

    assert duplicates == []


def test_no_vacuous_assertions():
    violations = []

    for path in iter_test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue

            reason = None
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                reason = "assert True"
            elif is_none_or_membership_tautology(node.test):
                reason = "mapping.get(key) is None or key in mapping tautology"
            elif isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.Or):
                if any(isinstance(value, ast.Constant) and value.value is True for value in node.test.values):
                    reason = "assert ... or True"
            elif isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
                left = node.test.left
                right = node.test.comparators[0]
                op = node.test.ops[0]
                if (
                    is_len_call(left)
                    and isinstance(op, ast.GtE)
                    and isinstance(right, ast.Constant)
                    and right.value == 0
                ) or (
                    isinstance(left, ast.Constant)
                    and left.value == 0
                    and isinstance(op, ast.LtE)
                    and is_len_call(right)
                ):
                    reason = "non-negative len assertion"

            if reason:
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {reason}"
                )

    assert violations == []


def test_no_test_functions_with_only_pass():
    violations = []

    for path in iter_test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in iter_test_functions(tree):
            body = executable_body(function)
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{function.lineno}: {function.name}"
                )

    assert violations == []


def test_no_conditional_only_assertions_without_prerequisite():
    """Reject common optional-output guards that can skip every verification.

    This intentionally targets high-confidence cases: a sentinel initialized to
    ``None`` (such as ``model_chat_output``), or a diagnostic collection such as
    ``errors``. Exhaustive if/else assertions and tests with an independent
    prerequisite assertion/mock check are left alone. Names that conventionally
    hold the event/output being proved are also treated as required collections.
    """
    violations = []
    required_collection_suffixes = (
        "error",
        "errors",
        "warning",
        "warnings",
        "diagnostics",
        "context",
        "contexts",
        "index",
        "indices",
        "message",
        "messages",
        "msgs",
        "event",
        "events",
        "output",
        "outputs",
        "result",
        "results",
    )

    for path in iter_test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in iter_test_functions(tree):
            body = executable_body(function)
            for statement in body:
                if not (
                    isinstance(statement, ast.If)
                    and not statement.orelse
                    and isinstance(statement.test, ast.Name)
                    and contains_verification(statement)
                ):
                    continue
                guard_name = statement.test.id
                high_confidence_guard = (
                    assigned_none_before(body, statement, guard_name)
                    or guard_name.lower().endswith(required_collection_suffixes)
                )
                if not high_confidence_guard:
                    continue
                other_statements = [item for item in body if item is not statement]
                if any(contains_verification(item) for item in other_statements):
                    continue
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{statement.lineno}: "
                    f"conditional-only verification guarded by {guard_name!r}"
                )

    assert violations == []


# pytest's default ``norecursedirs``: a test package with one of these names
# is silently never collected (``tests/build`` once hid ~250 tests).
PYTEST_DEFAULT_NORECURSE = {"build", "dist", "CVS", "_darcs", "node_modules", "venv", "{arch}"}


def _test_dirs():
    return [path for path in TESTS_ROOT.rglob("*")
            if path.is_dir() and path.name != "__pycache__" and "fixtures" not in path.parts]


def test_no_test_directory_is_hidden_from_collection():
    hidden = [str(path.relative_to(PROJECT_ROOT)) for path in _test_dirs()
              if path.name in PYTEST_DEFAULT_NORECURSE or path.name.startswith(".") or path.name.endswith(".egg")]
    assert hidden == []


def test_every_test_directory_is_a_package():
    """Package-qualified module names keep same-named test files distinct."""
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in [TESTS_ROOT, *_test_dirs()]
               if not (path / "__init__.py").is_file()]
    assert missing == []


def test_tests_use_only_tracked_inputs_and_helpers():
    """examples/ is gitignored and test/ + test_support.py were retired."""
    offenders = []
    for path in TESTS_ROOT.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module == "test_support" or node.module.split(".")[0] in {"test", "examples"}):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: imports {node.module}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                    node.value.startswith("examples/") or "/examples/json" in node.value):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: reads {node.value!r}")
    assert offenders == []
