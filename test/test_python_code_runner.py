"""
Unit tests for CodeRunner (magic_agents/node_system/python_code_runner.py).

Tests cover:
- compile() with valid/invalid code
- execute() with various handler scenarios
- Timeout enforcement
- Error handling (exceptions, syntax errors)
- Docstring sandbox warnings
"""
import asyncio
import multiprocessing
import os
import time

import pytest

from magic_agents.node_system.python_code_runner import CodeRunner


class TestCodeRunnerCompile:
    """Tests for CodeRunner.compile()."""

    def test_compile_valid_run_handler(self):
        """Valid run(handler) code compiles and returns callable."""
        runner = CodeRunner()
        func = runner.compile("def run(handler): return handler['x'] * 2")
        assert callable(func)
        assert func({"x": 5}) == 10

    def test_compile_simple_return(self):
        """Simple run(handler) with arithmetic."""
        runner = CodeRunner()
        func = runner.compile("def run(handler): return handler['a'] + handler['b']")
        assert func({"a": 3, "b": 5}) == 8

    def test_compile_missing_run_raises(self):
        """Code without run() raises ValueError."""
        runner = CodeRunner()
        with pytest.raises(ValueError, match="User code must define run\\(handler\\) function"):
            runner.compile("x = 1")

    def test_compile_non_callable_run_raises(self):
        """run bound to non-callable raises ValueError."""
        runner = CodeRunner()
        with pytest.raises(ValueError, match="'run' must be a callable function"):
            runner.compile("run = 42")

    def test_compile_syntax_error_raises(self):
        """Invalid syntax raises ValueError with syntax error message."""
        runner = CodeRunner()
        with pytest.raises(ValueError, match="Syntax error in user code"):
            runner.compile("def run(handler): return 1/")

    def test_compile_empty_code_raises(self):
        """Empty code string raises ValueError."""
        runner = CodeRunner()
        with pytest.raises(ValueError, match="User code must define run\\(handler\\) function"):
            runner.compile("")

    def test_compile_whitespace_only_raises(self):
        """Whitespace-only code raises ValueError."""
        runner = CodeRunner()
        with pytest.raises(ValueError, match="User code must define run\\(handler\\) function"):
            runner.compile("   \n  \n  ")

    def test_compile_restricted_builtins_available(self):
        """Restricted builtins like len(), str(), int() are available."""
        runner = CodeRunner()
        func = runner.compile("def run(handler): return len(handler['items'])")
        assert func({"items": [1, 2, 3]}) == 3

    def test_compile_run_with_empty_handler(self):
        """run(handler) with empty dict works."""
        runner = CodeRunner()
        func = runner.compile("def run(handler): return handler.get('x', 'default')")
        assert func({}) == 'default'


class TestCodeRunnerExecute:
    """Tests for CodeRunner.execute()."""

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Successful execution returns {'result': value}."""
        runner = CodeRunner()
        result = await runner.execute(
            "def run(handler): return handler['x'] + handler['y']",
            {"x": 10, "y": 20},
        )
        assert result == {"result": 30}

    @pytest.mark.asyncio
    async def test_execute_with_dict_result(self):
        """run() returning a dict works."""
        runner = CodeRunner()
        result = await runner.execute(
            "def run(handler): return {'sum': handler['a'] + handler['b']}",
            {"a": 3, "b": 5},
        )
        assert result == {"result": {"sum": 8}}

    @pytest.mark.asyncio
    async def test_execute_runtime_exception(self):
        """Runtime exception returns {'error': message}."""
        runner = CodeRunner()
        result = await runner.execute(
            "def run(handler): return 1 / 0",
            {},
        )
        assert result == {"error": "division by zero"}

    @pytest.mark.asyncio
    async def test_execute_empty_handler(self):
        """run() with empty handler dict works."""
        runner = CodeRunner()
        result = await runner.execute(
            "def run(handler): return {'keys': list(handler.keys())}",
            {},
        )
        assert result == {"result": {"keys": []}}

    @pytest.mark.asyncio
    async def test_execute_none_return(self):
        """run() returning None returns {'result': None}."""
        runner = CodeRunner()
        result = await runner.execute(
            "def run(handler): return None",
            {},
        )
        assert result == {"result": None}

    @pytest.mark.asyncio
    async def test_execute_compilation_error(self):
        """Syntax error in code returns {'error': message}, not raises."""
        runner = CodeRunner()
        result = await runner.execute(
            "def run(handler): return 1/",
            {},
        )
        assert set(result) == {"error"}
        assert result["error"].startswith("Syntax error in user code:")

    @pytest.mark.asyncio
    async def test_execute_missing_run(self):
        """Missing run() returns {'error': message}, not raises."""
        runner = CodeRunner()
        result = await runner.execute(
            "x = 1",
            {},
        )
        assert result == {"error": "User code must define run(handler) function"}

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """A non-terminating subprocess is killed at the configured deadline."""
        runner = CodeRunner(timeout=0.1)
        started = time.monotonic()

        result = await runner.execute(
            "def run(handler):\n    while True:\n        pass",
            {},
        )

        assert result == {"error": "execution timed out after 0.1 seconds"}
        assert time.monotonic() - started < 2.0
        assert not any(
            child.name == "magic-agents-python-exec" and child.is_alive()
            for child in multiprocessing.active_children()
        )

    @pytest.mark.asyncio
    async def test_execute_does_not_block_event_loop(self):
        """A blocked run(handler) worker does not block another coroutine."""
        runner = CodeRunner(timeout=2.0)
        execution_task = asyncio.create_task(
            runner.execute(
                "import time\ndef run(handler):\n    time.sleep(0.2)\n    return 'done'",
                {},
            )
        )
        await asyncio.sleep(0.01)

        assert not execution_task.done()
        assert await execution_task == {"result": "done"}

    @pytest.mark.asyncio
    async def test_subprocess_mode_isolates_pid_and_handler_mutation(self):
        runner = CodeRunner(safety_mode="subprocess")
        handler = {"items": [1]}

        result = await runner.execute(
            "import os\ndef run(handler):\n"
            "    handler['items'].append(2)\n"
            "    return {'pid': os.getpid(), 'items': handler['items']}",
            handler,
        )

        assert result["result"]["pid"] != os.getpid()
        assert result["result"]["items"] == [1, 2]
        assert handler == {"items": [1]}

    @pytest.mark.asyncio
    async def test_max_output_chars_truncates_string_results(self):
        runner = CodeRunner(max_output_chars=5)

        result = await runner.execute("def run(handler): return 'abcdefgh'", {})

        assert result == {"result": "abcde\n... [truncated 3 chars]"}

    @pytest.mark.asyncio
    async def test_restricted_mode_blocks_import_while_in_process_allows_it(self):
        source = "import os\ndef run(handler): return os.getpid()"

        restricted = await CodeRunner(safety_mode="restricted_builtins").execute(source, {})
        local = await CodeRunner(safety_mode="in_process").execute(source, {})

        assert "error" in restricted
        assert "__import__" in restricted["error"]
        assert local == {"result": os.getpid()}


class TestCodeRunnerSandboxDocs:
    """Tests for CodeRunner docstring sandbox warnings."""

    def test_docstring_has_security_warnings(self):
        """Class and __init__ docstrings must contain security warnings."""
        import inspect
        doc = inspect.getdoc(CodeRunner)
        assert doc is not None, "CodeRunner must have a class-level docstring"
        assert "NOT a security boundary" in doc, (
            "Class docstring must contain 'NOT a security boundary'"
        )
        assert "Do NOT use with untrusted/third-party code" in doc, (
            "Class docstring must contain its untrusted-code warning"
        )

    def test_init_docstring_has_warnings(self):
        """__init__ docstring must contain security warnings."""
        init_doc = CodeRunner.__init__.__doc__
        assert init_doc is not None, "CodeRunner.__init__ must have a docstring"
        assert "NOT a security boundary" in init_doc, (
            "__init__ docstring must contain 'NOT a security boundary'"
        )
        assert "Do not use with untrusted code" in init_doc, (
            "__init__ docstring must contain its untrusted-code warning"
        )
