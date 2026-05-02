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
        assert "error" in result
        assert "division by zero" in result["error"] or "ZeroDivisionError" in result["error"]

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
        assert "error" in result
        assert "Syntax error" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_missing_run(self):
        """Missing run() returns {'error': message}, not raises."""
        runner = CodeRunner()
        result = await runner.execute(
            "x = 1",
            {},
        )
        assert "error" in result
        assert "must define run" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Timeout enforcement returns timeout error dict via asyncio.wait_for."""
        runner = CodeRunner(timeout=0.01)
        # Use a tight loop that should far exceed the 10ms timeout
        # asyncio.wait_for + asyncio.to_thread will raise TimeoutError
        result = await runner.execute(
            "def run(handler):\n"
            "    total = 0\n"
            "    for _ in range(10**8):\n"
            "        total += 1\n"
            "    return total",
            {},
        )
        assert "error" in result
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_does_not_block_event_loop(self):
        """asyncio.to_thread allows other tasks to run during execution."""
        runner = CodeRunner(timeout=10.0)

        async def other_task():
            return "I ran concurrently"

        # Use a CPU-bound loop that runs in the thread (via to_thread).
        # Other_task should complete while the thread is still running.
        exec_task = asyncio.create_task(
            runner.execute(
                "def run(handler):\n"
                "    x = 0\n"
                "    for i in range(500000):\n"
                "        x += i\n"
                "    return handler['result'] + x",
                {"result": 42},
            )
        )
        other_task_obj = asyncio.create_task(other_task())

        exec_result, other_result = await asyncio.gather(exec_task, other_task_obj)
        assert other_result == "I ran concurrently"
        assert exec_result == {"result": 42 + sum(range(500000))}


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
        assert "Do not use with untrusted" in doc or "Do NOT use with untrusted" in doc, (
            "Class docstring must contain 'Do not use with untrusted code'"
        )

    def test_init_docstring_has_warnings(self):
        """__init__ docstring must contain security warnings."""
        init_doc = CodeRunner.__init__.__doc__
        assert init_doc is not None, "CodeRunner.__init__ must have a docstring"
        assert "NOT a security boundary" in init_doc, (
            "__init__ docstring must contain 'NOT a security boundary'"
        )
        assert "Do not use with untrusted" in init_doc, (
            "__init__ docstring must contain 'Do not use with untrusted code'"
        )
