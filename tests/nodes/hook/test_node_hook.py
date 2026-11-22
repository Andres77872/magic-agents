"""Unit tests for NodeHook node type.

Phase 9.5: Tests for:
- NodeHook receives HookContext
- NodeHook timeout enforced (test with slow function)
- NodeHook error isolation (test with exception-raising function)
- NodeHook emits via emit.user
- NodeHook compiles function template correctly

NOTE: NodeHook inherits magic_telemetry via Node.__init_subclass__, which
wraps process() and yields start/end telemetry events around the actual
hook output. Tests account for this by filtering telemetry events.
"""
import pytest

from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.context_factory import HookContextFactory
from magic_agents.models.factory.Nodes.HookNodeModel import (
    HookNodeModel,
    DEFAULT_INPUT_HOOK_CONTEXT,
)
from magic_agents.node_system.NodeHook import NodeHook


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_hook_node(
    function_template: str = "",
    timeout_override: int | None = None,
    hook_type: str = "custom",
    node_id: str = "hook-test-1",
) -> NodeHook:
    """Create a NodeHook with the given template and config."""
    model = HookNodeModel(
        function_template=function_template,
        timeout_override=timeout_override,
        hook_type=hook_type,
    )
    node = NodeHook(
        data=model,
        node_id=node_id,
        debug=False,
    )
    return node


def make_hook_context(**overrides) -> HookContext:
    """Create a minimal HookContext for testing."""
    ctx = HookContextFactory.build_node_context(
        execution_id=overrides.get("execution_id", "test-graph-1"),
        run_id=overrides.get("run_id", "run-abc"),
        node_id=overrides.get("node_id", "target-node-1"),
        node_type=overrides.get("node_type", "LLM"),
        inputs=overrides.get("inputs", {"query": "hello"}),
    )
    ctx.sequence_number = overrides.get("sequence_number", 1)
    return ctx


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestNodeHookTemplateCompilation:
    """Tests for _compile_hook_function() and _extract_function_name()."""

    def test_compiles_simple_function(self):
        """A simple async function template compiles correctly."""
        node = make_hook_node(
            function_template="""
async def my_hook(context, chat_log):
    return {"type": "done", "content": "ok"}
"""
        )

        func = node._compile_hook_function(node._function_template)

        assert func is not None
        assert callable(func)
        assert func.__name__ == "my_hook"

    def test_compiles_sync_function(self):
        """A sync function template also compiles."""
        node = make_hook_node(
            function_template="""
def my_hook(context, chat_log):
    return {"type": "done", "content": "ok"}
"""
        )

        func = node._compile_hook_function(node._function_template)

        assert func is not None
        assert callable(func)

    def test_empty_template_returns_none(self):
        """Empty template returns None (no function to call)."""
        node = make_hook_node(function_template="")

        func = node._compile_hook_function(node._function_template)

        assert func is None

    def test_whitespace_template_returns_none(self):
        """Whitespace-only template returns None."""
        node = make_hook_node(function_template="   \n  \n  ")

        func = node._compile_hook_function(node._function_template)

        assert func is None

    def test_extract_function_name_async(self):
        """_extract_function_name finds async def names."""
        node = make_hook_node(function_template="async def my_hook(ctx, log): pass")
        name = node._extract_function_name(node._function_template)

        assert name == "my_hook"

    def test_extract_function_name_sync(self):
        """_extract_function_name finds def names."""
        node = make_hook_node(function_template="def process_hook(ctx, log): pass")
        name = node._extract_function_name(node._function_template)

        assert name == "process_hook"

    def test_extract_function_name_no_def(self):
        """_extract_function_name returns None when no def found."""
        node = make_hook_node(function_template="x = 42")
        name = node._extract_function_name(node._function_template)

        assert name is None

    def test_extract_function_name_lambda(self):
        """_extract_function_name falls back to lambda assignment."""
        node = make_hook_node(function_template="my_func = lambda ctx, log: None")
        name = node._extract_function_name(node._function_template)

        assert name == "my_func"

    def test_compilation_error_returns_none(self):
        """Invalid Python syntax returns None without crashing."""
        node = make_hook_node(
            function_template="async def broken_hook(ctx, log):\n  invalid syntax {{{"
        )

        func = node._compile_hook_function(node._function_template)

        assert func is None

class TestNodeHookContextHandling:
    """Tests for NodeHook receiving and validating HookContext."""

    @pytest.mark.asyncio
    async def test_no_hook_context_skips_execution(self):
        """When no HookContext received, process() yields nothing (only telemetry)."""
        node = make_hook_node(
            function_template="async def hook(ctx, log): return {'ok': True}"
        )

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        # Only 2 telemetry events (start + end), no hook output
        assert len(results) == 2, (
            "Expected only telemetry start/end events when no HookContext"
        )

    @pytest.mark.asyncio
    async def test_invalid_input_type_skips(self):
        """When input is not a HookContext, process() skips execution."""
        node = make_hook_node(
            function_template="async def hook(ctx, log): return {'ok': True}"
        )
        # Set a non-HookContext input
        node.inputs[DEFAULT_INPUT_HOOK_CONTEXT] = "not a HookContext"

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        # Only 2 telemetry events (start + end), no hook output
        assert len(results) == 2, (
            "Expected only telemetry events when input is not HookContext"
        )

    @pytest.mark.asyncio
    async def test_hook_context_received_and_used(self):
        """NodeHook receives HookContext and template function uses it."""
        node = make_hook_node(
            function_template="""
async def my_hook(context, chat_log):
    return {"type": "result", "content": f"node={context.node_id}"}
"""
        )
        assert isinstance(node, NodeHook)
        ctx = make_hook_context(node_id="target-node-1")
        node.inputs[DEFAULT_INPUT_HOOK_CONTEXT] = ctx

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        # 3 events: telemetry_start + hook_result + telemetry_end
        assert len(results) == 3, (
            f"Expected 3 events (telemetry + result + telemetry), got {len(results)}"
        )
        # Hook result is the middle event
        assert results[1]["type"] == "result"
        assert results[1]["content"] == "node=target-node-1"

    @pytest.mark.asyncio
    async def test_emit_interface_injected_into_context(self):
        """HookContext.emit is set during process()."""
        node = make_hook_node(
            function_template="""
async def my_hook(context, chat_log):
    assert context.emit is not None, "emit should be injected"
    return {"type": "result", "content": "emit_ok"}
"""
        )
        ctx = make_hook_context()
        node.inputs[DEFAULT_INPUT_HOOK_CONTEXT] = ctx

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        # 3 events: telemetry_start + result + telemetry_end
        assert len(results) == 3
        assert results[1]["content"] == "emit_ok"


class TestNodeHookTimeout:
    """Tests for NodeHook timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_enforced(self):
        """A slow function hits the timeout and yields a debug error event."""
        node = make_hook_node(
            function_template="""
async def slow_hook(context, chat_log):
    import asyncio
    await asyncio.sleep(10)
    return {"type": "result", "content": "should not reach"}
"""
        )
        ctx = make_hook_context()
        node.inputs[DEFAULT_INPUT_HOOK_CONTEXT] = ctx
        node._timeout_seconds = 0.05  # 50ms for fast test

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        # 3 events: telemetry_start + error_event + telemetry_end
        assert len(results) == 3, (
            f"Expected 3 events (telemetry + error + telemetry), got {len(results)}"
        )
        error = results[1]
        assert error.get("type") == "debug", (
            f"Expected type='debug', got '{error.get('type')}'"
        )
        error_content = error.get("content", {})
        assert error_content.get("error_type") == "HookTimeout", (
            f"Expected HookTimeout, got '{error_content.get('error_type')}'"
        )

    @pytest.mark.asyncio
    async def test_normal_function_not_timed_out(self):
        """A quick function completes before timeout."""
        node = make_hook_node(
            function_template="""
async def quick_hook(context, chat_log):
    return {"type": "result", "content": "completed"}
"""
        )
        ctx = make_hook_context()
        node.inputs[DEFAULT_INPUT_HOOK_CONTEXT] = ctx

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        # 3 events: telemetry_start + result + telemetry_end
        assert len(results) == 3
        assert results[1]["content"] == "completed"

    def test_default_timeout_applied(self):
        """NodeHook uses DEFAULT_TIMEOUT_SECONDS when no override."""
        node = make_hook_node(
            function_template="""
async def hook(context, chat_log):
    return {"type": "result", "content": "ok"}
"""
        )
        assert node._timeout_seconds == 30, (
            f"Expected default timeout of 30s, got {node._timeout_seconds}"
        )

    def test_timeout_override_from_model(self):
        """timeout_override from HookNodeModel is used."""
        model = HookNodeModel(
            function_template="""
async def hook(context, chat_log):
    return {"type": "result", "content": "ok"}
""",
            timeout_override=60,
        )
        node = NodeHook(data=model, node_id="test")
        assert node._timeout_seconds == 60, (
            f"Expected timeout of 60s from model override, got {node._timeout_seconds}"
        )


class TestNodeHookErrorIsolation:
    """Tests for NodeHook error isolation during execution."""

    @pytest.mark.asyncio
    async def test_exception_in_function_yields_debug_error(self):
        """When function raises an exception, a debug error event is yielded."""
        node = make_hook_node(
            function_template="""
async def failing_hook(context, chat_log):
    raise ValueError("something went wrong")
"""
        )
        ctx = make_hook_context()
        node.inputs[DEFAULT_INPUT_HOOK_CONTEXT] = ctx

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        # 3 events: telemetry_start + error_event + telemetry_end
        assert len(results) == 3, (
            f"Expected 3 events, got {len(results)}"
        )
        error = results[1]
        assert error.get("type") == "debug", (
            f"Expected type='debug', got '{error.get('type')}'"
        )
        error_content = error.get("content", {})
        assert error_content.get("error_type") == "HookError", (
            f"Expected HookError, got '{error_content.get('error_type')}'"
        )

class TestNodeHookEmitIntegration:
    """Tests for NodeHook emitting via emit.user/debug/feedback."""

    @pytest.mark.asyncio
    async def test_hook_returns_emit_user_result(self):
        """Hook function using emit.user returns type='content'."""
        node = make_hook_node(
            function_template="""
async def hook_with_emit(context, chat_log):
    result = context.emit.user("Progress: 50%")
    return result
"""
        )
        ctx = make_hook_context()
        node.inputs[DEFAULT_INPUT_HOOK_CONTEXT] = ctx

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        content = [r["content"] for r in results if r["type"] == "content"]
        # emit.user returns the prep-wrapped ChatCompletionModel; the telemetry
        # wrapper adds empty content frames around it.
        texts = [chunk["content"].choices[0].delta.content for chunk in content]
        assert [text for text in texts if text] == ["Progress: 50%"]

    @pytest.mark.asyncio
    async def test_hook_returns_emit_debug_result(self):
        """Hook function using emit.debug returns type='debug'."""
        node = make_hook_node(
            function_template="""
async def multi_hook(context, chat_log):
    return context.emit.debug("STEP", {"step": 1})
"""
        )
        ctx = make_hook_context()
        node.inputs[DEFAULT_INPUT_HOOK_CONTEXT] = ctx

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        # 3 events: telemetry_start + debug_result + telemetry_end
        assert len(results) == 3
        debug_result = results[1]
        assert debug_result["type"] == "debug"
        assert debug_result["content"]["event_type"] == "STEP"

    @pytest.mark.asyncio
    async def test_hook_function_returning_none(self):
        """Hook function that returns None yields no hook output (only telemetry)."""
        node = make_hook_node(
            function_template="""
async def noop_hook(context, chat_log):
    return None  # No output
"""
        )
        ctx = make_hook_context()
        node.inputs[DEFAULT_INPUT_HOOK_CONTEXT] = ctx

        results = []
        async for result in node.process(chat_log=None):
            results.append(result)

        # Only 2 telemetry events, no hook output since function returned None
        assert len(results) == 2, (
            f"Expected 2 telemetry events, got {len(results)}"
        )


class TestNodeHookHandleConstants:
    """Tests for NodeHook input/output handle constants."""

    def test_default_handle_constants(self):
        """Default handle names match constants."""
        node = make_hook_node()

        assert node.INPUT_HANDLE_HOOK_CONTEXT == "handle-hook-context"
        assert node.OUTPUT_HANDLE_USER == "handle-user-output"
        assert node.OUTPUT_HANDLE_DEBUG == "handle-debug-output"
        assert node.OUTPUT_HANDLE_FEEDBACK == "handle-feedback-output"

    def test_capture_internal_state(self):
        """_capture_internal_state includes NodeHook-specific fields."""
        model = HookNodeModel(
            function_template="async def h(ctx, log): pass",
            hook_type="pre",
            timeout_override=15,
        )
        node = NodeHook(data=model, node_id="test")
        state = node._capture_internal_state()

        assert state["hook_type"] == "pre"
        assert state["timeout"] == 15
        assert state["template_length"] > 0
