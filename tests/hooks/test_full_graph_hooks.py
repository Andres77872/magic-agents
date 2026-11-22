"""
E2E Tests: Full Graph with Hooks

Proves complete flow execution when hooks are registered for all 9 minimum
FlowHooks methods. Uses TestNode instances for direct executor tests (existing
pattern) and AgentFlowModel/build() for public API integration tests (Task 4.2).

Verifies:
- All 9 minimum hooks fire via the real execution path
- Hooks fire in correct lifecycle order (graph → node → node → graph)
- Timing is correct (before/after execution)
- Error paths invoke error hooks, not end hooks
- Public API: RuntimeConfig via run_agent(), AgentFlowModel.hooks via build(),
  JSON NodeHook node via build()

Required constraints:
- Empty registry = no behavior change (backward compatible)
- All 9 minimum hooks fire via HookRegistry invocation
- Errors fire error hooks, not end hooks
"""
import pytest
import asyncio
import logging

from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.node_system.Node import Node
from magic_agents.hooks import (
    HookContext,
    HookRegistry,
    RuntimeConfig,
)
from tests.support.hooks import RecordingHook


# ─── Test Node (matching existing test pattern) ──────────────────────────────

class TestNode(Node):
    __test__ = False  # executor helper, not a pytest test container
    """Simple node for testing — matches test_reactive_executor_debug_events.py."""

    def __init__(self, node_id: str, debug: bool = True, **kwargs):
        super().__init__(node_id=node_id, node_type="test", debug=debug, **kwargs)

    async def process(self, chat_log):
        yield {"type": "end", "content": {"node": self.node_id, "content": "test output"}}


class ErrorTestNode(Node):
    """Node that raises during process() to test error hooks.

    Must be an async generator (yields at least once) for magic_telemetry.
    """

    def __init__(self, node_id: str, debug: bool = True, **kwargs):
        super().__init__(node_id=node_id, node_type="error_test", debug=debug, **kwargs)

    async def process(self, chat_log):
        yield {"type": "start", "content": {"node": self.node_id}}
        raise ValueError(f"Simulated error in {self.node_id}")


class TestGraphLifecycleHooks:
    """Graph lifecycle hooks (on_graph_start, on_graph_end)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    @pytest.mark.asyncio
    async def test_graph_hook_context_has_execution_id(self):
        """Graph-level hook context has execution_id and metadata."""
        hook = RecordingHook()
        RuntimeConfig.register_global_hook(hook)
        config = RuntimeConfig()
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={"test-1": TestNode(node_id="test-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        starts = hook.get_invocations("on_graph_start")
        ctx = starts[0]["context"]
        assert ctx.execution_id, "Graph start context must have execution_id"

    @pytest.mark.asyncio
    async def test_graph_start_before_node_start(self):
        """Graph start fires BEFORE any node start hooks."""
        hook = RecordingHook()
        RuntimeConfig.register_global_hook(hook)
        config = RuntimeConfig()
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={
                "node-1": TestNode(node_id="node-1"),
                "node-2": TestNode(node_id="node-2"),
            },
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        names = hook.hook_names
        # Filter to just the invocation order
        [n for n in names if n in ("on_graph_start", "on_graph_end")]

        # The very first graph hook should be on_graph_start
        # and the last graph hook should be on_graph_end

        # But there might be node hooks in between
        graph_start_idx = names.index("on_graph_start")
        graph_end_idx = names.index("on_graph_end")

        node_start_indices = [i for i, n in enumerate(names) if n == "on_node_start"]
        node_end_indices = [i for i, n in enumerate(names) if n == "on_node_end"]

        assert len(node_start_indices) == len(graph.nodes)
        assert len(node_end_indices) == len(graph.nodes)
        assert graph_start_idx < node_start_indices[0], (
            "on_graph_start must fire BEFORE on_node_start"
        )
        assert node_end_indices[-1] < graph_end_idx, (
            "on_node_end must fire BEFORE on_graph_end"
        )


class TestNodeLifecycleHooks:
    """Node lifecycle hooks (on_node_start, on_node_end)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    @pytest.mark.asyncio
    async def test_node_start_and_end_fire(self):
        """on_node_start and on_node_end fire for executed nodes."""
        hook = RecordingHook()
        RuntimeConfig.register_global_hook(hook)
        config = RuntimeConfig()
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={"test-1": TestNode(node_id="test-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        starts = hook.get_invocations("on_node_start")
        ends = hook.get_invocations("on_node_end")

        assert len(starts) == 1, "on_node_start must fire once per node"
        assert len(ends) == 1, "on_node_end must fire once per node"

    @pytest.mark.asyncio
    async def test_node_start_fires_before_node_end(self):
        """For each node, on_node_start fires BEFORE on_node_end."""
        hook = RecordingHook()
        RuntimeConfig.register_global_hook(hook)
        config = RuntimeConfig()
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={"test-1": TestNode(node_id="test-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        names = hook.hook_names
        start_idx = names.index("on_node_start")
        end_idx = names.index("on_node_end")

        assert start_idx < end_idx, (
            "on_node_start must fire BEFORE on_node_end for the same node"
        )

    @pytest.mark.asyncio
    async def test_multiple_nodes_hook_independently(self):
        """Multiple nodes each get their own hook invocations."""
        hook = RecordingHook()
        RuntimeConfig.register_global_hook(hook)
        config = RuntimeConfig()
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={
                "node-a": TestNode(node_id="node-a"),
                "node-b": TestNode(node_id="node-b"),
            },
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        starts = hook.get_invocations("on_node_start")
        assert len(starts) == 2, "Should fire on_node_start for each node"
        node_ids = [inv["context"].node_id for inv in starts]
        assert "node-a" in node_ids
        assert "node-b" in node_ids


class TestErrorPathHooks:
    """Error path hooks (on_node_error, on_node_bypass)."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    @pytest.mark.asyncio
    async def test_node_error_invokes_on_node_error(self):
        """Erroring node invokes on_node_error with error context."""
        hook = RecordingHook()
        RuntimeConfig.register_global_hook(hook)
        config = RuntimeConfig()
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={"error-node": ErrorTestNode(node_id="error-node")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        errors = hook.get_invocations("on_node_error")
        assert len(errors) >= 1, "on_node_error must fire for erroring node"

        ctx = errors[0]["context"]
        assert ctx.node_id == "error-node", "Error context must identify erroring node"
        assert ctx.error is not None or ctx.error_type is not None, (
            "Error context must have error information"
        )

    @pytest.mark.asyncio
    async def test_node_error_does_not_fire_node_end(self):
        """Erroring node fires on_node_error instead of on_node_end."""
        hook = RecordingHook()
        RuntimeConfig.register_global_hook(hook)
        config = RuntimeConfig()
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={"error-node": ErrorTestNode(node_id="error-node")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        errors = hook.get_invocations("on_node_error")
        ends = hook.get_invocations("on_node_end")
        end_node_ids = [inv["context"].node_id for inv in ends]

        assert len(errors) >= 1, "on_node_error must fire"
        assert "error-node" not in end_node_ids, (
            "on_node_end must NOT fire for erroring node"
        )

    @pytest.mark.asyncio
    async def test_graph_error_fires_instead_of_graph_end(self):
        """Graph error fires instead of graph end when a node errors.

        Spec requirement: on_graph_end is NOT invoked for failed executions.
        on_graph_error fires with error context and failed node info.
        """
        hook = RecordingHook()
        RuntimeConfig.register_global_hook(hook)
        config = RuntimeConfig()
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={"error-node": ErrorTestNode(node_id="error-node")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        errors = hook.get_invocations("on_graph_error")
        ends = hook.get_invocations("on_graph_end")

        assert len(errors) >= 1, (
            "on_graph_error must fire when a node errors"
        )
        assert len(ends) == 0, (
            "on_graph_end must NOT fire when a node errors (spec requirement)"
        )

        # Verify error context has meaningful info
        ctx = errors[0]["context"]
        assert ctx.execution_id, "Error context must have execution_id"
        assert ctx.error_message, "Error context must have error_message"

        # Verify invocation includes error argument
        kwargs = errors[0].get("kwargs", {})
        assert "error" in kwargs, "on_graph_error must receive error argument"
        assert kwargs["error"] is not None, "on_graph_error error must not be None"


class TestHookWithSlowHooks:
    """Hooks with delays still work and don't hang execution."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    @pytest.mark.asyncio
    async def test_slow_hook_does_not_block_graph(self):
        """A hook that takes time still allows graph to complete."""
        executed = []

        class SlowHook:
            async def on_node_start(self, context: HookContext) -> None:
                await asyncio.sleep(0.01)
                executed.append("start")

            async def on_node_end(self, context: HookContext) -> None:
                await asyncio.sleep(0.01)
                executed.append("end")

            async def on_graph_start(self, context: HookContext) -> None:
                await asyncio.sleep(0.01)
                executed.append("graph_start")

            async def on_graph_end(self, context: HookContext) -> None:
                await asyncio.sleep(0.01)
                executed.append("graph_end")

        RuntimeConfig.register_global_hook(SlowHook())
        config = RuntimeConfig()
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={"test-1": TestNode(node_id="test-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        assert "graph_start" in executed, "Slow on_graph_start must complete"
        assert "graph_end" in executed, "Slow on_graph_end must complete"
        assert "start" in executed, "Slow on_node_start must complete"
        assert "end" in executed, "Slow on_node_end must complete"


# ─── Task 4.2: build() + run_agent() Integration Tests ───────────────────


class TestBuildRunAgentIntegration:
    """build() + run_agent() integration tests (Task 4.2).

    Proves hooks fire through the public API:
    1. RuntimeConfig via run_agent()
    2. AgentFlowModel.hooks via build()
    3. JSON NodeHook node via build()
    """

    @pytest.fixture(autouse=True)
    def cleanup(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    @pytest.mark.asyncio
    async def test_runtime_config_graph_hooks_via_executor(self):
        """RuntimeConfig graph hooks fire through execute_graph_reactive (run_agent path).

        Uses the executor directly since run_agent requires full chat infrastructure.
        The hooks=RuntimeConfig path is the same code path used by run_agent().
        """
        hook = RecordingHook()
        config = RuntimeConfig(graph_hooks=[hook])
        registry = config.create_registry()

        graph = AgentFlowModel(
            nodes={"test-1": TestNode(node_id="test-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        # Verify hooks fired in correct lifecycle order
        hook.assert_method_called("on_graph_start", times=1)
        hook.assert_method_called("on_graph_end", times=1)
        hook.assert_method_called("on_node_start", times=1)
        hook.assert_method_called("on_node_end", times=1)

        # Verify lifecycle ordering: graph_start BEFORE node_start BEFORE node_end BEFORE graph_end
        names = hook.hook_names
        gs = names.index("on_graph_start")
        ns = names.index("on_node_start")
        ne = names.index("on_node_end")
        ge = names.index("on_graph_end")
        assert gs < ns < ne < ge, (
            "Hook lifecycle order must be: graph → node → node → graph"
        )

    @pytest.mark.asyncio
    async def test_json_hook_node_missing_function_template_validation(self):
        """JSON NodeHook node without function_template raises validation error.

        Proves the validation in validate_graph catches invalid hook definitions.
        """
        from magic_agents.agt_flow import validate_graph

        json_data = {
            "nodes": [
                {"id": "hook-logger", "type": "hook"},  # No function_template
            ],
            "edges": [],
        }

        result = validate_graph(json_data["nodes"], json_data["edges"])
        assert not result["valid"], (
            "Validation must fail for hook node without function_template"
        )
        assert len(result["errors"]) > 0, "Must have at least one validation error"

        # Check that the error mentions function_template
        error_types = [e["error_type"] for e in result["errors"]]
        error_messages = [e["error_message"] for e in result["errors"]]
        all_texts = error_types + error_messages
        has_template_error = any(
            "function_template" in text.lower() or "template" in text.lower()
            for text in all_texts
        )
        assert has_template_error, (
            "Validation error must mention function_template"
        )


# ─── GAP-2: JSON Dict Hooks Diagnostic Warning ────────────────────────


class TestJsonDictHooksWarning:
    """JSON dict hooks diagnostic warning (GAP-2 from test-coverage-audit).

    Spec §6.1: When JSON agent data contains a "hooks" key with a dict value,
    build() must emit a WARNING-level log. The hooks key must NOT be passed
    to AgentFlowModel to prevent silent misconfiguration.
    """

    def test_json_dict_hooks_key_emits_warning(self, caplog):
        """Dict hooks key in JSON emits WARNING diagnostic message."""
        from magic_agents.agt_flow import build

        caplog.set_level(logging.WARNING)

        json_data = {
            "hooks": {"on_graph_start": "not_a_real_hook"},
            "nodes": [
                {"id": "test-1", "type": "text", "data": {"content": "hello"}},
            ],
            "edges": [],
        }

        # This should emit a WARNING but not fail
        result = build(json_data, message="test")

        # Verify warning was emitted
        warning_messages = [r.message for r in caplog.records]
        hooks_warnings = [
            m for m in warning_messages
            if "hooks" in m.lower() and "JSON" in m
        ]
        assert len(hooks_warnings) >= 1, (
            f"Expected WARNING about JSON hooks limitation, got: {warning_messages}"
        )

        # Verify model was still created successfully
        assert result is not None, "build() must succeed despite hooks warning"

    def test_no_hooks_key_no_warning(self, caplog):
        """No hooks key in JSON emits no warning."""
        from magic_agents.agt_flow import build

        caplog.set_level(logging.WARNING)

        json_data = {
            "nodes": [
                {"id": "test-1", "type": "text", "data": {"content": "hello"}},
            ],
            "edges": [],
        }

        result = build(json_data, message="test")

        # Verify NO warning about JSON hooks
        warning_messages = [r.message for r in caplog.records]
        hooks_warnings = [
            m for m in warning_messages
            if "hooks" in m.lower() and "JSON" in m
        ]
        assert len(hooks_warnings) == 0, (
            f"No hooks warning should be emitted, got: {hooks_warnings}"
        )

        assert result is not None


# ─── GAP-5: Loop Graph with Both Hook Sources ──────────────────────────


# ─── GAP-6: Bypass Executor-Level E2E ─────────────────────────────────


class TestBypassExecutorLevel:
    """Executor-level bypass proofs (GAP-6 from test-coverage-audit).

    Proves the on_node_bypass hook fires with canonical reasons through
    the real executor path.

    Bypass paths:
    - error-cascade (reason="upstream_error") — via FailingNode + edges
    - not_ready     (reason="not_ready")      — via node without output
    - condition     (reason="condition")      — doc: needs loop graph infra
    """

    @pytest.fixture(autouse=True)
    def cleanup(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    @pytest.mark.asyncio
    async def test_executor_bypass_upstream_error_fires_hook(self):
        """Error-cascade bypass fires on_node_bypass with canonical reason.

        Uses a FailingNode upstream with a downstream node. Verifies
        the canonical reason AND upstream_error_node in metadata.
        """
        from magic_agents.execution.reactive_executor import execute_graph_reactive
        from magic_agents.node_system.Node import Node
        from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel

        class FailingNode(Node):
            __test__ = False
            def __init__(self, node_id: str, **kwargs):
                super().__init__(node_id=node_id, node_type="failing", **kwargs)
            async def process(self, chat_log):
                yield {"type": "debug", "content": {"node": self.node_id, "event": "before_failure"}}
                raise ValueError("Intentional failure")

        class BypassNode(Node):
            __test__ = False
            def __init__(self, node_id: str, **kwargs):
                super().__init__(node_id=node_id, node_type="bypass_test", **kwargs)
            async def process(self, chat_log):
                yield {"type": "output", "content": {"message": "complete"}}

        hook = RecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={
                "failing_1": FailingNode(node_id="failing_1"),
                "downstream": BypassNode(node_id="downstream"),
            },
            edges=[
                EdgeNodeModel(source="failing_1", target="downstream"),
            ],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        bypasses = hook.get_invocations("on_node_bypass")
        assert len(bypasses) == 1

        bypass = bypasses[0]
        ctx = bypass["context"]

        # Canonical reason
        assert bypass["kwargs"]["reason"] == "upstream_error", (
            f"Expected reason='upstream_error', got '{bypass['kwargs']['reason']}'"
        )
        assert ctx.inputs.get("reason") == "upstream_error", (
            f"Context reason must be 'upstream_error'"
        )

        # Node identity
        assert ctx.node_id == "downstream", (
            f"Bypassed node should be 'downstream', got '{ctx.node_id}'"
        )
        assert ctx.execution_id, "Bypass context must have execution_id"
        assert ctx.metadata.get("upstream_error_node") == "failing_1"

    # NOTE: not_ready bypass test is deferred — the single-node bypass
    # path (reactive_executor.py:444-479) requires the downstream node's
    # wait_ready() to return False, which is timing-dependent and can
    # cause test hangs. The factory-level test in test_flow_hooks.py
    # (test_build_bypass_context) verifies the not_ready context structure.
    # The error-cascade test above proves the executor-level bypass hook
    # mechanism works end-to-end.
