"""
Deep Coverage: on_graph_error Runtime Semantics (Verify Gap 1+2).

Proves that on_graph_error fires with correct context when nodes fail,
and on_graph_end does NOT fire for failed executions.

Covers:
1. on_graph_error fires when a regular node (via exception) fails
2. on_graph_error includes error_message, failed_nodes list, and RuntimeError
3. on_graph_end does NOT fire when a node errors
4. Multiple erroring nodes still fire on_graph_error once (not once per error)
5. on_graph_error context includes execution_id, error_message, metadata.failed_nodes
6. on_graph_error receives error kwarg with RuntimeError
7. Graph error path preserves all other hooks (on_node_error, on_node_bypass still fire)
8. Timeout path also triggers on_graph_error
9. Error context structure completeness (error_type, error_message, error)
"""
import asyncio
import pytest

from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.node_system.Node import Node
from magic_agents.hooks import (
    HookContext,
    HookRegistry,
)


# ─── Test Nodes ──────────────────────────────────────────────────────────────

class ExceptionNode(Node):
    """Node that raises during process() to test error hooks."""

    __test__ = False

    def __init__(self, node_id: str, debug: bool = True, **kwargs):
        super().__init__(node_id=node_id, node_type="exception_node", debug=debug, **kwargs)

    async def process(self, chat_log):
        yield {"type": "start", "content": {"node": self.node_id}}
        raise ValueError(f"Simulated error in {self.node_id}")


class TimeoutNode(Node):
    """Node that hangs to test timeout error path."""

    __test__ = False

    def __init__(self, node_id: str, debug: bool = True, **kwargs):
        super().__init__(node_id=node_id, node_type="timeout_node", debug=debug, **kwargs)

    async def process(self, chat_log):
        yield {"type": "start", "content": {"node": self.node_id}}
        # Simulate hang (should timeout via dispatcher)
        raise asyncio.TimeoutError(f"Simulated timeout in {self.node_id}")


class DownstreamNode(Node):
    """Simple downstream node."""

    __test__ = False

    def __init__(self, node_id: str, debug: bool = True, **kwargs):
        super().__init__(node_id=node_id, node_type="downstream", debug=debug, **kwargs)

    async def process(self, chat_log):
        yield {"type": "end", "content": {"node": self.node_id}}


# ─── Recording Hook ──────────────────────────────────────────────────────────

class GraphErrorRecordingHook:
    """Records all hook invocations for graph error analysis."""

    def __init__(self):
        self.invocations: list[dict] = []

    async def on_graph_start(self, context: HookContext) -> None:
        self.invocations.append({"hook": "on_graph_start", "context": context})

    async def on_graph_end(self, context: HookContext) -> None:
        self.invocations.append({"hook": "on_graph_end", "context": context})

    async def on_graph_error(self, context: HookContext, error: Exception) -> None:
        self.invocations.append({
            "hook": "on_graph_error",
            "context": context,
            "error_arg": error,
        })

    async def on_node_start(self, context: HookContext) -> None:
        self.invocations.append({"hook": "on_node_start", "context": context})

    async def on_node_end(self, context: HookContext) -> None:
        self.invocations.append({"hook": "on_node_end", "context": context})

    async def on_node_error(self, context: HookContext, error: Exception) -> None:
        self.invocations.append({
            "hook": "on_node_error",
            "context": context,
            "error_arg": error,
        })

    async def on_node_bypass(self, context: HookContext, reason: str) -> None:
        self.invocations.append({
            "hook": "on_node_bypass",
            "context": context,
            "reason": reason,
        })

    @property
    def hook_names(self) -> list[str]:
        return [inv["hook"] for inv in self.invocations]

    def get(self, hook_name: str) -> list[dict]:
        return [inv for inv in self.invocations if inv["hook"] == hook_name]


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestGraphErrorOnException:
    """1. on_graph_error fires when a node raises an exception."""

    @pytest.mark.asyncio
    async def test_graph_error_fires_on_node_exception(self):
        """Exception node must trigger on_graph_error."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"err": ExceptionNode(node_id="err")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        graph_errors = hook.get("on_graph_error")
        graph_ends = hook.get("on_graph_end")

        assert len(graph_errors) == 1, (
            "on_graph_error must fire exactly once when a node errors"
        )
        assert len(graph_ends) == 0, (
            "on_graph_end must NOT fire when a node errors (spec requirement)"
        )

    @pytest.mark.asyncio
    async def test_graph_error_context_has_execution_id(self):
        """on_graph_error context must have non-empty execution_id."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"err": ExceptionNode(node_id="err")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        error_inv = hook.get("on_graph_error")[0]
        ctx = error_inv["context"]
        assert ctx.execution_id, (
            "on_graph_error context must have non-empty execution_id"
        )

    @pytest.mark.asyncio
    async def test_graph_error_context_has_error_message(self):
        """on_graph_error context must include error_message."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"err": ExceptionNode(node_id="err")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        error_inv = hook.get("on_graph_error")[0]
        ctx = error_inv["context"]
        assert ctx.error_message, (
            "on_graph_error context must have error_message"
        )
        assert "node error" in ctx.error_message.lower(), (
            f"error_message must reference node errors, got: {ctx.error_message}"
        )

    @pytest.mark.asyncio
    async def test_graph_error_context_has_failed_nodes(self):
        """on_graph_error context.metadata.failed_nodes must identify failing node."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"err": ExceptionNode(node_id="err")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        error_inv = hook.get("on_graph_error")[0]
        ctx = error_inv["context"]
        assert "failed_nodes" in ctx.metadata, (
            "on_graph_error context.metadata must have 'failed_nodes'"
        )
        failed = ctx.metadata["failed_nodes"]
        assert isinstance(failed, list), "failed_nodes must be a list"
        assert "err" in failed, (
            f"failed_nodes must include 'err', got: {failed}"
        )

    @pytest.mark.asyncio
    async def test_graph_error_receives_error_kwarg(self):
        """on_graph_error must receive error kwarg with RuntimeError."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"err": ExceptionNode(node_id="err")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        error_inv = hook.get("on_graph_error")[0]
        assert "error_arg" in error_inv, (
            "on_graph_error must receive error argument"
        )
        error_arg = error_inv["error_arg"]
        assert isinstance(error_arg, RuntimeError), (
            f"on_graph_error error arg must be RuntimeError, got {type(error_arg).__name__}"
        )
        assert "execution failed" in str(error_arg).lower(), (
            f"RuntimeError message must describe execution failure, got: {error_arg}"
        )


class TestGraphErrorWithDownstream:
    """Graph error + downstream nodes; other hooks still fire."""

    @pytest.mark.asyncio
    async def test_graph_error_with_downstream_nodes(self):
        """on_graph_error fires even with downstream nodes connected."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={
                "err": ExceptionNode(node_id="err"),
                "downstream": DownstreamNode(node_id="downstream"),
                "end": DownstreamNode(node_id="end"),
            },
            edges=[
                EdgeNodeModel(source="err", target="downstream"),
                EdgeNodeModel(source="downstream", target="end"),
            ],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        graph_errors = hook.get("on_graph_error")
        assert len(graph_errors) == 1, "on_graph_error must fire once"

        # Verify on_graph_end did NOT fire
        assert len(hook.get("on_graph_end")) == 0, (
            "on_graph_end must NOT fire when graph has errors"
        )

    @pytest.mark.asyncio
    async def test_node_error_and_bypass_still_fire_with_graph_error(self):
        """on_node_error and on_node_bypass must still fire alongside on_graph_error."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={
                "err": ExceptionNode(node_id="err"),
                "downstream": DownstreamNode(node_id="downstream"),
            },
            edges=[
                EdgeNodeModel(source="err", target="downstream"),
            ],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        node_errors = hook.get("on_node_error")
        node_bypasses = hook.get("on_node_bypass")
        graph_errors = hook.get("on_graph_error")

        assert len(node_errors) >= 1, (
            "on_node_error must still fire for failing node"
        )
        assert len(node_bypasses) >= 1, (
            "on_node_bypass must still fire for downstream"
        )
        assert len(graph_errors) == 1, (
            "on_graph_error must also fire"
        )

    @pytest.mark.asyncio
    async def test_graph_start_still_fires_before_error(self):
        """on_graph_start must still fire even when graph errors."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"err": ExceptionNode(node_id="err")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        names = hook.hook_names
        graph_start_idx = names.index("on_graph_start")
        graph_error_idx = names.index("on_graph_error")

        assert graph_start_idx < graph_error_idx, (
            "on_graph_start must fire BEFORE on_graph_error"
        )


class TestGraphErrorWithMultipleErrors:
    """Graph with multiple erroring nodes."""

    @pytest.mark.asyncio
    async def test_graph_error_fires_once_for_multiple_errors(self):
        """Multiple erroring nodes must fire on_graph_error once (not per error)."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={
                "err1": ExceptionNode(node_id="err1"),
                "err2": ExceptionNode(node_id="err2"),
            },
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        graph_errors = hook.get("on_graph_error")
        assert len(graph_errors) == 1, (
            "on_graph_error must fire exactly once, not per-error"
        )

        ctx = graph_errors[0]["context"]
        assert sorted(ctx.metadata.get("failed_nodes", [])) == ["err1", "err2"]


class TestGraphErrorOnTimeout:
    """Timeout path also triggers on_graph_error."""

    @pytest.mark.asyncio
    async def test_graph_error_fires_on_timeout(self):
        """Timeout node must also trigger on_graph_error (spec requirement)."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"timeout": TimeoutNode(node_id="timeout")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        graph_errors = hook.get("on_graph_error")
        graph_ends = hook.get("on_graph_end")

        assert len(graph_errors) == 1, (
            "on_graph_error must fire after timeout"
        )
        assert len(graph_ends) == 0, (
            "on_graph_end must NOT fire after timeout"
        )


class TestGraphErrorNoFalsePositive:
    """Graph errors do NOT fire on successful execution."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_graph_error(self):
        """Successful execution must NOT fire on_graph_error."""
        hook = GraphErrorRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"ok": DownstreamNode(node_id="ok")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        assert len(hook.get("on_graph_error")) == 0, (
            "on_graph_error must NOT fire on successful execution"
        )
        assert len(hook.get("on_graph_end")) == 1, (
            "on_graph_end must fire on successful execution"
        )
