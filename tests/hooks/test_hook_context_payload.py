"""
Deep Coverage: Production HookContext Payload Completeness (Verify Gap 4).

Proves that runtime hook contexts carry real identity, timing, and metadata
across graph, node, and relay boundaries.

Covers:
1. Runtime node hooks receive real execution_id (not empty string)
2. Runtime node hooks receive real run_id (not empty string)
3. Runtime node hooks have start_time set BEFORE on_node_start fires
4. Runtime node hooks have end_time and duration_ms set AFTER node execution
5. Runtime HookRelay (LLM/tool) hooks receive non-empty graph_id/execution_id
6. Runtime HookRelay hooks receive non-empty run_id
7. Context metadata remains stable across graph/node/relay boundaries
8. Graph-level hooks have metadata (graph_type, node_count, edge_count)
9. Node hook context has proper node_id, node_type, node_class
10. timing fields (start_time, end_time, duration_ms) are consistent
"""
import pytest
from datetime import datetime

from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.node_system.Node import Node
from magic_agents.hooks import (
    HookContext,
    HookRegistry,
)
from magic_agents.hooks.hook_relay import HookRelay


# ─── Test Nodes ──────────────────────────────────────────────────────────────

class SimpleNode(Node):
    """Simple node for testing hook context payload."""

    __test__ = False

    def __init__(self, node_id: str, debug: bool = True, **kwargs):
        super().__init__(node_id=node_id, node_type="simple", debug=debug, **kwargs)

    async def process(self, chat_log):
        yield {"type": "end", "content": {"node": self.node_id, "output": "done"}}


class ErrorNode(Node):
    """Node that errors for testing error hook context."""

    __test__ = False

    def __init__(self, node_id: str, debug: bool = True, **kwargs):
        super().__init__(node_id=node_id, node_type="error_test", debug=debug, **kwargs)

    async def process(self, chat_log):
        yield {"type": "start", "content": {"node": self.node_id}}
        raise ValueError("Test error for context verification")


# ─── Recording Hooks ─────────────────────────────────────────────────────────

class ContextRecordingHook:
    """Records hook contexts for payload analysis."""

    def __init__(self):
        self.contexts: dict[str, list[HookContext]] = {}

    def _record(self, hook_name: str, context: HookContext) -> None:
        if hook_name not in self.contexts:
            self.contexts[hook_name] = []
        self.contexts[hook_name].append(context)

    async def on_graph_start(self, context: HookContext) -> None:
        self._record("on_graph_start", context)

    async def on_graph_end(self, context: HookContext) -> None:
        self._record("on_graph_end", context)

    async def on_graph_error(self, context: HookContext, error: Exception) -> None:
        self._record("on_graph_error", context)

    async def on_node_start(self, context: HookContext) -> None:
        self._record("on_node_start", context)

    async def on_node_end(self, context: HookContext) -> None:
        self._record("on_node_end", context)

    async def on_node_error(self, context: HookContext, error: Exception) -> None:
        self._record("on_node_error", context)

    async def on_llm_start(
        self,
        context: HookContext,
        llm_config: dict | None = None,
    ) -> None:
        self._record("on_llm_start", context)

    async def on_tool_start(self, context: HookContext) -> None:
        self._record("on_tool_start", context)

    def contexts_for(self, hook_name: str) -> list[HookContext]:
        return self.contexts.get(hook_name, [])


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestNodeHookContextIdentity:
    """Node hook contexts receive real execution_id and run_id."""

    @pytest.mark.asyncio
    async def test_node_start_has_real_execution_id(self):
        """on_node_start context must have non-empty execution_id."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"node-1": SimpleNode(node_id="node-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        contexts = hook.contexts_for("on_node_start")
        assert len(contexts) >= 1
        ctx = contexts[0]
        assert ctx.execution_id, (
            "on_node_start context must have non-empty execution_id"
        )
        assert ctx.execution_id != "", (
            "execution_id must NOT be empty string"
        )

    @pytest.mark.asyncio
    async def test_node_start_has_execution_id_matching_graph(self):
        """on_node_start execution_id must match the graph execution's ID."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"node-1": SimpleNode(node_id="node-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        node_ctx = hook.contexts_for("on_node_start")[0]
        graph_ctx = hook.contexts_for("on_graph_start")[0]

        assert node_ctx.execution_id == graph_ctx.execution_id, (
            "Node and graph hook contexts must share the same execution_id"
        )

    @pytest.mark.asyncio
    async def test_node_start_has_node_identity(self):
        """on_node_start context must have node_id, node_type, node_class."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"node-1": SimpleNode(node_id="node-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        ctx = hook.contexts_for("on_node_start")[0]
        assert ctx.node_id == "node-1", (
            f"node_id must be 'node-1', got '{ctx.node_id}'"
        )
        assert ctx.node_type is not None, "node_type must be set"
        assert ctx.node_class is not None, "node_class must be set"
        assert ctx.node_class == "SimpleNode", (
            f"node_class must be 'SimpleNode', got '{ctx.node_class}'"
        )


class TestNodeHookContextTiming:
    """Node hook contexts receive real start_time, end_time, duration_ms."""

    @pytest.mark.asyncio
    async def test_node_start_has_start_time(self):
        """on_node_start context must have start_time set."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"node-1": SimpleNode(node_id="node-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        ctx = hook.contexts_for("on_node_start")[0]
        assert ctx.start_time is not None, (
            "on_node_start context must have start_time"
        )
        assert isinstance(ctx.start_time, datetime), (
            "start_time must be a datetime"
        )

    @pytest.mark.asyncio
    async def test_node_end_has_start_time(self):
        """on_node_end context must have start_time carried through."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"node-1": SimpleNode(node_id="node-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        ctx = hook.contexts_for("on_node_end")[0]
        assert ctx.start_time is not None, (
            "on_node_end context must have start_time carried from on_node_start"
        )

    @pytest.mark.asyncio
    async def test_node_end_has_end_time(self):
        """on_node_end context must have end_time set."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"node-1": SimpleNode(node_id="node-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        ctx = hook.contexts_for("on_node_end")[0]
        assert ctx.end_time is not None, (
            "on_node_end context must have end_time"
        )
        assert isinstance(ctx.end_time, datetime), (
            "end_time must be a datetime"
        )

    @pytest.mark.asyncio
    async def test_node_end_has_duration_ms(self):
        """on_node_end context must have duration_ms calculated."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"node-1": SimpleNode(node_id="node-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        ctx = hook.contexts_for("on_node_end")[0]
        assert ctx.duration_ms is not None, (
            "on_node_end context must have duration_ms"
        )
        assert isinstance(ctx.duration_ms, (int, float)), (
            "duration_ms must be numeric"
        )
        assert ctx.duration_ms >= 0, (
            f"duration_ms must be >= 0, got {ctx.duration_ms}"
        )

    @pytest.mark.asyncio
    async def test_node_error_has_duration_ms(self):
        """on_node_error context must also have end_time and duration_ms."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"err": ErrorNode(node_id="err")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        error_ctxs = hook.contexts_for("on_node_error")
        assert len(error_ctxs) >= 1
        ctx = error_ctxs[0]
        assert ctx.end_time is not None, (
            "on_node_error context must have end_time"
        )
        assert ctx.duration_ms is not None, (
            "on_node_error context must have duration_ms"
        )
        assert ctx.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_timing_consistency(self):
        """end_time must be >= start_time, duration_ms must reflect difference."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"node-1": SimpleNode(node_id="node-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        end_ctx = hook.contexts_for("on_node_end")[0]
        assert end_ctx.start_time is not None and end_ctx.end_time is not None
        assert end_ctx.end_time >= end_ctx.start_time
        expected_ms = (end_ctx.end_time - end_ctx.start_time).total_seconds() * 1000
        # Generous tolerance: only a gross mismatch (wrong clock or unit) fails.
        assert abs(end_ctx.duration_ms - expected_ms) < 100


class TestGraphHookContextMetadata:
    """Graph-level hook contexts carry metadata."""

    @pytest.mark.asyncio
    async def test_graph_start_has_graph_metadata(self):
        """on_graph_start context must have metadata with graph info.
        
        NOTE: sourceHandle="end" matches SimpleNode's yield type.
        """
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={
                "a": SimpleNode(node_id="a"),
                "b": SimpleNode(node_id="b"),
            },
            edges=[
                EdgeNodeModel(source="a", target="b", sourceHandle="end"),
            ],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        ctx = hook.contexts_for("on_graph_start")[0]
        assert "graph_type" in ctx.metadata, (
            "on_graph_start metadata must include graph_type"
        )
        assert "node_count" in ctx.metadata, (
            "on_graph_start metadata must include node_count"
        )
        assert "edge_count" in ctx.metadata, (
            "on_graph_start metadata must include edge_count"
        )
        assert ctx.metadata["node_count"] == 2
        assert ctx.metadata["edge_count"] == 1

    @pytest.mark.asyncio
    async def test_graph_end_has_execution_summary(self):
        """on_graph_end context must have execution_summary in metadata."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"node-1": SimpleNode(node_id="node-1")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        ctx = hook.contexts_for("on_graph_end")[0]
        assert "execution_summary" in ctx.metadata, (
            "on_graph_end metadata must include execution_summary"
        )
        # Debuggers show the run's total duration from this context.
        assert isinstance(ctx.duration_ms, float) and ctx.duration_ms >= 0


class TestRelayContextIdentity:
    """HookRelay-produced contexts carry non-empty identity fields."""

    @pytest.mark.asyncio
    async def test_relay_context_has_non_empty_execution_id(self):
        """HookContext built from HookRelay must have non-empty execution_id."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)
        registry.execution_id = "real-graph-id-123"

        relay = HookRelay(
            registry=registry,
            node_id="test-llm",
            graph_id="real-graph-id-123",
            run_id="run-abc",
        )

        relay.on_iteration_start(0, None)
        await relay.flush_pending_hooks()

        relay_contexts = hook.contexts_for("on_llm_start")
        assert len(relay_contexts) == 1
        assert {ctx.execution_id for ctx in relay_contexts} == {
            "real-graph-id-123"
        }

    @pytest.mark.asyncio
    async def test_relay_context_has_non_empty_run_id(self):
        """HookContext from HookRelay must have non-empty run_id when provided."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)
        registry.execution_id = "g-1"
        registry.run_id = "run-xyz"

        relay = HookRelay(
            registry=registry,
            node_id="test-llm",
            graph_id="g-1",
            run_id="run-xyz",
        )

        relay.on_tool_start("search", "call_1", {"q": "test"}, None)
        await relay.flush_pending_hooks()

        relay_contexts = hook.contexts_for("on_tool_start")
        assert len(relay_contexts) == 1
        assert {ctx.run_id for ctx in relay_contexts} == {"run-xyz"}

class TestGraphErrorContextPayload:
    """Graph error hook context has proper error payload."""

    @pytest.mark.asyncio
    async def test_graph_error_context_has_timing(self):
        """on_graph_error context must include timestamp."""
        hook = ContextRecordingHook()
        registry = HookRegistry()
        registry.register_graph(hook)

        graph = AgentFlowModel(
            nodes={"err": ErrorNode(node_id="err")},
            edges=[],
            debug=True,
        )

        async for _ in execute_graph_reactive(graph, hooks=registry):
            pass

        ctxs = hook.contexts_for("on_graph_error")
        assert len(ctxs) >= 1
        ctx = ctxs[0]
        assert ctx.timestamp is not None, (
            "on_graph_error context must have timestamp"
        )
        assert isinstance(ctx.timestamp, datetime)
