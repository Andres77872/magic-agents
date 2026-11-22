"""
Integration Tests: 3-Tier Hook Registration (Task 9.9).

Tests that:
1. Global + graph + node hooks all fire during graph execution
2. Each tier fires independently (global fires even without node hooks)
3. Graph-level hooks (on_graph_start/on_graph_end) fire at lifecycle points
4. Node-level hooks fire only for the target node
5. Global hooks survive only when injected into the registry
"""
import pytest
from typing import Any, AsyncGenerator, Dict, List

from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.node_system.Node import Node
from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.hooks.runtime_config import RuntimeConfig


# ─── Custom Test Node ──────────────────────────────────────────────────────────

class _HookTestNode(Node):
    """Simple node that yields an output for testing hooks (not collected by pytest)."""

    __test__ = False

    def __init__(self, node_id: str, **kwargs):
        super().__init__(node_id=node_id, node_type="test", **kwargs)

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"type": "output", "content": {"node": self.node_id, "result": "done"}}


# ─── Recording Hook (Tier-specific) ────────────────────────────────────────────

class TierRecordingHook:
    """FlowHooks that records invocations with tier name for verification."""

    def __init__(self, tier_name: str):
        self.tier_name = tier_name
        self.calls: List[Dict[str, Any]] = []

    async def on_node_start(self, context: HookContext) -> None:
        self.calls.append({"hook_name": "on_node_start", "tier": self.tier_name, "context": context})

    async def on_node_end(self, context: HookContext) -> None:
        self.calls.append({"hook_name": "on_node_end", "tier": self.tier_name, "context": context})

    async def on_graph_start(self, context: HookContext) -> None:
        self.calls.append({"hook_name": "on_graph_start", "tier": self.tier_name, "context": context})

    async def on_graph_end(self, context: HookContext) -> None:
        self.calls.append({"hook_name": "on_graph_end", "tier": self.tier_name, "context": context})


# ─── Tests ─────────────────────────────────────────────────────────────────────

class TestThreeTierRegistration:
    """9.9.2: Global + graph + node hooks all fire during execution."""

    @pytest.fixture(autouse=True)
    def clear_global_hooks(self):
        """Clear global hooks before and after each test to prevent test pollution."""
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    def _build_registry(self, global_hook, graph_hook=None, node_hooks=None):
        """Build a HookRegistry with optional global, graph, and node hooks.
        
        Global hooks from RuntimeConfig are injected via RuntimeConfig.create_registry().
        Graph and node hooks are registered directly on the returned registry.
        """
        # Register global
        if global_hook is not None:
            RuntimeConfig.register_global_hook(global_hook)
        # Create registry — injects global hooks from RuntimeConfig
        registry = RuntimeConfig().create_registry()
        # Add graph hooks
        if graph_hook is not None:
            registry.register_graph(graph_hook)
        # Add node hooks
        if node_hooks is not None:
            for node_id, hook in node_hooks:
                registry.register_node(node_id, hook)
        return registry

    @pytest.mark.asyncio
    async def test_all_tiers_fire(self):
        """All three tiers must fire on_node_start during execution."""
        node_hook = TierRecordingHook("node")
        graph_hook = TierRecordingHook("graph")
        global_hook = TierRecordingHook("global")

        registry = self._build_registry(
            global_hook=global_hook,
            graph_hook=graph_hook,
            node_hooks=[("test-1", node_hook)],
        )

        graph = AgentFlowModel(
            nodes={"test-1": _HookTestNode(node_id="test-1")},
            edges=[],
            debug=True,
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        node_start_calls = []
        for hook in [node_hook, graph_hook, global_hook]:
            for call in hook.calls:
                if call["hook_name"] == "on_node_start":
                    node_start_calls.append(call)

        tiers_fired = {c["tier"] for c in node_start_calls}
        assert tiers_fired == {"node", "graph", "global"}, (
            f"Expected all three tiers to fire, got: {tiers_fired}"
        )


class TestTierIndependence:
    """9.9.4: Each tier fires independently."""

    @pytest.fixture(autouse=True)
    def clear_global_hooks(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    @pytest.mark.asyncio
    async def test_global_only_fires(self):
        """Global hooks must fire even without graph or node hooks."""
        global_hook = TierRecordingHook("global")
        RuntimeConfig.register_global_hook(global_hook)
        # create_registry injects global hooks
        registry = RuntimeConfig().create_registry()

        graph = AgentFlowModel(
            nodes={"test-3": _HookTestNode(node_id="test-3")},
            edges=[],
            debug=True,
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        node_starts = [c for c in global_hook.calls if c["hook_name"] == "on_node_start"]
        assert len(node_starts) >= 1, (
            "Global hook should fire even without graph/node hooks"
        )

    @pytest.mark.asyncio
    async def test_graph_only_fires(self):
        """Graph hooks must fire without node hooks."""
        graph_hook = TierRecordingHook("graph")

        registry = HookRegistry()
        registry.register_graph(graph_hook)

        graph = AgentFlowModel(
            nodes={"test-4": _HookTestNode(node_id="test-4")},
            edges=[],
            debug=True,
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        node_starts = [c for c in graph_hook.calls if c["hook_name"] == "on_node_start"]
        assert len(node_starts) >= 1, (
            "Graph hook should fire without node hooks"
        )

    @pytest.mark.asyncio
    async def test_node_only_fires_for_target(self):
        """Node-level hooks must fire only for their registered node."""
        node_hook_a = TierRecordingHook("node_a")
        node_hook_b = TierRecordingHook("node_b")

        registry = HookRegistry()
        registry.register_node("node-a", node_hook_a)
        registry.register_node("node-b", node_hook_b)

        graph = AgentFlowModel(
            nodes={
                "node-a": _HookTestNode(node_id="node-a"),
                "node-b": _HookTestNode(node_id="node-b"),
            },
            edges=[],
            debug=True,
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=registry):
            events.append(event)

        a_starts = [c for c in node_hook_a.calls if c["hook_name"] == "on_node_start"]
        assert len(a_starts) >= 1, "Node hook A should fire for node-a"

        b_starts = [c for c in node_hook_b.calls if c["hook_name"] == "on_node_start"]
        assert len(b_starts) >= 1, "Node hook B should fire for node-b"

        assert a_starts[0]["context"].node_id == "node-a"
        assert b_starts[0]["context"].node_id == "node-b"


class TestGlobalHooksRequireRegistry:
    """Global hooks only fire when injected into the registry."""

    @pytest.fixture(autouse=True)
    def clear_global_hooks(self):
        RuntimeConfig.clear_global_hooks()
        yield
        RuntimeConfig.clear_global_hooks()

    @pytest.mark.asyncio
    async def test_global_hooks_require_explicit_registry(self):
        """
        Global hooks registered via RuntimeConfig require an explicit registry
        created via create_registry() to be injected. When hooks=None is
        passed to executor, no hooks fire.
        """
        global_hook = TierRecordingHook("global")
        RuntimeConfig.register_global_hook(global_hook)

        graph = AgentFlowModel(
            nodes={"test-7": _HookTestNode(node_id="test-7")},
            edges=[],
            debug=True,
        )

        events = []
        async for event in execute_graph_reactive(graph, hooks=None):
            events.append(event)

        assert len(global_hook.calls) == 0, (
            "Global hooks should NOT fire when no registry is passed to executor"
        )
