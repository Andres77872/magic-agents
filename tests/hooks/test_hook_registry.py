"""Unit tests for HookRegistry 3-tier hook dispatch.

Phase 9.2: Tests for:
- 3-tier registration works (global, graph, node)
- Execution order Node → Graph → Global
- Error isolation (hook exception doesn't propagate)
- Empty registry check is_empty()
- Sync hooks wrapped via asyncio.to_thread
- Multiple hooks at same tier execute in parallel
"""
import pytest

from magic_agents.util.aio import wait_for
import asyncio

from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.context_factory import HookContextFactory
from magic_agents.hooks.hook_registry import HookRegistry


def context(execution_id: str = "test-1", node_id: str | None = None) -> HookContext:
    """Build a registry invocation context through the supported factory."""
    return HookContextFactory.build_node_context(
        execution_id=execution_id,
        node_id=node_id,
    )


# ─── Helper: A FlowHooks-compatible hook that tracks invocations ──────────────


class TrackingHook:
    """Hook that records invocations for order/execution verification."""

    def __init__(self, name: str = "hook"):
        self.name = name
        self.calls: list[tuple[str, str]] = []  # (hook_name, tier_context)

    async def on_node_start(self, context: HookContext) -> None:
        self.calls.append(("on_node_start", context.metadata.get("tier", "unknown")))

    async def on_node_end(self, context: HookContext) -> None:
        self.calls.append(("on_node_end", context.metadata.get("tier", "unknown")))

    async def on_graph_start(self, context: HookContext) -> None:
        self.calls.append(("on_graph_start", context.metadata.get("tier", "unknown")))

    async def on_graph_end(self, context: HookContext) -> None:
        self.calls.append(("on_graph_end", context.metadata.get("tier", "unknown")))

    async def on_node_error(self, context: HookContext, error: Exception) -> None:
        self.calls.append(("on_node_error", str(error)))

    async def on_node_bypass(self, context: HookContext, reason: str) -> None:
        self.calls.append(("on_node_bypass", reason))


# ─── Helper: A FlowHooks-compatible sync hook ────────────────────────────────


class SyncTrackingHook:
    """Sync hook (not async) that records invocations."""

    def __init__(self, name: str = "sync"):
        self.name = name
        self.calls: list[str] = []

    def on_node_start(self, context: HookContext) -> None:
        self.calls.append("sync_on_node_start")


class FailingHook:
    """Hook that raises an exception to test error isolation."""

    async def on_node_start(self, context: HookContext) -> None:
        msg = "intentional hook failure"
        raise ValueError(msg)


@pytest.mark.asyncio
class TestHookRegistryRegistration:
    """Tests for 3-tier registration."""

    async def test_register_global_adds_to_global_tier(self):
        """register_global adds hooks to the global tier."""
        registry = HookRegistry()
        hook = TrackingHook("global")
        registry.register_global(hook)

        assert registry.get_global_hooks_count() == 1
        assert registry.get_total_hooks_count() == 1

    async def test_register_graph_adds_to_graph_tier(self):
        """register_graph adds hooks to the graph tier."""
        registry = HookRegistry()
        hook = TrackingHook("graph")
        registry.register_graph(hook)

        assert registry.get_graph_hooks_count() == 1
        assert registry.get_total_hooks_count() == 1

    async def test_register_node_adds_to_node_tier(self):
        """register_node adds hooks to the node-specific tier."""
        registry = HookRegistry()
        hook = TrackingHook("node")
        registry.register_node("node-1", hook)

        assert registry.get_node_hooks_count() == 1
        assert registry.get_total_hooks_count() == 1

    async def test_multiple_hooks_same_node(self):
        """Multiple hooks can be registered for the same node."""
        registry = HookRegistry()
        registry.register_node("node-1", TrackingHook("a"))
        registry.register_node("node-1", TrackingHook("b"))

        assert registry.get_node_hooks_count() == 2

    async def test_multiple_nodes_each_with_hooks(self):
        """Different nodes can each have their own hooks."""
        registry = HookRegistry()
        registry.register_node("node-1", TrackingHook("a"))
        registry.register_node("node-2", TrackingHook("b"))
        registry.register_node("node-2", TrackingHook("c"))

        assert registry.get_node_hooks_count() == 3

    async def test_empty_registry_has_zero_counts(self):
        """Fresh registry has zero hooks at all tiers."""
        registry = HookRegistry()
        assert registry.get_global_hooks_count() == 0
        assert registry.get_graph_hooks_count() == 0
        assert registry.get_node_hooks_count() == 0
        assert registry.get_total_hooks_count() == 0

    async def test_is_empty_returns_true_for_fresh_registry(self):
        """is_empty() returns True when no hooks are registered."""
        assert HookRegistry().is_empty() is True

    async def test_is_empty_returns_false_with_global_hook(self):
        """is_empty() returns False when global hooks exist."""
        registry = HookRegistry()
        registry.register_global(TrackingHook())
        assert registry.is_empty() is False

    async def test_is_empty_returns_false_with_graph_hook(self):
        """is_empty() returns False when graph hooks exist."""
        registry = HookRegistry()
        registry.register_graph(TrackingHook())
        assert registry.is_empty() is False

    async def test_is_empty_returns_false_with_node_hook(self):
        """is_empty() returns False when node hooks exist."""
        registry = HookRegistry()
        registry.register_node("n1", TrackingHook())
        assert registry.is_empty() is False

    async def test_init_with_global_hooks(self):
        """Constructor accepts pre-registered global hooks."""
        hook = TrackingHook("pre-registered")
        registry = HookRegistry(global_hooks=[hook])

        assert registry.get_global_hooks_count() == 1
        assert registry.is_empty() is False


@pytest.mark.asyncio
class TestHookRegistryExecutionOrder:
    """Tests for execution order: Node → Graph → Global."""

    async def test_invoke_returns_none_when_no_hooks(self):
        """invoke() returns None when no hooks are registered (silent no-op)."""
        registry = HookRegistry()
        ctx = context(node_id="node-1")

        result = await registry.invoke("on_node_start", ctx)

        assert result is None

    async def test_all_three_tiers_fire_for_a_node_event(self):
        """A node event reaches node, graph and global hooks (order: see tier sequence test)."""
        registry = HookRegistry()
        node_hook = TrackingHook("node")
        graph_hook = TrackingHook("graph")
        global_hook = TrackingHook("global")

        registry.register_node("node-1", node_hook)
        registry.register_graph(graph_hook)
        registry.register_global(global_hook)

        ctx = context(node_id="node-1")

        await registry.invoke("on_node_start", ctx)

        # All three hooks should have fired
        assert len(node_hook.calls) == 1
        assert len(graph_hook.calls) == 1
        assert len(global_hook.calls) == 1

    async def test_node_hook_not_invoked_for_different_node(self):
        """Node-level hooks only fire for the target node_id."""
        registry = HookRegistry()
        node_hook = TrackingHook("node-a")
        registry.register_node("node-a", node_hook)

        # Invoke for a different node
        ctx = context(node_id="node-b")
        await registry.invoke("on_node_start", ctx)

        assert len(node_hook.calls) == 0, (
            "Node hook should not fire for a different node_id"
        )

    async def test_graph_and_global_fire_regardless_of_node_id(self):
        """Graph and global hooks fire even without a node_id in context."""
        registry = HookRegistry()
        graph_hook = TrackingHook("graph")
        global_hook = TrackingHook("global")

        registry.register_graph(graph_hook)
        registry.register_global(global_hook)

        ctx = context()
        await registry.invoke("on_graph_start", ctx)

        assert len(graph_hook.calls) == 1, (
            "Graph hooks should fire without node_id"
        )
        assert len(global_hook.calls) == 1, (
            "Global hooks should fire without node_id"
        )

    async def test_tier_hooks_fire_in_correct_sequence(self):
        """All three tiers fire when hooks are registered at each tier."""
        registry = HookRegistry()
        order: list[str] = []

        class OrderedNodeHook:
            async def on_node_start(self, context):
                order.append("node")

        class OrderedGraphHook:
            async def on_node_start(self, context):
                order.append("graph")

        class OrderedGlobalHook:
            async def on_node_start(self, context):
                order.append("global")

        registry.register_node("n1", OrderedNodeHook())
        registry.register_graph(OrderedGraphHook())
        registry.register_global(OrderedGlobalHook())

        ctx = context(node_id="n1")
        await registry.invoke("on_node_start", ctx)

        assert order == ["node", "graph", "global"], (
            f"Expected order: node → graph → global, got: {order}"
        )

    async def test_multiple_hooks_at_same_tier(self):
        """Multiple hooks at the same tier all fire."""
        registry = HookRegistry()
        a = TrackingHook("a")
        b = TrackingHook("b")
        registry.register_global(a)
        registry.register_global(b)

        ctx = context()
        await registry.invoke("on_graph_start", ctx)

        assert len(a.calls) == 1
        assert len(b.calls) == 1

    async def test_only_implemented_methods_fire(self):
        """A partial hook gets the events it implements; the rest are silent no-ops."""
        registry = HookRegistry()
        calls = []

        class PartialHook:
            async def on_graph_start(self, context):
                calls.append("on_graph_start")

        registry.register_global(PartialHook())
        ctx = context()

        await registry.invoke("on_graph_start", ctx)
        await registry.invoke("on_graph_end", ctx)

        assert calls == ["on_graph_start"]


@pytest.mark.asyncio
class TestHookRegistryErrorIsolation:
    """Tests for error isolation in hook invocation."""

    async def test_other_hooks_still_fire_after_failure(self):
        """When one hook fails, other hooks still execute."""
        registry = HookRegistry()
        good = TrackingHook("good")
        bad = FailingHook()

        registry.register_global(good)
        registry.register_global(bad)

        ctx = context()

        # Should not raise
        await registry.invoke("on_node_start", ctx)

        # Good hook should still have fired
        assert len(good.calls) == 1, (
            "Good hook should fire even when another hook fails"
        )

    async def test_node_hooks_fire_after_graph_hook_failure(self):
        """Earlier tier failures should not prevent later tiers from executing."""
        registry = HookRegistry()

        class GraphFailingHook:
            async def on_node_start(self, context):
                msg = "graph failure"
                raise RuntimeError(msg)

        graph_check = TrackingHook("graph-check")
        global_check = TrackingHook("global-check")

        registry.register_node("n1", graph_check)
        registry.register_graph(GraphFailingHook())
        registry.register_global(global_check)

        ctx = context(node_id="n1")
        await registry.invoke("on_node_start", ctx)

        assert len(graph_check.calls) == 1, (
            "Node-level hook should fire before the failing graph hook"
        )
        # Global hooks fire after graph, and since asyncio.gather runs ALL tasks,
        # all hooks should complete regardless of failures
        assert len(global_check.calls) == 1, (
            "Global hooks should fire even after graph hook failure"
        )

    async def test_sync_hook_exception_isolated(self):
        """A raising sync hook neither propagates nor stops the other hooks."""

        class SyncFailingHook:
            def on_node_start(self, context):
                msg = "sync fail"
                raise RuntimeError(msg)

        registry = HookRegistry()
        good = TrackingHook("good")
        registry.register_global(SyncFailingHook())
        registry.register_global(good)

        await registry.invoke("on_node_start", context())

        assert len(good.calls) == 1


@pytest.mark.asyncio
class TestHookRegistrySyncHooks:
    """Tests for sync hook wrapping via asyncio.to_thread."""

    async def test_sync_hook_executes_correctly(self):
        """Sync hooks execute and their side effects are visible."""
        registry = HookRegistry()
        hook = SyncTrackingHook("sync")
        registry.register_global(hook)

        ctx = context()

        await registry.invoke("on_node_start", ctx)

        assert hook.calls == ["sync_on_node_start"], (
            "Sync hook side effects should be visible after invoke"
        )

    async def test_mixed_async_and_sync_hooks(self):
        """Async and sync hooks can be mixed; all execute."""
        registry = HookRegistry()
        results: list[str] = []

        class MySyncHook:
            def on_node_start(self, context):
                results.append("sync")

        class MyAsyncHook:
            async def on_node_start(self, context):
                results.append("async")

        registry.register_global(MySyncHook())
        registry.register_global(MyAsyncHook())

        ctx = context()
        await registry.invoke("on_node_start", ctx)

        assert "sync" in results
        assert "async" in results

    async def test_node_hook_with_error_and_kwargs(self):
        """Invoke passes **kwargs through to the hook method."""
        registry = HookRegistry()
        captured = {}

        class KwargHook:
            async def on_node_error(self, context, error):
                captured["error"] = error
                captured["node_id"] = context.node_id

        registry.register_global(KwargHook())
        ctx = context(node_id="n1")
        error = ValueError("test error")

        await registry.invoke("on_node_error", ctx, error=error)

        assert captured.get("error") is error
        assert captured.get("node_id") == "n1"


@pytest.mark.asyncio
class TestHookRegistryParallelExecution:
    """Tests for parallel execution of hooks at the same tier."""

    async def test_multiple_hooks_execute_in_parallel(self):
        """Same-tier hooks run concurrently: each waits for all of them to have started."""
        registry = HookRegistry()
        expected = 3
        started = []
        all_started = asyncio.Event()

        class BarrierHook:
            async def on_node_start(self, context):
                started.append(self)
                if len(started) == expected:
                    all_started.set()
                await all_started.wait()

        for _ in range(expected):
            registry.register_global(BarrierHook())

        # Serial execution would block forever in the first hook.
        await wait_for(registry.invoke("on_node_start", context()), 2)

        assert len(started) == expected
