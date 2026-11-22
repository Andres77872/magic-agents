"""
Async Bridge Tests: HookRelay pending future tracking and flush.

Tests the async bridge reliability improvements (Epic 2):
1. Pending future tracked after async invocation
2. flush_pending_hooks() completes all pending futures
3. Timeout does not crash (warning logged)
4. Error in pending future isolated (logged, not propagated)
5. Flush with no pending futures is a no-op
6. Multiple flush calls safe (idempotent)

Uses AsyncRecordingHook to ensure hooks go through the async bridge path.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from magic_agents.hooks.hook_relay import HookRelay
from magic_agents.hooks.hook_registry import HookRegistry
from tests.support.hooks import AsyncRecordingHook, RecordingHook


def agent_state(iteration: int = 0) -> SimpleNamespace:
    """Return the state fields HookRelay reads without mocking AgentState."""
    return SimpleNamespace(iteration=iteration)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up any global hooks between tests."""
    from magic_agents.hooks import RuntimeConfig
    RuntimeConfig.clear_global_hooks()
    yield
    RuntimeConfig.clear_global_hooks()


class TestPendingFutureTracking:
    """Tests for _pending_futures tracking in HookRelay."""

    @pytest.mark.asyncio
    async def test_pending_future_tracked_after_async_invocation(self):
        """After calling on_iteration_start, _pending_futures has 1 task."""
        hooks = AsyncRecordingHook(delay=0.01)
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        assert len(relay._pending_futures) == 0

        # Trigger an async invocation via _invoke_via_registry
        state = agent_state()
        relay.on_iteration_start(0, state)

        # Should have one pending future immediately
        assert len(relay._pending_futures) == 1, (
            f"Expected 1 pending future, got {len(relay._pending_futures)}"
        )

        # Flush to clean up
        await relay.flush_pending_hooks()
        assert len(relay._pending_futures) == 0

    @pytest.mark.asyncio
    async def test_multiple_invocations_tracked_independently(self):
        """Multiple hook invocations each add to _pending_futures."""
        hooks = AsyncRecordingHook(delay=0.01)
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        state = agent_state()

        relay.on_iteration_start(0, state)
        relay.on_iteration_start(1, state)
        relay.on_iteration_start(2, state)

        # Should have 3 pending futures
        assert len(relay._pending_futures) == 3

        await relay.flush_pending_hooks()
        assert len(relay._pending_futures) == 0

    @pytest.mark.asyncio
    async def test_completed_future_remains_tracked_until_flush(self):
        """Completed tasks remain tracked until the explicit flush boundary."""
        hooks = RecordingHook()  # No delay, fast completion
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        state = agent_state()

        relay.on_iteration_start(0, state)

        # Let the scheduled hook finish without crossing the explicit flush boundary.
        await asyncio.sleep(0.05)

        assert len(relay._pending_futures) == 1
        pending_task = next(iter(relay._pending_futures))
        assert pending_task.done()
        assert hooks.call_count("on_llm_start") == 1

        await relay.flush_pending_hooks()
        assert relay._pending_futures == set()


class TestFlushBehavior:
    """Tests for flush_pending_hooks() behavior."""

    @pytest.mark.asyncio
    async def test_flush_completes_all_pending_futures(self):
        """flush_pending_hooks completes all futures within timeout."""
        hooks = AsyncRecordingHook(delay=0.01)
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        state = agent_state()

        # Fire 3 async invocations
        relay.on_iteration_start(0, state)
        relay.on_iteration_start(1, state)
        relay.on_iteration_start(2, state)

        assert len(relay._pending_futures) == 3

        # Flush should complete all
        await relay.flush_pending_hooks()

        assert len(relay._pending_futures) == 0
        assert hooks.call_count("on_llm_start") == 3

    @pytest.mark.asyncio
    async def test_flush_with_no_pending_is_noop(self):
        """flush_pending_hooks with empty list is a no-op."""
        relay = HookRelay()
        assert relay._pending_futures == set()

        # Should not raise and return immediately
        await relay.flush_pending_hooks()
        assert relay._pending_futures == set()

        # Call again with empty list — still safe
        await relay.flush_pending_hooks()
        assert relay._pending_futures == set()

    @pytest.mark.asyncio
    async def test_flush_timeout_logs_warning(self, caplog):
        """Flush timeout logs a warning but does not crash."""
        hooks = AsyncRecordingHook(delay=5.0)  # Slow hooks
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1",
                         run_id="r-1", timeout=0.05)  # Very short timeout

        state = agent_state()

        relay.on_iteration_start(0, state)
        tracked_tasks = set(relay._pending_futures)
        assert len(tracked_tasks) == 1

        caplog.set_level(logging.WARNING)
        await relay.flush_pending_hooks()

        # Should have logged a warning about timeout
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        timeout_warnings = [m for m in warning_messages if "timed out" in m.lower()]
        assert timeout_warnings == [
            "1 pending hook(s) timed out after 0.1s. Cancelling."
        ]
        assert relay._pending_futures == set()

        # Observe cancellation so no timed-out task leaks beyond the test.
        await asyncio.gather(*tracked_tasks, return_exceptions=True)
        assert all(task.cancelled() for task in tracked_tasks)

    @pytest.mark.asyncio
    async def test_multiple_flush_calls_safe(self):
        """Multiple flush calls are idempotent and safe."""
        hooks = AsyncRecordingHook(delay=0.01)
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        state = agent_state()

        relay.on_iteration_start(0, state)

        # First flush
        await relay.flush_pending_hooks()
        assert len(relay._pending_futures) == 0

        # Second flush (no pending) — should be safe
        await relay.flush_pending_hooks()

        # Fire more and flush again
        relay.on_iteration_start(1, state)
        await relay.flush_pending_hooks()
        assert len(relay._pending_futures) == 0


class TestErrorIsolation:
    """Error in pending futures is isolated."""

    @pytest.mark.asyncio
    async def test_error_in_pending_future_isolated(self, caplog):
        """Error in an async hook does not propagate through flush."""
        class ErrorHook:
            """Hook that raises on on_llm_start."""
            async def on_llm_start(self, context, **kwargs):
                raise ValueError("Intentional hook error for testing")

        registry = HookRegistry()
        registry.register_graph(ErrorHook())
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        state = agent_state()
        caplog.set_level(logging.WARNING)

        relay.on_iteration_start(0, state)

        # flush should NOT raise even though hook raised
        await relay.flush_pending_hooks()

        # Subsequent invocations should still work
        relay.on_iteration_start(1, state)
        await relay.flush_pending_hooks()

        assert relay._pending_futures == set()
        hook_errors = [
            record.message
            for record in caplog.records
            if "Intentional hook error for testing" in record.message
        ]
        assert hook_errors == [
            "Hook ErrorHook.on_llm_start failed: Intentional hook error for testing",
            "Hook ErrorHook.on_llm_start failed: Intentional hook error for testing",
        ]

    @pytest.mark.asyncio
    async def test_error_does_not_block_other_hooks(self):
        """Error in one hook does not prevent other hooks from being called."""
        calls: list[str] = []

        class MixedHook:
            async def on_llm_start(self, context, **kwargs):
                calls.append("llm_start")
                raise ValueError("Hook error")
            async def on_tool_start(self, context, **kwargs):
                calls.append("tool_start")

        registry = HookRegistry()
        registry.register_graph(MixedHook())
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        state = agent_state()

        relay.on_iteration_start(0, state)
        relay.on_tool_start("tool", "call-1", {"q": "x"}, state)

        await relay.flush_pending_hooks()

        # Both hooks should have been invoked even if one raised
        assert calls == ["llm_start", "tool_start"]

    @pytest.mark.asyncio
    async def test_raising_async_hook_via_flow_hooks_path(self):
        """Async hook error via _invoke_async_method path is isolated."""
        class RaisingHook:
            async def on_llm_start(self, context, **kwargs):
                raise RuntimeError("Async hook failure")

        relay = HookRelay(flow_hooks=RaisingHook(), node_id="llm-1", graph_id="g-1")

        state = agent_state()

        relay.on_iteration_start(0, state)
        assert len(relay._pending_futures) == 1
        task = next(iter(relay._pending_futures))

        # Flush isolates the failure from the caller while retaining it on the task.
        await relay.flush_pending_hooks()
        assert relay._pending_futures == set()
        assert task.done()
        error = task.exception()
        assert isinstance(error, RuntimeError)
        assert str(error) == "Async hook failure"


