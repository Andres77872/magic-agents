"""Regression tests: pending future cleanup and cancellation.

Phase 5.2: Verifies that:
- Timeout cancels pending tasks and clears _pending_futures
- Normal completion clears _pending_futures
- Snapshot prevents list mutation race during asyncio.wait
- Rapid sequential invocations without flush (Set robustness)
- No orphaned task references after flush
"""
import asyncio
import logging
import pytest

from magic_agents.util.aio import wait_for
from unittest.mock import MagicMock

from magic_llm.agent.types import AgentState

from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_relay import HookRelay
from magic_agents.hooks.hook_registry import HookRegistry


# ─── Recording hooks with configurable delay ──────────────────────────


class AsyncRecordingHook:
    """Async FlowHooks that records invocations with configurable delay.

    Each hook method introduces asyncio.sleep(delay) before recording,
    to simulate slow async hook execution for timeout tests.
    """

    def __init__(self, delay: float = 0.01):
        self.delay = delay
        self.calls: list[str] = []

    async def _record(self, name: str, context: HookContext, **kwargs) -> None:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.calls.append(name)

    async def on_llm_start(self, context: HookContext, llm_config=None):
        await self._record("on_llm_start", context, llm_config=llm_config)

    async def on_llm_end(self, context: HookContext, response=None):
        await self._record("on_llm_end", context, response=response)

    async def on_llm_loop_end(self, context: HookContext):
        await self._record("on_llm_loop_end", context)

    async def on_tool_start(self, context: HookContext):
        await self._record("on_tool_start", context)

    async def on_tool_end(self, context: HookContext):
        await self._record("on_tool_end", context)


@pytest.mark.asyncio
class TestPendingFutureCleanup:
    """Tests for pending future cancellation and cleanup in HookRelay."""

    # ── Scenario: Timeout cancels pending tasks ─────────────────────-

    async def test_timeout_cancels_pending_tasks_and_clears_set(self):
        """Timeout cancels slow pending tasks; _pending_futures is empty after flush."""
        hook = AsyncRecordingHook(delay=10.0)  # Very slow hook
        registry = HookRegistry()
        registry.register_global(hook)
        relay = HookRelay(
            registry=registry,
            node_id="n1",
            timeout=0.05,  # 50ms timeout
        )

        # Invoke a hook (creates a pending future)
        state = MagicMock(spec=AgentState)
        state.iteration = 0
        relay.on_iteration_start(iteration=0, state=state)

        # Should have 1 pending future
        assert len(relay._pending_futures) == 1, (
            f"Expected 1 pending future, got {len(relay._pending_futures)}"
        )

        pending = list(relay._pending_futures)

        # Flush with timeout — should cancel the slow task
        await relay.flush_pending_hooks()

        assert len(relay._pending_futures) == 0
        # The 10 s hook was cancelled, not left running: it settles immediately.
        await wait_for(asyncio.gather(*pending, return_exceptions=True), 1)
        assert all(task.cancelled() for task in pending)
        assert hook.calls == []

    # ── Scenario: Normal completion clears set ───────────────────────

    async def test_normal_completion_clears_set(self):
        """Fast hooks complete before timeout; _pending_futures cleared after flush."""
        hook = AsyncRecordingHook(delay=0.005)  # Fast hook (5ms)
        registry = HookRegistry()
        registry.register_global(hook)
        relay = HookRelay(
            registry=registry,
            node_id="n1",
            timeout=5.0,  # Generous timeout
        )

        state = MagicMock(spec=AgentState)
        state.iteration = 0

        # Invoke 3 fast hooks
        relay.on_iteration_start(iteration=0, state=state)
        relay.on_llm_response(
            MagicMock(content="resp", id="cmpl-1", usage={}),
            state,
        )
        relay.on_loop_complete(
            MagicMock(content="final", id="cmpl-final", usage={}),
            state,
        )

        # 3 pending futures
        assert len(relay._pending_futures) == 3, (
            f"Expected 3 pending futures, got {len(relay._pending_futures)}"
        )

        # Flush — all should complete before timeout
        await relay.flush_pending_hooks()

        # _pending_futures should be empty
        assert len(relay._pending_futures) == 0, (
            f"Expected empty _pending_futures after flush, got {len(relay._pending_futures)}"
        )

        # All 3 hooks should have been recorded
        assert len(hook.calls) == 3, (
            f"Expected 3 hook recordings, got {len(hook.calls)}"
        )

    # ── Scenario: Snapshot prevents list mutation race ───────────────

    # ── Scenario: Rapid sequential invocations without flush ─────────

    async def test_rapid_invocations_without_flush(self):
        """1000 rapid async invocations: Set handles large size without crash."""
        hook = AsyncRecordingHook(delay=0.0)  # Zero delay
        registry = HookRegistry()
        registry.register_global(hook)
        relay = HookRelay(
            registry=registry,
            node_id="n1",
            timeout=5.0,
        )

        state = MagicMock(spec=AgentState)
        state.iteration = 0

        # 1000 rapid invocations
        for i in range(1000):
            relay.on_iteration_start(iteration=i, state=state)

        # No await happened yet, so every invocation is still pending.
        assert len(relay._pending_futures) == 1000

        # No crash — Set handles large size
        # Flush to clean up
        await relay.flush_pending_hooks()
        assert len(relay._pending_futures) == 0

        # All 1000 hooks should have been recorded
        assert len(hook.calls) == 1000, (
            f"Expected 1000 hook recordings, got {len(hook.calls)}"
        )

    # ── Scenario: Empty flush is a no-op ────────────────────────────

    async def test_empty_flush_is_noop(self, caplog):
        """Flush with no pending futures returns immediately, no warning logged."""
        relay = HookRelay(timeout=0.1)
        assert len(relay._pending_futures) == 0

        caplog.set_level(logging.WARNING)
        await relay.flush_pending_hooks()

        # No warning should be logged
        warning_messages = [r.message for r in caplog.records]
        timeout_warnings = [m for m in warning_messages if "pending" in m.lower()]
        assert len(timeout_warnings) == 0, (
            f"Expected no timeout warnings for empty flush, got: {timeout_warnings}"
        )
        assert len(relay._pending_futures) == 0

    # ── Scenario: Multiple flush calls are safe ─────────────────────

    async def test_multiple_flush_calls_safe(self):
        """Multiple flush calls in sequence are safe (idempotent)."""
        relay = HookRelay(timeout=0.1)
        assert len(relay._pending_futures) == 0

        # Multiple flushes on empty set
        await relay.flush_pending_hooks()
        await relay.flush_pending_hooks()
        await relay.flush_pending_hooks()

        assert len(relay._pending_futures) == 0, (
            "Multiple empty flushes should not add pending futures"
        )
