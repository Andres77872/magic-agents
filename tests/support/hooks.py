"""Recording ``FlowHooks`` implementations shared by the hook tests.

Register one through ``HookRegistry`` (or ``RuntimeConfig``) and assert on the
recorded invocations afterwards::

    hook = RecordingHook()
    registry.register_graph(hook)
    ...
    hook.assert_method_called("on_node_start", times=3)
    hook.assert_context_field("on_node_start", "node_id", "test-1")
"""

from __future__ import annotations

import asyncio
from typing import Any


class RecordingHook:
    """Async ``FlowHooks`` implementation that records every invocation."""

    def __init__(self, execution_id: str = "", run_id: str = "") -> None:
        self.invocations: list[dict[str, Any]] = []
        self.execution_id = execution_id
        self.run_id = run_id

    async def _record(self, hook_name: str, context: Any, **kwargs: Any) -> None:
        self.invocations.append({"hook_name": hook_name, "context": context, "kwargs": kwargs})

    # ── Queries ────────────────────────────────────────────────────────

    def get_invocations(self, name: str) -> list[dict[str, Any]]:
        return [inv for inv in self.invocations if inv["hook_name"] == name]

    def call_count(self, name: str) -> int:
        return len(self.get_invocations(name))

    @property
    def hook_names(self) -> list[str]:
        """Hook names in invocation order."""
        return [inv["hook_name"] for inv in self.invocations]

    def reset(self) -> None:
        self.invocations.clear()

    # ── Assertions ─────────────────────────────────────────────────────

    def assert_method_called(self, name: str, times: int | None = None) -> None:
        count = self.call_count(name)
        if times is None:
            assert count > 0, f"Expected {name} to be called at least once, got 0"
        else:
            assert count == times, f"Expected {name} to be called {times} time(s), got {count}"

    def assert_context_field(self, method: str, field: str, value: Any) -> None:
        """Assert every invocation of ``method`` carries ``context.<field> == value``."""
        invocations = self.get_invocations(method)
        assert invocations, f"No invocations of '{method}' to assert context on"
        for inv in invocations:
            actual = getattr(inv["context"], field, None)
            assert actual == value, f"Expected {method}.{field} == {value!r}, got {actual!r}"

    # ── FlowHooks protocol ─────────────────────────────────────────────

    async def on_graph_start(self, context: Any) -> None:
        await self._record("on_graph_start", context)

    async def on_graph_end(self, context: Any) -> None:
        await self._record("on_graph_end", context)

    async def on_graph_error(self, context: Any, error: Exception) -> None:
        await self._record("on_graph_error", context, error=str(error))

    async def on_node_start(self, context: Any) -> None:
        await self._record("on_node_start", context)

    async def on_node_end(self, context: Any) -> None:
        await self._record("on_node_end", context)

    async def on_node_error(self, context: Any, error: Exception) -> None:
        await self._record("on_node_error", context, error=str(error))

    async def on_node_bypass(self, context: Any, reason: str) -> None:
        await self._record("on_node_bypass", context, reason=reason)

    async def on_llm_start(self, context: Any, llm_config: dict[str, Any] | None = None) -> None:
        await self._record("on_llm_start", context, llm_config=llm_config)

    async def on_llm_end(self, context: Any, response: dict[str, Any] | None = None) -> None:
        await self._record("on_llm_end", context, response=response)

    async def on_llm_loop_end(self, context: Any) -> None:
        await self._record("on_llm_loop_end", context)

    async def on_tool_start(self, context: Any) -> None:
        await self._record("on_tool_start", context)

    async def on_tool_end(self, context: Any) -> None:
        await self._record("on_tool_end", context)


class AsyncRecordingHook(RecordingHook):
    """``RecordingHook`` that sleeps before recording, for async-bridge tests."""

    def __init__(self, delay: float = 0.01, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delay = delay

    async def _record(self, hook_name: str, context: Any, **kwargs: Any) -> None:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        await super()._record(hook_name, context, **kwargs)
