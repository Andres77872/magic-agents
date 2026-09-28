"""Reusable graph/debug SSE FlowHooks implementation.

Concrete API SSE transport is injected as a sink/recorder. This module has no
api.magic_llm imports and can be reused by any graph runtime consumer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Protocol

from magic_agents.hooks.flow_hooks import HookContext

logger = logging.getLogger(__name__)


class DebugEventSink(Protocol):
    def record(self, event: dict[str, Any]) -> Any: ...


class _RunClock:
    """Monotonic origin shared by a root hook and every hook forked from it."""

    __slots__ = ("origin",)

    def __init__(self) -> None:
        self.origin: float | None = None


class DebugSSEHook:
    """FlowHooks implementation that emits debug SSE-compatible envelopes.

    Every frame's content carries ``t_ms``: milliseconds since the root graph
    started, read from a monotonic clock when the event happened. Child graphs
    share the root's origin, so all frames of a run sit on one timeline.
    """

    def __init__(
        self,
        *,
        sink: DebugEventSink | asyncio.Queue,
        id_chat: str,
        parent_path: tuple[str, ...] = (),
    ) -> None:
        self._sink = sink
        self._id_chat = id_chat
        self._parent_path = tuple(parent_path)
        self._clock = _RunClock()

    def fork_for_child(
        self,
        *,
        child_run_id: str,
        parent_node_id: str,
    ) -> "DebugSSEHook":
        'Share the sink and clock while isolating ancestry for concurrent child graphs.'
        child = DebugSSEHook(
            sink=self._sink,
            id_chat=self._id_chat,
            parent_path=(*self._parent_path, parent_node_id),
        )
        child._clock = self._clock
        return child

    def _t_ms(self, context: HookContext | None, *, restart: bool = False) -> float:
        at = getattr(context, "monotonic_time", None)
        if at is None:
            at = time.perf_counter()
        if restart or self._clock.origin is None:
            self._clock.origin = at
        return round((at - self._clock.origin) * 1000, 3)

    def _emit(
        self,
        event_type: str,
        content: dict[str, Any],
        *,
        context: HookContext | None = None,
        summary: bool = False,
    ) -> None:
        # A root graph start opens a new run; child graph starts keep its origin.
        content["t_ms"] = self._t_ms(
            context,
            restart=event_type == "graph_start" and not self._parent_path,
        )
        node_id = content.get("node_id")
        source_path = [*self._parent_path]
        if isinstance(node_id, str) and node_id:
            source_path.append(node_id)
        event = {
            "type": "debug_summary" if summary else "debug",
            "source_node_path": source_path,
            "event_type": event_type,
            "id_chat": self._id_chat,
            "content": content,
        }
        try:
            if hasattr(self._sink, "put_nowait"):
                self._sink.put_nowait(event)
            elif hasattr(self._sink, "record"):
                self._sink.record(event)
            elif callable(self._sink):
                self._sink(event)
            else:
                raise TypeError("DebugSSEHook sink must be a Queue, record() object, or callable")
        except asyncio.QueueFull:
            logger.warning("DebugSSEHook queue full; dropping %s event", event_type)

    async def on_graph_start(self, context: HookContext) -> None:
        inputs = context.inputs or {}
        self._emit(
            "graph_start",
            {
                "execution_id": context.execution_id or "",
                "run_id": context.run_id or "",
                "node_count": inputs.get("node_count", 0),
            },
            context=context,
        )

    @staticmethod
    def _execution_counts(context: HookContext) -> dict[str, Any]:
        summary = (context.metadata or {}).get("execution_summary")
        if not isinstance(summary, dict):
            return {}
        return {
            "node_count": summary.get("total"),
            "executed_count": summary.get("completed", 0),
            "bypassed_count": summary.get("bypassed", 0),
            "failed_count": summary.get("errors", 0),
        }

    async def on_graph_end(self, context: HookContext) -> None:
        self._emit(
            "graph_end",
            {
                "execution_id": context.execution_id or "",
                "duration_ms": context.duration_ms,
                **self._execution_counts(context),
            },
            context=context,
            summary=True,
        )

    async def on_graph_error(self, context: HookContext, error: Exception) -> None:
        self._emit(
            "graph_error",
            {
                "execution_id": context.execution_id or "",
                "error_type": type(error).__name__ if error else "UnknownError",
                "error_message": str(error) if error else "Unknown error",
                "duration_ms": context.duration_ms,
                **self._execution_counts(context),
            },
            context=context,
            summary=True,
        )

    async def on_node_start(self, context: HookContext) -> None:
        self._emit(
            "node_start",
            {
                "node_id": context.node_id,
                "node_type": context.node_type,
                "node_class": context.node_class,
                "inputs": context.inputs,
            },
            context=context,
        )

    async def on_node_end(self, context: HookContext) -> None:
        self._emit(
            "node_end",
            {
                "node_id": context.node_id,
                "node_type": context.node_type,
                "duration_ms": context.duration_ms,
                "outputs": context.outputs,
            },
            context=context,
        )

    async def on_node_error(self, context: HookContext, error: Exception) -> None:
        self._emit(
            "node_error",
            {
                "node_id": context.node_id,
                "node_type": context.node_type,
                "error_type": type(error).__name__ if error else "UnknownError",
                "error_message": str(error) if error else "Unknown error",
                "duration_ms": context.duration_ms,
            },
            context=context,
        )

    async def on_node_bypass(self, context: HookContext, reason: str) -> None:
        self._emit("node_bypass", {"node_id": context.node_id, "reason": reason}, context=context)

    # ── LLM Lifecycle ──────────────────────────────────────────────────

    async def on_llm_start(
        self,
        context: HookContext,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        self._emit(
            "llm_start",
            {
                "node_id": context.node_id,
                "model": (context.inputs or {}).get("model", ""),
                "provider": (context.inputs or {}).get("provider", ""),
                "llm_config": llm_config,
            },
            context=context,
        )

    async def on_llm_end(
        self,
        context: HookContext,
        response: dict[str, Any] | None = None,
    ) -> None:
        outputs = context.outputs or {}
        tokens = {k: v for k, v in outputs.items()
                  if k.endswith("_tokens") or k.startswith("cached_tokens_") or k == "raw_usage_json"}
        usage_reported = outputs.get("usage_reported")
        if usage_reported is None:
            usage_reported = any(
                isinstance(tokens.get(k), (int, float)) and tokens[k] > 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
        self._emit(
            "llm_end",
            {
                "node_id": context.node_id,
                "finish_reason": outputs.get("finish_reason"),
                "tokens": tokens,
                "usage_reported": bool(usage_reported),
            },
            context=context,
        )

    async def on_llm_loop_end(self, context: HookContext) -> None:
        outputs = context.outputs or {}
        content_preview = (outputs.get("content_preview", "") or "")[:100]
        self._emit(
            "llm_loop_end",
            {
                "node_id": context.node_id,
                "total_iterations": outputs.get("total_iterations"),
                "content_preview": content_preview,
            },
            context=context,
        )

    # ── Tool Lifecycle ─────────────────────────────────────────────────

    async def on_tool_start(self, context: HookContext) -> None:
        inputs = context.inputs or {}
        self._emit(
            "tool_start",
            {
                "node_id": context.node_id,
                "tool_name": inputs.get("tool_name"),
                "tool_call_id": inputs.get("tool_call_id"),
            },
            context=context,
        )

    async def on_tool_end(self, context: HookContext) -> None:
        outputs = context.outputs or {}
        inputs = context.inputs or {}
        content = {
            "node_id": context.node_id,
            "tool_name": outputs.get("tool_name") or inputs.get("tool_name"),
            # Pairs this end with its tool_start when one tool runs in parallel.
            "tool_call_id": outputs.get("tool_call_id") or inputs.get("tool_call_id"),
            "success": outputs.get("success"),
            "execution_time_ms": outputs.get("execution_time_ms"),
        }
        if content["success"] is False:
            content["error_type"] = context.error_type
            content["error_message"] = context.error_message
        self._emit("tool_end", content, context=context)
