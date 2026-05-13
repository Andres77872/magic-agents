"""Reusable graph/debug SSE FlowHooks implementation.

Concrete API SSE transport is injected as a sink/recorder. This module has no
api.magic_llm imports and can be reused by any graph runtime consumer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

from magic_agents.hooks.flow_hooks import HookContext

logger = logging.getLogger(__name__)


class DebugEventSink(Protocol):
    def record(self, event: dict[str, Any]) -> Any: ...


class DebugSSEHook:
    """FlowHooks implementation that emits debug SSE-compatible envelopes."""

    def __init__(self, *, sink: DebugEventSink | asyncio.Queue, id_chat: str) -> None:
        self._sink = sink
        self._id_chat = id_chat

    def _emit(self, event_type: str, content: dict[str, Any], *, summary: bool = False) -> None:
        event = {
            "type": "debug_summary" if summary else "debug",
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
        )

    async def on_graph_end(self, context: HookContext) -> None:
        self._emit(
            "graph_end",
            {"execution_id": context.execution_id or "", "duration_ms": context.duration_ms},
            summary=True,
        )

    async def on_graph_error(self, context: HookContext, error: Exception) -> None:
        self._emit(
            "graph_error",
            {
                "execution_id": context.execution_id or "",
                "error_type": type(error).__name__ if error else "UnknownError",
                "error_message": str(error) if error else "Unknown error",
            },
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
        )

    async def on_node_error(self, context: HookContext, error: Exception) -> None:
        self._emit(
            "node_error",
            {
                "node_id": context.node_id,
                "node_type": context.node_type,
                "error_type": type(error).__name__ if error else "UnknownError",
                "error_message": str(error) if error else "Unknown error",
            },
        )

    async def on_node_bypass(self, context: HookContext, reason: str) -> None:
        self._emit("node_bypass", {"node_id": context.node_id, "reason": reason})

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
        )

    async def on_llm_end(
        self,
        context: HookContext,
        response: dict[str, Any] | None = None,
    ) -> None:
        outputs = context.outputs or {}
        tokens = {k: v for k, v in outputs.items()
                  if k.endswith("_tokens") or k.startswith("cached_tokens_") or k == "raw_usage_json"}
        self._emit(
            "llm_end",
            {
                "node_id": context.node_id,
                "finish_reason": outputs.get("finish_reason"),
                "tokens": tokens,
            },
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
        )

    async def on_tool_end(self, context: HookContext) -> None:
        outputs = context.outputs or {}
        self._emit(
            "tool_end",
            {
                "node_id": context.node_id,
                "tool_name": outputs.get("tool_name") or (context.inputs or {}).get("tool_name"),
                "success": outputs.get("success"),
            },
        )
