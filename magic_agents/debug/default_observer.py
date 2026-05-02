"""
DefaultObserver — production DebugObserver implementation.

Wraps the existing magic_agents/debug/ internals (DebugContext, EmitterRegistry,
DebugCollector) to provide the same behavior as the legacy GraphDebugFeedback
path, but through the observer interface.

Uses BaseObserver for shared exception-safe error handling.
"""

import logging
from datetime import datetime, UTC
from typing import Any, Dict, Optional

from magic_agents.debug.base_observer import BaseObserver
from magic_agents.debug.config import DebugConfig, default_config
from magic_agents.debug.context import DebugContext
from magic_agents.debug.emitter import EmitterRegistry
from magic_agents.debug.events import DebugEvent, DebugEventType

logger = logging.getLogger(__name__)


class DefaultObserver(BaseObserver):
    """Production DebugObserver implementation.

    Delegates to DebugContext for capture → transform → emit → collect.
    Uses BaseObserver's _safe_call pattern for exception isolation.

    On on_graph_end, uses DebugCollector to produce a summary
    and emits it through the registered emitters. The summary is serialized
    via to_legacy_format() for backward compatibility.
    """

    def __init__(
        self,
        execution_id: str,
        graph_type: str,
        config: Optional[DebugConfig] = None,
        emitter_registry: Optional[EmitterRegistry] = None,
        total_nodes: int = 0,
    ) -> None:
        """Initialize the DefaultObserver.

        Args:
            execution_id: Unique trace identifier for this execution.
            graph_type: Type identifier for the graph.
            config: DebugConfig for filtering/sampling/redaction.
                   Uses default_config() if None.
            emitter_registry: EmitterRegistry for output emission.
                   Creates a new one if None.
            total_nodes: Total number of nodes in the graph.
        """
        self._execution_id = execution_id
        self._graph_type = graph_type
        self._config = config or default_config()
        self._emitter_registry = emitter_registry or EmitterRegistry()
        self._total_nodes = total_nodes

        # Create DebugContext — coordinates capture, transform, emit, collect
        self._ctx = DebugContext(
            execution_id=execution_id,
            graph_type=graph_type,
            enabled=self._config.enabled,
            total_nodes=total_nodes,
            config=self._config,
        )
        # Wire our emitter registry into the context
        self._ctx._emitters = self._emitter_registry  # type: ignore[attr-defined]

    @property
    def emitter_registry(self) -> EmitterRegistry:
        """Get the emitter registry for adding callbacks."""
        return self._emitter_registry

    # ── Graph lifecycle ─────────────────────────────────────────────────

    async def _on_graph_start(
        self,
        graph_type: str,
        execution_id: str,
        node_count: int,
        edge_count: int,
    ) -> None:
        """Emit graph start event via DebugContext capture."""
        if not self._ctx.enabled:
            return
        event = self._ctx.capture.on_graph_start(
            graph_type=graph_type,
            total_nodes=node_count,
        )
        await self._ctx.emit(event)

    async def _on_graph_end(
        self,
        graph_type: str,
        execution_id: str,
        total_duration_ms: float,
        node_count: int,
        executed_count: int,
        bypassed_count: int,
        failed_count: int,
    ) -> None:
        """Emit graph end summary event via DebugCollector.

        Collects execution summary from DebugCollector, serializes to legacy
        format (backward compatible), and emits exactly ONE GRAPH_END event
        through the emitter registry.

        The basic capture.on_graph_end() event is deliberately SKIPPED — it
        carries no node-level data and would produce a duplicate external
        ``debug_summary`` event when normalized by the API layer. Only the
        summary event with full ``to_legacy_format()`` data is emitted.
        """
        if not self._ctx.enabled:
            return

        # Collect and emit summary (single source of truth for graph_end)
        summary = self._ctx.collector.get_summary()
        summary.total_duration_ms = total_duration_ms
        summary.executed_nodes = executed_count
        summary.bypassed_nodes = bypassed_count
        summary.failed_nodes = failed_count

        # Serialize to legacy format for backward compatibility
        legacy_summary = summary.to_legacy_format()

        # Emit as a single summary event through the emitter registry
        summary_event = DebugEvent(
            event_type=DebugEventType.GRAPH_END,
            execution_id=execution_id,
            payload=legacy_summary,
            timestamp=datetime.now(UTC),
        )
        await self._ctx.emit(summary_event)

    # ── Node lifecycle ──────────────────────────────────────────────────

    async def _on_node_start(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        inputs: Dict[str, Any],
    ) -> None:
        """Capture and emit node start event."""
        if not self._ctx.enabled:
            return
        event = self._ctx.capture.on_node_start(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            inputs=inputs,
        )
        await self._ctx.emit(event)

    async def _on_node_end(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        outputs: Dict[str, Any],
        internal_state: Dict[str, Any],
        duration_ms: float,
        start_time: str,
    ) -> None:
        """Capture and emit node end event."""
        if not self._ctx.enabled:
            return
        event = self._ctx.capture.on_node_end(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            outputs=outputs,
            internal_state=internal_state,
            duration_ms=duration_ms,
            start_time=datetime.fromisoformat(start_time) if start_time else datetime.now(UTC),
        )
        await self._ctx.emit(event)

    async def _on_node_error(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        error: str,
        error_type: str,
        inputs: Dict[str, Any],
        outputs: Optional[Dict[str, Any]],
        duration_ms: float,
        start_time: str,
    ) -> None:
        """Capture and emit node error event."""
        if not self._ctx.enabled:
            return
        event = self._ctx.capture.on_node_error(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            error=Exception(f"{error_type}: {error}"),
            context={
                "error": error,
                "error_type": error_type,
                "inputs": inputs,
                "outputs": outputs or {},
                "duration_ms": duration_ms,
                "start_time": start_time,
            },
        )
        await self._ctx.emit(event)

    async def _on_node_bypass(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        reason: str,
    ) -> None:
        """Capture and emit node bypass event."""
        if not self._ctx.enabled:
            return
        event = self._ctx.capture.on_node_bypass(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            reason=reason,
            bypass_source=reason,
            inputs_at_bypass={},
        )
        await self._ctx.emit(event)

    async def _on_custom(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a custom event."""
        if not self._ctx.enabled:
            return
        # Create a raw DebugEvent for custom events
        event = DebugEvent(
            execution_id=self._execution_id,
            payload=data,
            timestamp=datetime.now(UTC),
        )
        await self._ctx.emit(event)
