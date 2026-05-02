"""
NullObserver — no-op implementation of DebugObserver.

All hooks are async no-ops. Used when debug is disabled via the global
kill switch (DEBUG_ENABLED=False) or per-graph debug=False.

Implements DebugObserver directly (no ABC) to avoid any overhead —
no logging, no state, no side effects.
"""

from typing import Any, Dict, Optional

from magic_agents.debug.observer import DebugObserver


class NullObserver:
    """No-op DebugObserver implementation.

    Every method is an async no-op that returns None immediately.
    Satisfies isinstance(observer, DebugObserver) at runtime.
    """

    async def on_graph_start(
        self,
        graph_type: str,
        execution_id: str,
        node_count: int,
        edge_count: int,
    ) -> None:
        pass

    async def on_graph_end(
        self,
        graph_type: str,
        execution_id: str,
        total_duration_ms: float,
        node_count: int,
        executed_count: int,
        bypassed_count: int,
        failed_count: int,
    ) -> None:
        pass

    async def on_node_start(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        inputs: Dict[str, Any],
    ) -> None:
        pass

    async def on_node_end(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        outputs: Dict[str, Any],
        internal_state: Dict[str, Any],
        duration_ms: float,
        start_time: str,
    ) -> None:
        pass

    async def on_node_error(
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
        pass

    async def on_node_bypass(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        reason: str,
    ) -> None:
        pass

    async def on_custom(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        pass
