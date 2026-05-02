"""
CompositeObserver — combines parent (graph-level) and child (node-level) observers.

Each hook calls await parent.hook(...) then await child.hook(...).
Used by ObserverRegistry.observer_for() when a node provides a custom observer
without suppress_parent=True — both the graph-level and node-level observers fire.

Exceptions from each component are handled independently so one failure
does not block the other.
"""

import logging
from typing import Any, Dict, Optional

from magic_agents.debug.observer import DebugObserver

logger = logging.getLogger(__name__)


class CompositeObserver:
    """Combines parent (graph-level) + child (node-level) observers.

    Every hook invocation delegates to both observers in order:
    parent first, then child. Exceptions from one do not block the other.
    """

    def __init__(
        self,
        parent: DebugObserver,
        child: DebugObserver,
    ) -> None:
        """Initialize the composite observer.

        Args:
            parent: The graph-level observer (fires first).
            child: The node-level observer (fires second).
        """
        self._parent = parent
        self._child = child

    async def _call_both(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Call the same hook on both parent and child, isolating exceptions."""
        for idx, observer in enumerate([self._parent, self._child]):
            try:
                hook = getattr(observer, hook_name)
                await hook(*args, **kwargs)
            except Exception:
                comp = "parent" if idx == 0 else "child"
                logger.exception(
                    "CompositeObserver: %s '%s' failed", comp, hook_name
                )

    async def on_graph_start(
        self,
        graph_type: str,
        execution_id: str,
        node_count: int,
        edge_count: int,
    ) -> None:
        await self._call_both("on_graph_start", graph_type, execution_id, node_count, edge_count)

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
        await self._call_both(
            "on_graph_end", graph_type, execution_id, total_duration_ms,
            node_count, executed_count, bypassed_count, failed_count,
        )

    async def on_node_start(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        inputs: Dict[str, Any],
    ) -> None:
        await self._call_both("on_node_start", node_id, node_type, node_class, inputs)

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
        await self._call_both(
            "on_node_end", node_id, node_type, node_class,
            outputs, internal_state, duration_ms, start_time,
        )

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
        await self._call_both(
            "on_node_error", node_id, node_type, node_class,
            error, error_type, inputs, outputs, duration_ms, start_time,
        )

    async def on_node_bypass(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        reason: str,
    ) -> None:
        await self._call_both("on_node_bypass", node_id, node_type, node_class, reason)

    async def on_custom(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        await self._call_both("on_custom", event_type, data)
