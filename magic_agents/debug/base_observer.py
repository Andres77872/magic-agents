"""
BaseObserver — optional exception-safe base class for DebugObserver implementations.

Provides a _safe_call pattern that wraps each hook in try/except + logger.exception().
Subclasses override the _on_{hook_name} abstract methods.

This is OPTIONAL. Implementations like NullObserver skip it entirely by
implementing DebugObserver directly (no ABC overhead). Production observers
like DefaultObserver use it for shared error-handling.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseObserver(ABC):
    """Optional exception-safe base for DebugObserver implementations.

    Each public hook method delegates to _safe_call, which wraps the
    corresponding _on_{hook_name} abstract method in try/except + logging.
    Subclasses override the _on_* methods.
    """

    async def _safe_call(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Wrap a hook call in try/except with logging.

        Args:
            hook_name: The hook method name (without 'on_' or '_on_' prefix).
            *args: Positional arguments to pass to the hook.
            **kwargs: Keyword arguments to pass to the hook.
        """
        try:
            hook = getattr(self, f"_on_{hook_name}")
            await hook(*args, **kwargs)
        except Exception:
            logger.exception("Observer hook '%s' failed", hook_name)

    # ── Graph lifecycle ─────────────────────────────────────────────────

    async def on_graph_start(
        self,
        graph_type: str,
        execution_id: str,
        node_count: int,
        edge_count: int,
    ) -> None:
        await self._safe_call("graph_start", graph_type, execution_id, node_count, edge_count)

    @abstractmethod
    async def _on_graph_start(
        self,
        graph_type: str,
        execution_id: str,
        node_count: int,
        edge_count: int,
    ) -> None:
        ...

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
        await self._safe_call(
            "graph_end", graph_type, execution_id, total_duration_ms,
            node_count, executed_count, bypassed_count, failed_count,
        )

    @abstractmethod
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
        ...

    # ── Node lifecycle ──────────────────────────────────────────────────

    async def on_node_start(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        inputs: Dict[str, Any],
    ) -> None:
        await self._safe_call("node_start", node_id, node_type, node_class, inputs)

    @abstractmethod
    async def _on_node_start(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        inputs: Dict[str, Any],
    ) -> None:
        ...

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
        await self._safe_call(
            "node_end", node_id, node_type, node_class,
            outputs, internal_state, duration_ms, start_time,
        )

    @abstractmethod
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
        ...

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
        await self._safe_call(
            "node_error", node_id, node_type, node_class,
            error, error_type, inputs, outputs, duration_ms, start_time,
        )

    @abstractmethod
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
        ...

    # ── Bypass ──────────────────────────────────────────────────────────

    async def on_node_bypass(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        reason: str,
    ) -> None:
        await self._safe_call("node_bypass", node_id, node_type, node_class, reason)

    @abstractmethod
    async def _on_node_bypass(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        reason: str,
    ) -> None:
        ...

    # ── Custom ──────────────────────────────────────────────────────────

    async def on_custom(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        await self._safe_call("custom", event_type, data)

    @abstractmethod
    async def _on_custom(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        ...
