"""
DebugObserver Protocol — public contract for graph/node lifecycle observation.

All hooks are async. Implementations must catch and handle exceptions internally;
callers SHOULD catch and log observer exceptions to prevent cascading failures.

The protocol uses flat keyword arguments (no bundled context dict) to make the
contract explicit and type-safe. @runtime_checkable enables isinstance() checks.
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class DebugObserver(Protocol):
    """Async observer protocol for graph/node execution lifecycle events.

    Implementations are not required to inherit from this class — structural
    typing (Protocol) means any object with matching method signatures satisfies
    the contract. Use isinstance(observer, DebugObserver) to verify at runtime.

    All hooks are async. Each hook receives explicit keyword parameters only —
    there is no bundled 'context' dict, keeping the contract explicit.
    """

    # ── Graph lifecycle (owned by executor) ──────────────────────────────

    async def on_graph_start(
        self,
        graph_type: str,
        execution_id: str,
        node_count: int,
        edge_count: int,
    ) -> None:
        """Called when graph execution begins.

        Args:
            graph_type: Type identifier for the graph (e.g. 'chat', 'agent').
            execution_id: Unique trace identifier for this execution.
            node_count: Total number of nodes in the graph.
            edge_count: Total number of edges in the graph.
        """
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
        """Called when graph execution completes (all nodes finished).

        Args:
            graph_type: Type identifier for the graph.
            execution_id: Unique trace identifier for this execution.
            total_duration_ms: Wall-clock duration of the entire execution.
            node_count: Total number of nodes in the graph.
            executed_count: Number of nodes that executed successfully.
            bypassed_count: Number of nodes that were bypassed.
            failed_count: Number of nodes that failed.
        """
        ...

    # ── Node lifecycle (owned by Node.__call__) ──────────────────────────

    async def on_node_start(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        inputs: Dict[str, Any],
    ) -> None:
        """Called before a node's process() method executes.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type identifier (e.g. 'LLM', 'TEXT', 'CONDITIONAL').
            node_class: Python class name of the node implementation.
            inputs: The input dict that will be passed to process().
        """
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
        """Called after a node's process() completes successfully.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type identifier.
            node_class: Python class name.
            outputs: The output dict produced by process().
            internal_state: Snapshot of node-internal variables post-execution.
            duration_ms: Wall-clock duration of process() in milliseconds.
            start_time: ISO-8601 timestamp when process() began.
        """
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
        """Called when a node's process() raises an exception.

        The error is re-raised after the observer call — the observer does
        NOT suppress the exception.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type identifier.
            node_class: Python class name.
            error: The exception message (str(e)).
            error_type: The exception class name (type(e).__name__).
            inputs: The inputs that were passed to process().
            outputs: Partial outputs if any, or None if process() produced none.
            duration_ms: Wall-clock duration from start to exception.
            start_time: ISO-8601 timestamp when process() began.
        """
        ...

    # ── Bypass (owned by executor) ───────────────────────────────────────

    async def on_node_bypass(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        reason: str,
    ) -> None:
        """Called when the executor determines a node should be skipped.

        Args:
            node_id: Unique identifier for the bypassed node.
            node_type: Type identifier.
            node_class: Python class name.
            reason: Human-readable explanation of why the node was bypassed.
        """
        ...

    # ── Custom / extension (for per-node specialization) ──────────────────

    async def on_custom(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Called for custom events not covered by the standard lifecycle.

        Implementations MUST NOT raise on unrecognized event_type —
        they SHOULD log a warning and continue.

        Args:
            event_type: Arbitrary string identifying the custom event.
            data: Arbitrary payload dict for the custom event.
        """
        ...
