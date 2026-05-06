"""
HookContextFactory — validated HookContext construction for all event types.

Provides static factory methods that construct HookContext instances with
required fields populated, extra kwargs absorbed, and optional warning mode
for missing required fields.

Design decisions:
- Factory methods accept `**extra` for forward compatibility — unknown kwargs
  are absorbed into context.inputs (or context.outputs) without error.
- Optional `warn_on_missing: bool = False` parameter logs warnings for missing
  required fields but never blocks execution (Phase 3c — diagnostic mode).
- Direct HookContext() construction emits a DeprecationWarning (M4). Use the
  factory instead.
- All methods return a fully constructed HookContext — no partial contexts.

Usage:
    from magic_agents.hooks.context_factory import HookContextFactory

    ctx = HookContextFactory.build_graph_context(
        execution_id="abc123",
        run_id="run-1",
        metadata={"graph_id": "g1", "node_count": 3, "edge_count": 2},
    )
    await hooks.invoke("on_graph_start", ctx)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional
from datetime import datetime, UTC

from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.contracts import BypassReason


def _make_context(**kwargs: Any) -> HookContext:
    """Construct HookContext with deprecation warning suppressed.

    The factory is the canonical way to create HookContext instances.
    Suppress the deprecation warning since we ARE the recommended path.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return HookContext(**kwargs)

logger = logging.getLogger(__name__)


class HookContextFactory:
    """Factory for validated HookContext construction.

    Each method constructs a HookContext with required fields enforced.
    Extra kwargs are absorbed into inputs/outputs for forward compatibility.
    Methods never raise on unexpected kwargs — they are silently absorbed.
    """

    # ── Graph Lifecycle ─────────────────────────────────────────────────

    @staticmethod
    def build_graph_context(
        execution_id: str,
        run_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        warn_on_missing: bool = False,
        **extra: Any,
    ) -> HookContext:
        """Build a graph-level HookContext (graph start/end/error).

        Args:
            execution_id: Unique graph execution ID.
            run_id: Optional run ID for traceability.
            metadata: Dict containing graph_id, node_count, edge_count.
            warn_on_missing: If True, log warning when required fields missing.
            **extra: Extra kwargs absorbed into context.inputs.

        Returns:
            HookContext with graph-level identity and metadata in inputs.
        """
        metadata = metadata or {}
        inputs: Dict[str, Any] = {}

        # Populate inputs from metadata
        graph_id = metadata.get("graph_id")
        node_count = metadata.get("node_count")
        edge_count = metadata.get("edge_count")

        if graph_id is not None:
            inputs["graph_id"] = graph_id
        elif warn_on_missing:
            logger.warning("build_graph_context: missing 'graph_id' in metadata")

        if node_count is not None:
            inputs["node_count"] = node_count
        elif warn_on_missing:
            logger.warning("build_graph_context: missing 'node_count' in metadata")

        if edge_count is not None:
            inputs["edge_count"] = edge_count
        elif warn_on_missing:
            logger.warning("build_graph_context: missing 'edge_count' in metadata")

        # Absorb extra kwargs into inputs
        inputs.update(extra)

        return _make_context(
            execution_id=execution_id,
            run_id=run_id,
            node_id=None,
            node_type=None,
            node_class=None,
            inputs=inputs,
            metadata=metadata,
        )

    # ── Node Lifecycle ──────────────────────────────────────────────────

    @staticmethod
    def build_node_context(
        execution_id: str,
        run_id: str = "",
        node_id: Optional[str] = None,
        node_type: Optional[str] = None,
        node_class: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        parent_run_id: Optional[str] = None,
        warn_on_missing: bool = False,
        **extra: Any,
    ) -> HookContext:
        """Build a node-level HookContext (node start/end/error).

        Args:
            execution_id: Unique graph execution ID.
            run_id: Optional run ID for traceability.
            node_id: The node's ID.
            node_type: The node's type string.
            node_class: The node's class name.
            inputs: Node input data.
            outputs: Node output data.
            start_time: Optional execution start time.
            parent_run_id: Optional parent run ID.
            warn_on_missing: If True, log warning when node_id/type/class missing.
            **extra: Extra kwargs absorbed into context.inputs.

        Returns:
            HookContext with node identity and data.
        """
        if warn_on_missing:
            if not node_id:
                logger.warning("build_node_context: missing 'node_id'")
            if not node_type:
                logger.warning("build_node_context: missing 'node_type'")
            if not node_class:
                logger.warning("build_node_context: missing 'node_class'")

        merged_inputs: Dict[str, Any] = dict(inputs or {})
        merged_inputs.update(extra)

        return _make_context(
            execution_id=execution_id,
            run_id=run_id,
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            inputs=merged_inputs,
            outputs=outputs or {},
            start_time=start_time,
            parent_run_id=parent_run_id,
        )

    # ── Edge Hook Context ───────────────────────────────────────────────

    @staticmethod
    def build_edge_context(
        execution_id: str,
        run_id: str = "",
        node_id: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        content: Any = None,
        source_handle: Optional[str] = None,
        target_handle: Optional[str] = None,
        sequence_number: int = 0,
        node_type: Optional[str] = None,
        node_class: Optional[str] = None,
        warn_on_missing: bool = False,
        **extra: Any,
    ) -> HookContext:
        """Build an edge-level HookContext for NodeHook dispatch.

        Carries unique handle-level routing data (source, target, handles)
        in context.inputs for edge hook consumers.

        Args:
            execution_id: Unique graph execution ID.
            run_id: Optional run ID for traceability.
            node_id: Source node ID (node that produced the output).
            source: Source node ID (routing data).
            target: Target node ID (routing data).
            content: The payload content traversing the edge.
            source_handle: Source handle name.
            target_handle: Target handle name.
            sequence_number: Edge traversal sequence number.
            node_type: Source node type.
            node_class: Source node class name.
            warn_on_missing: If True, log warnings.
            **extra: Extra kwargs absorbed into context.inputs.

        Returns:
            HookContext with edge routing data in inputs.
        """
        inputs: Dict[str, Any] = {
            "content": content,
            "source": source or "",
            "target": target or "",
        }
        if source_handle is not None:
            inputs["source_handle"] = source_handle
        if target_handle is not None:
            inputs["target_handle"] = target_handle

        inputs.update(extra)

        if warn_on_missing:
            if not source:
                logger.warning("build_edge_context: missing 'source'")
            if not target:
                logger.warning("build_edge_context: missing 'target'")

        return _make_context(
            execution_id=execution_id,
            run_id=run_id,
            sequence_number=sequence_number,
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            inputs=inputs,
        )

    # ── Bypass Context ──────────────────────────────────────────────────

    @staticmethod
    def build_bypass_context(
        execution_id: str,
        run_id: str = "",
        node_id: Optional[str] = None,
        node_type: Optional[str] = None,
        node_class: Optional[str] = None,
        reason: BypassReason = "not_ready",
        metadata: Optional[Dict[str, Any]] = None,
        warn_on_missing: bool = False,
        **extra: Any,
    ) -> HookContext:
        """Build a bypass HookContext (on_node_bypass).

        Args:
            execution_id: Unique graph execution ID.
            run_id: Optional run ID for traceability.
            node_id: The bypassed node's ID.
            node_type: The bypassed node's type.
            node_class: The bypassed node's class name.
            reason: Canonical bypass reason (upstream_error, condition, not_ready).
            metadata: Bypass metadata (phase, upstream_error_node, etc.).
            warn_on_missing: If True, log warnings.
            **extra: Extra kwargs absorbed into context.inputs.

        Returns:
            HookContext with bypass data.
        """
        merged_metadata = dict(metadata or {})
        merged_metadata.update(extra)

        inputs: Dict[str, Any] = {"reason": reason}

        return _make_context(
            execution_id=execution_id,
            run_id=run_id,
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            inputs=inputs,
            metadata=merged_metadata,
        )

    # ── LLM Lifecycle ───────────────────────────────────────────────────

    @staticmethod
    def build_llm_context(
        execution_id: str,
        run_id: str = "",
        node_id: Optional[str] = None,
        node_type: Optional[str] = None,
        node_class: Optional[str] = None,
        model: str = "",
        streaming: bool = False,
        iteration: Optional[int] = None,
        sequence_number: int = 0,
        warn_on_missing: bool = False,
        **extra: Any,
    ) -> HookContext:
        """Build an LLM lifecycle HookContext (on_llm_start, on_llm_end).

        Produces a consistent structure for BOTH tool and non-tool LLM paths
        (S4 requirement). Both paths produce the same required fields:
        model, streaming, execution_id, node_id, node_type, node_class.

        Args:
            execution_id: Unique graph execution ID.
            run_id: Optional run ID for traceability.
            node_id: The NodeLLM node ID.
            node_type: The node type (typically "LLM").
            node_class: The node class name (typically "NodeLLM").
            model: The LLM model name.
            streaming: Whether streaming is enabled.
            iteration: Optional iteration number.
            sequence_number: Sequence number for ordering.
            warn_on_missing: If True, log warnings.
            **extra: Extra kwargs absorbed into context.inputs.

        Returns:
            HookContext with LLM lifecycle data.
        """
        inputs: Dict[str, Any] = {
            "model": model,
            "streaming": streaming,
        }
        if iteration is not None:
            inputs["iteration"] = iteration

        # Absorb extra kwargs into inputs (e.g., llm_config, content_preview)
        inputs.update(extra)

        if warn_on_missing:
            if not model:
                logger.warning("build_llm_context: missing 'model'")

        return _make_context(
            execution_id=execution_id,
            run_id=run_id,
            sequence_number=sequence_number,
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            inputs=inputs,
        )

    # ── Tool Lifecycle ──────────────────────────────────────────────────

    @staticmethod
    def build_tool_context(
        execution_id: str,
        run_id: str = "",
        node_id: Optional[str] = None,
        node_type: Optional[str] = None,
        node_class: Optional[str] = None,
        tool_name: str = "",
        tool_call_id: str = "",
        arguments: Optional[Dict[str, Any]] = None,
        result: Any = None,
        success: bool = True,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        warn_on_missing: bool = False,
        **extra: Any,
    ) -> HookContext:
        """Build a tool lifecycle HookContext (on_tool_start, on_tool_end).

        Args:
            execution_id: Unique graph execution ID.
            run_id: Optional run ID for traceability.
            node_id: The source node ID.
            node_type: The source node type.
            node_class: The source node class name.
            tool_name: The tool name.
            tool_call_id: The tool call identifier.
            arguments: The tool arguments dict.
            result: The tool execution result object.
            success: Whether the tool succeeded.
            error_message: Optional error message (on failure).
            execution_time_ms: Tool execution duration in ms (S3).
            warn_on_missing: If True, log warnings.
            **extra: Extra kwargs absorbed into context.inputs.

        Returns:
            HookContext with tool lifecycle data.
        """
        inputs: Dict[str, Any] = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "arguments": arguments or {},
        }
        inputs.update(extra)

        outputs: Dict[str, Any] = {
            "tool_name": tool_name,
            "success": success,
            "execution_time_ms": execution_time_ms,
        }

        if warn_on_missing:
            if not tool_name:
                logger.warning("build_tool_context: missing 'tool_name'")
            if not tool_call_id:
                logger.warning("build_tool_context: missing 'tool_call_id'")

        return _make_context(
            execution_id=execution_id,
            run_id=run_id,
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            inputs=inputs,
            outputs=outputs,
            error_message=error_message,
        )
