"""
FlowHooks Protocol and HookContext for magic-agents hook system.

This module defines the observer-only lifecycle hooks for flow execution
and the context payload passed to each hook invocation.

Contracts:
- All hook methods are async (observer-first, async-capable)
- Hooks MUST NOT modify execution state or alter control flow
- Exceptions are isolated (logged, execution continues)
- Protocol uses on_{component}_{action} naming convention
"""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable, Optional, Any, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, UTC


@runtime_checkable
class FlowHooks(Protocol):
    """Observer-only lifecycle hooks for flow execution.
    
    All methods are optional (no-op default via Protocol pattern).
    Hooks MUST NOT modify execution state or alter control flow.
    Exceptions are isolated (logged, execution continues).
    
    Naming follows industry-standard on_{component}_{action} pattern.
    
    Channel Ordering Contract:
    For START events (graph start, node start):
      Phase 0 (debug event yield) → Phase 1 (observer) → Phase 4 (hook)
    For END events (graph end, node end):
      Phase 4 (hook) → Phase 1 (observer) → Phase 0 (debug event yield)
    For BYPASS events:
      Phase 1 (observer) → Phase 4 (hook)
    For ERROR events:
      Phase 4 (hook) → Phase 1 (observer) → Phase 0 (debug event yield)
    
    This ordering ensures debug events are yielded first on start (persistence
    before dispatch) and hooks fire first on end/error (observability before
    persistence).
    """
    
    # === GRAPH LIFECYCLE (Tier 1, 2) ===
    
    async def on_graph_start(self, context: HookContext) -> None:
        """Called when graph execution begins.
        
        Invoked AFTER validation, BEFORE task creation.
        """
        ...
    
    async def on_graph_end(self, context: HookContext) -> None:
        """Called when graph execution completes successfully.
        
        Invoked AFTER all node tasks finish.
        """
        ...
    
    async def on_graph_error(self, context: HookContext, error: Exception) -> None:
        """Called when graph execution fails.
        
        Invoked when a blocking error occurs.
        on_graph_end is NOT invoked for failed executions.
        """
        ...
    
    # === NODE LIFECYCLE (Tier 1, 2, 3) ===
    
    async def on_node_start(self, context: HookContext) -> None:
        """Called when a node begins execution.
        
        Invoked BEFORE _start_debug_tracking() and process().
        """
        ...
    
    async def on_node_end(self, context: HookContext) -> None:
        """Called when a node completes execution successfully.
        
        Invoked AFTER process() yields outputs and _end_debug_tracking().
        """
        ...
    
    async def on_node_error(self, context: HookContext, error: Exception) -> None:
        """Called when a node execution fails.
        
        Invoked in exception catch block BEFORE yield_debug_error().
        """
        ...
    
    async def on_node_bypass(self, context: HookContext, reason: str) -> None:
        """Called when a node is bypassed.
        
        Invoked when bypass signal propagates (conditional routing or error cascade).
        Reason indicates bypass cause: 'condition_false', 'upstream_error', etc.
        """
        ...
    
    # === LLM LIFECYCLE (NodeLLM-specific) ===
    
    async def on_llm_start(self, context: HookContext, llm_config: Optional[Dict[str, Any]] = None) -> None:
        """Called before LLM call in NodeLLM.
        
        llm_config is populated with real config data when available from
        the HookRelay path. Contains model, provider, streaming, tools,
        tool_choice, deduplicate, and other available config fields.
        May be None when config data is entirely unavailable.
        """
        ...
    
    async def on_llm_end(self, context: HookContext, response: Optional[Dict[str, Any]] = None) -> None:
        """Called after LLM call completes.
        
        response contains the LLM output summary.
        """
        ...
    
    async def on_llm_loop_end(self, context: HookContext) -> None:
        """Called ONCE after agent loop completion (aggregated data).
        
        Carries accumulated content from ALL iterations/generations.
        Fires for BOTH streaming and non-streaming paths.
        Does NOT fire on budget-exceeded exit (use on_node_error instead).
        
        Per-iteration data is available via on_llm_end events (N times).
        This event is the aggregate signal (1 time per loop).
        
        Args:
            context: HookContext with LLMLoopEndInputs in outputs.
                Key fields: model, content, iteration (0-indexed),
                total_iterations (1-indexed count), provider_request_id,
                prompt_tokens, completion_tokens, total_tokens.
        """
        ...
    
    # === TOOL LIFECYCLE (via relay from magic-llm) ===
    
    async def on_tool_start(self, context: HookContext) -> None:
        """Called before tool execution (relayed from magic-llm).
        
        context.inputs contains tool_name, tool_call_id, arguments.
        """
        ...
    
    async def on_tool_end(self, context: HookContext) -> None:
        """Called after tool execution (relayed from magic-llm).
        
        context.outputs contains tool result.
        context.error_message contains tool error if failed.
        """
        ...


@dataclass
class HookContext:
    """Full payload context for all hooks.
    
    Provides execution identity, node context, data, timing, error fields,
    and emit interface for hook output capabilities.
    
    Deep copies are provided for inputs/outputs to prevent mutation.
    """
    
    def __post_init__(self) -> None:
        """Emit deprecation warning when HookContext is constructed directly.
        
        Use HookContextFactory.build_*_context() methods for schema-validated
        contexts instead.
        """
        import traceback
        # Only warn if this is a direct HookContext construction,
        # not from a subclass
        if type(self) is HookContext:
            warnings.warn(
                "Direct HookContext() construction is deprecated. "
                "Use HookContextFactory.build_*_context() methods for "
                "schema-validated contexts.",
                DeprecationWarning,
                stacklevel=2,
            )
    
    # === Identity ===
    execution_id: str  # Unique execution trace ID (graph_id)
    sequence_number: int = 0  # Order within execution
    run_id: str = ''  # Run ID from chat_log
    
    # === Node context (empty for graph-level hooks) ===
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    node_class: Optional[str] = None
    
    # === Data ===
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    # === Timing ===
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # === Error (for error hooks) ===
    error: Optional[Exception] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # === Metadata ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # === Parent context (for nested executions) ===
    parent_run_id: Optional[str] = None
    
    # === Emit helpers (injected at runtime) ===
    # Forward reference to avoid circular import
    emit: Optional[Any] = None  # EmitInterface injected at runtime
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for logging/transmission.
        
        Error objects are not serialized directly - only type/message.
        Emit interface is excluded from serialization.
        
        Note: Only serializes base HookContext fields. Extension subclass fields
        are intentionally omitted (they are deprecated and not populated by
        HookContextFactory).
        """
        return {
            "execution_id": self.execution_id,
            "sequence_number": self.sequence_number,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "node_class": self.node_class,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "timestamp": self.timestamp.isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "parent_run_id": self.parent_run_id,
        }


# === Node-Specific Context Extensions ===


@dataclass
class NodeLLMHookContext(HookContext):
    """[DEPRECATED] Extended context for NodeLLM-specific hooks.
    
    Provides LLM-specific fields: model, provider, streaming, tokens, tools.
    This extension is deprecated. Use HookContext with metadata fields instead.
    """
    
    model: Optional[str] = None
    provider: Optional[str] = None
    streaming: bool = False
    token_count: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    chunk_index: Optional[int] = None  # For streaming chunk hooks


@dataclass
class NodeMcpHookContext(HookContext):
    """[DEPRECATED] Extended context for NodeMcp-specific hooks.
    
    Provides MCP-specific fields: session, tools discovered, bundle.
    This extension is deprecated. Use HookContext with metadata fields instead.
    """
    
    session_id: Optional[str] = None
    tools_discovered: Optional[int] = None
    bundle: Optional[Dict[str, Any]] = None


@dataclass
class NodeLoopHookContext(HookContext):
    """[DEPRECATED] Extended context for NodeLoop-specific hooks.
    
    Provides loop-specific fields: items count, current index, aggregation.
    This extension is deprecated. Use HookContext with metadata fields instead.
    """
    
    items_count: int = 0
    item_index: Optional[int] = None
    aggregated_result: Optional[Any] = None