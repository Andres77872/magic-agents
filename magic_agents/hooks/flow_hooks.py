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
        
        llm_config contains model, provider, streaming, tools, etc.
        """
        ...
    
    async def on_llm_end(self, context: HookContext, response: Optional[Dict[str, Any]] = None) -> None:
        """Called after LLM call completes.
        
        response contains the LLM output summary.
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
    """Extended context for NodeLLM-specific hooks.
    
    Provides LLM-specific fields: model, provider, streaming, tokens, tools.
    """
    
    model: Optional[str] = None
    provider: Optional[str] = None
    streaming: bool = False
    token_count: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    chunk_index: Optional[int] = None  # For streaming chunk hooks


@dataclass
class NodeMcpHookContext(HookContext):
    """Extended context for NodeMcp-specific hooks.
    
    Provides MCP-specific fields: session, tools discovered, bundle.
    """
    
    session_id: Optional[str] = None
    tools_discovered: Optional[int] = None
    bundle: Optional[Dict[str, Any]] = None


@dataclass
class NodeLoopHookContext(HookContext):
    """Extended context for NodeLoop-specific hooks.
    
    Provides loop-specific fields: items count, current index, aggregation.
    """
    
    items_count: int = 0
    item_index: Optional[int] = None
    aggregated_result: Optional[Any] = None