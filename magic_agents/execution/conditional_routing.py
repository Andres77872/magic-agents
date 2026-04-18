"""
ConditionalRouting Protocol — Agnostic conditional routing interface.

This protocol defines the minimum contract that any conditional-like node
must implement to work with the reactive executor's bypass propagation logic.

By using a protocol instead of isinstance(NodeConditional), the executor
becomes agnostic to the specific node type and can work with any node
that implements this routing contract.

Detection Strategy:
    The executor uses `hasattr(node, 'selected_handle')` to detect conditional-like
    nodes AFTER execution (when selected_handle is set by process()).
    
    For pre-execution detection (e.g., skipping in static phase), the executor
    checks `hasattr(node, 'condition_template')` — the attribute that identifies
    a node as conditional-like before execution.
    
    New conditional-like node types should implement both attributes to be
    properly detected by the executor.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ConditionalRouting(Protocol):
    """
    Protocol for nodes that perform conditional routing.
    
    Any node implementing this protocol can be treated as a conditional-like
    node by the reactive executor, without requiring isinstance checks.
    
    Required attributes (set by the node):
        selected_handle: The handle selected during the last execution.
            Set by the node's process() method after evaluating the condition.
            Used by the executor for bypass propagation.
    
    Optional attributes:
        default_handle: Fallback handle if selected_handle has no matching edge.
        output_handles: List of declared output handle names for validation.
    
    Pre-execution detection:
        Nodes should also have a `condition_template` attribute (or similar)
        that identifies them as conditional-like BEFORE execution. This is
        used by the executor to skip conditionals without inputs in the
        static phase.
    
    Usage:
        # Post-execution check (selected_handle is set)
        if isinstance(node, ConditionalRouting):
            selected = node.selected_handle
            # ... bypass propagation logic
        
        # Pre-execution check (before selected_handle is set)
        if hasattr(node, 'condition_template'):
            # This is a conditional-like node
            if not has_inputs:
                continue  # Skip in static phase
    """
    
    # Required: Set by the node after condition evaluation
    selected_handle: str
    
    # Optional: Fallback handle for routing errors
    default_handle: Optional[str]
    
    # Optional: Declared output handles for build-time validation
    output_handles: Optional[List[str]]
