"""
Event-based reactive execution module for magic-agents.

This module implements a new runtime execution model that enables:
- Automatic parallel execution of independent nodes
- Reactive dataflow where nodes execute when all inputs are ready
- Event-driven bypass propagation for conditional paths
- Same JSON agent definition format (no changes required)

Architecture:
- NodeInputTracker: Tracks expected/received inputs for each node
- GraphEventDispatcher: Coordinates events between nodes
- ReactiveExecutor: Main execution engine

The key insight is transforming from "edge-driven" to "node-driven" execution.
"""

from magic_agents.execution.input_tracker import NodeInputTracker
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from magic_agents.execution.reactive_executor import (
    execute_graph_reactive,
    execute_graph_loop_reactive,
)

__all__ = [
    "NodeInputTracker",
    "GraphEventDispatcher",
    "execute_graph_reactive",
    "execute_graph_loop_reactive",
]
