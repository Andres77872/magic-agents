"""
GraphEventDispatcher - Coordinates event flow between nodes.

This is the central coordinator that enables reactive parallel execution.
It manages:
- Input/output routing between nodes
- Bypass event propagation for conditional paths
- Node state tracking
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Set, Optional
from enum import Enum
from dataclasses import dataclass, field

from magic_agents.execution.input_tracker import NodeInputTracker, InputInfo
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """State machine for node execution."""
    PENDING = "pending"       # Waiting for inputs
    READY = "ready"          # All inputs received, ready to execute
    EXECUTING = "executing"  # Currently running
    COMPLETED = "completed"  # Finished successfully
    BYPASSED = "bypassed"    # Skipped due to conditional bypass
    ERROR = "error"          # Execution failed


@dataclass
class NodeExecution:
    """Tracks execution state and outputs for a node."""
    node_id: str
    state: NodeState = NodeState.PENDING
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    task: Optional[asyncio.Task] = None


class GraphEventDispatcher:
    """
    Coordinates event flow between nodes for reactive execution.
    
    This class builds dependency maps at initialization and provides
    methods to dispatch inputs, propagate bypasses, and track state.
    
    Key responsibilities:
    1. Build incoming/outgoing edge maps for quick lookup
    2. Create input trackers for each node
    3. Route outputs from completed nodes to downstream nodes
    4. Propagate bypass events through conditional paths
    5. Track overall execution state
    """
    
    def __init__(
        self,
        nodes: Dict[str, Any],
        edges: List[EdgeNodeModel],
        max_concurrent: int = 10,
        timeout: float = 60.0
    ):
        """
        Initialize the event dispatcher.
        
        Args:
            nodes: Dictionary of node_id -> Node instance
            edges: List of EdgeNodeModel defining connections
            max_concurrent: Maximum concurrent node executions
            timeout: Timeout for waiting on inputs (seconds)
        """
        self.nodes = nodes
        self.edges = edges
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
        # Build dependency maps
        self._incoming: Dict[str, List[EdgeNodeModel]] = {}
        self._outgoing: Dict[str, List[EdgeNodeModel]] = {}
        self._build_edge_maps()
        
        # Create input trackers for each node
        self._trackers: Dict[str, NodeInputTracker] = {}
        self._build_trackers()
        
        # Track node execution state
        self._executions: Dict[str, NodeExecution] = {
            node_id: NodeExecution(node_id=node_id)
            for node_id in nodes.keys()
        }
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Output queue for streaming results
        self._output_queue: asyncio.Queue = asyncio.Queue()
        
        logger.debug(
            "GraphEventDispatcher initialized: %d nodes, %d edges",
            len(nodes), len(edges)
        )
    
    def _build_edge_maps(self):
        """Build incoming and outgoing edge maps for quick lookup."""
        for edge in self.edges:
            # Incoming edges (edges where this node is the target)
            self._incoming.setdefault(edge.target, []).append(edge)
            # Outgoing edges (edges where this node is the source)
            self._outgoing.setdefault(edge.source, []).append(edge)
    
    def _build_trackers(self):
        """Create input trackers for each node based on incoming edges."""
        for node_id in self.nodes.keys():
            incoming = self._incoming.get(node_id, [])
            
            # Build expected inputs from incoming edges
            expected_inputs = [
                InputInfo(
                    handle=edge.targetHandle,
                    source_node=edge.source,
                    source_handle=edge.sourceHandle
                )
                for edge in incoming
            ]
            
            self._trackers[node_id] = NodeInputTracker(
                node_id=node_id,
                expected_inputs=expected_inputs
            )
            
            logger.debug(
                "Node %s tracker: expects %d inputs from %s",
                node_id,
                len(expected_inputs),
                [i.source_node for i in expected_inputs]
            )
    
    def get_source_nodes(self) -> List[str]:
        """Get nodes with no incoming edges (entry points)."""
        return [
            node_id for node_id in self.nodes.keys()
            if not self._incoming.get(node_id)
        ]
    
    def get_tracker(self, node_id: str) -> Optional[NodeInputTracker]:
        """Get the input tracker for a node."""
        return self._trackers.get(node_id)
    
    def get_state(self, node_id: str) -> NodeState:
        """Get the current state of a node."""
        if node_id in self._executions:
            return self._executions[node_id].state
        return NodeState.PENDING
    
    def set_state(self, node_id: str, state: NodeState):
        """Set the state of a node."""
        if node_id in self._executions:
            old_state = self._executions[node_id].state
            self._executions[node_id].state = state
            logger.debug("Node %s: %s -> %s", node_id, old_state.value, state.value)
    
    async def dispatch_input(self, target_node_id: str, handle: str, content: Any):
        """
        Deliver input to a node's handle.
        
        Args:
            target_node_id: ID of the receiving node
            handle: Target handle name
            content: The content to deliver
        """
        if target_node_id not in self._trackers:
            logger.warning("Unknown target node: %s", target_node_id)
            return
        
        tracker = self._trackers[target_node_id]
        node = self.nodes.get(target_node_id)
        
        # Store in node's inputs dict
        if node:
            node.inputs[handle] = content
        
        # Notify tracker
        await tracker.receive_input(handle, content)
    
    async def dispatch_bypass(self, target_node_id: str, handle: str = None):
        """
        Notify a node that an input is bypassed.
        
        Args:
            target_node_id: ID of the receiving node
            handle: Specific handle to bypass, or None for all
        """
        if target_node_id not in self._trackers:
            return
        
        await self._trackers[target_node_id].receive_bypass(handle)
    
    async def propagate_outputs(self, source_node_id: str, outputs: Dict[str, Any]):
        """
        Propagate a node's outputs to all downstream nodes.
        
        Args:
            source_node_id: ID of the node that produced outputs
            outputs: Dictionary of output handle -> content
        """
        for edge in self._outgoing.get(source_node_id, []):
            source_handle = edge.sourceHandle
            
            if source_handle in outputs:
                output = outputs[source_handle]
                # Extract content from prep() wrapper if present
                if isinstance(output, dict) and 'content' in output:
                    content = output['content']
                else:
                    content = output
                
                await self.dispatch_input(
                    edge.target,
                    edge.targetHandle,
                    content
                )
                
                logger.debug(
                    "Propagated %s.%s -> %s.%s",
                    source_node_id, source_handle,
                    edge.target, edge.targetHandle
                )
    
    async def propagate_conditional_bypass(
        self,
        conditional_node_id: str,
        selected_handle: str
    ) -> Dict[str, Any]:
        """
        Enhanced bypass propagation with fan-out support.
        
        When a conditional selects a handle:
        1. All edges with selected_handle -> targets EXECUTE
        2. All edges with other handles -> targets get BYPASS signal
        
        Handles fan-out correctly: if multiple edges share the same
        sourceHandle, all their targets will be treated the same way.
        
        Args:
            conditional_node_id: ID of the conditional node
            selected_handle: The handle that was selected (not bypassed)
            
        Returns:
            Dict with selected_handle, active_targets, and bypassed_targets
        """
        outgoing = self._outgoing.get(conditional_node_id, [])
        
        # Group targets by their selection status
        selected_targets = set()
        bypassed_targets = set()
        
        for edge in outgoing:
            if edge.sourceHandle == selected_handle:
                selected_targets.add(edge.target)
                logger.debug(
                    "Active path: %s.%s -> %s",
                    conditional_node_id, edge.sourceHandle, edge.target
                )
            else:
                bypassed_targets.add(edge.target)
                logger.debug(
                    "Bypass path: %s.%s -> %s",
                    conditional_node_id, edge.sourceHandle, edge.target
                )
        
        # Handle special case: target appears in both selected and bypassed
        # (shouldn't happen in well-formed graphs, but be defensive)
        # If a target is reachable via selected handle, don't bypass it
        bypassed_targets -= selected_targets
        
        # Propagate bypass to non-selected targets
        for target in bypassed_targets:
            await self._recursive_bypass(target)
        
        return {
            "selected_handle": selected_handle,
            "active_targets": list(selected_targets),
            "bypassed_targets": list(bypassed_targets)
        }
    
    async def handle_bypass_all_signal(self, node_id: str):
        """
        Handle BYPASS_ALL signal from conditional (error case).
        
        When a conditional emits BYPASS_ALL, all its downstream
        nodes should be bypassed regardless of the handle.
        
        Args:
            node_id: ID of the node that emitted BYPASS_ALL
        """
        logger.warning("Node %s emitted BYPASS_ALL signal, bypassing all downstream", node_id)
        for edge in self._outgoing.get(node_id, []):
            await self._recursive_bypass(edge.target)
    
    async def _recursive_bypass(self, node_id: str):
        """
        Recursively bypass a node and all its downstream nodes.
        
        A node is bypassed if ALL its incoming edges are bypassed.
        """
        if self.get_state(node_id) in (NodeState.BYPASSED, NodeState.COMPLETED):
            return
        
        tracker = self._trackers.get(node_id)
        if not tracker:
            return
        
        # Mark all inputs as bypassed
        await tracker.receive_bypass()
        
        # If the node should bypass (all inputs bypassed)
        if tracker.is_bypassed:
            self.set_state(node_id, NodeState.BYPASSED)
            
            # Mark the node as bypassed
            node = self.nodes.get(node_id)
            if node and hasattr(node, 'mark_bypassed'):
                node.mark_bypassed()
            
            # Recursively bypass downstream
            for edge in self._outgoing.get(node_id, []):
                await self._recursive_bypass(edge.target)
    
    def get_ready_nodes(self) -> List[str]:
        """Get list of nodes that are ready to execute."""
        ready = []
        for node_id, tracker in self._trackers.items():
            state = self.get_state(node_id)
            if state == NodeState.PENDING and tracker.is_ready:
                if tracker.should_execute:
                    ready.append(node_id)
        return ready
    
    def all_completed(self) -> bool:
        """Check if all nodes have completed or been bypassed."""
        for node_id in self.nodes.keys():
            state = self.get_state(node_id)
            if state not in (NodeState.COMPLETED, NodeState.BYPASSED, NodeState.ERROR):
                return False
        return True
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of execution state."""
        states = {}
        for state in NodeState:
            states[state.value] = []
        
        for node_id, execution in self._executions.items():
            states[execution.state.value].append(node_id)
        
        return {
            "total": len(self.nodes),
            "states": states,
            "completed": len(states[NodeState.COMPLETED.value]),
            "bypassed": len(states[NodeState.BYPASSED.value]),
            "errors": len(states[NodeState.ERROR.value]),
            "pending": len(states[NodeState.PENDING.value]),
        }
    
    def reset_for_iteration(self, node_ids: List[str] = None):
        """
        Reset state for iteration (loop execution).
        
        Args:
            node_ids: Specific nodes to reset, or None for all
        """
        nodes_to_reset = node_ids if node_ids else list(self.nodes.keys())
        
        for node_id in nodes_to_reset:
            if node_id in self._trackers:
                self._trackers[node_id].reset()
            if node_id in self._executions:
                self._executions[node_id].state = NodeState.PENDING
                self._executions[node_id].outputs.clear()
                self._executions[node_id].error = None
                self._executions[node_id].task = None
            
            # Also reset the node itself
            node = self.nodes.get(node_id)
            if node:
                node._response = None
                node.outputs.clear()
                if hasattr(node, 'generated'):
                    node.generated = ''
        
        logger.debug("Reset %d nodes for iteration", len(nodes_to_reset))
