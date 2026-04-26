"""
NodeInputTracker - Tracks which inputs a node is waiting for.

This implements the reactive execution model where nodes wait for all
their expected inputs before becoming ready for execution.

CRITICAL FIX: Tracker now keys by edge.id instead of targetHandle.
This fixes the fan-in bug where multiple edges targeting the same handle
would collapse into a single entry, causing silent data loss.

Each incoming edge is tracked independently via its unique edge.id,
enabling proper fan-in semantics and bypass propagation per edge.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Set, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InputInfo:
    """
    Information about an expected input edge.
    
    CRITICAL: edge_id is the PRIMARY KEY for tracking. Each edge gets a unique
    ID assigned during build, enabling edge-level tracking for fan-in scenarios.
    
    Attributes:
        edge_id: Unique edge identifier (primary key for tracker)
        handle: Target handle name (routing key for dispatch)
        source_node: Source node ID
        source_handle: Source handle name
        content: Received content (None until received)
        received: Whether input has been received
        bypassed: Whether edge was bypassed (conditional path not taken)
    """
    edge_id: str           # PRIMARY KEY - unique per edge
    handle: str            # Target handle name (routing)
    source_node: str
    source_handle: str
    content: Optional[Any] = None
    received: bool = False
    bypassed: bool = False


class NodeInputTracker:
    """
    Tracks which inputs a node is waiting for and their status.
    
    CRITICAL FIX: Now keyed by edge.id instead of targetHandle.
    This enables proper fan-in tracking where multiple edges targeting
    the same handle are tracked independently.
    
    Each node in the graph gets a tracker that:
    1. Knows which input edges are expected (keyed by edge.id)
    2. Tracks when each edge input is received or bypassed
    3. Signals when the node is ready to execute
    
    States:
    - PENDING: Not all inputs accounted for
    - READY: All inputs received/bypassed, ready to execute
    - should_execute: At least one real input received
    - is_bypassed: ALL inputs were bypassed (skip execution)
    """
    
    def __init__(self, node_id: str, expected_inputs: List[InputInfo] = None):
        """
        Initialize input tracker.
        
        CRITICAL: Keys by edge_id, NOT handle. This fixes fan-in collapse bug.
        
        Args:
            node_id: ID of the node being tracked
            expected_inputs: List of InputInfo objects describing expected inputs
        """
        self.node_id = node_id
        # KEY CHANGE: Dict keyed by edge_id, not handle
        self._expected_inputs: Dict[str, InputInfo] = {}
        self._ready_event = asyncio.Event()
        self._lock = asyncio.Lock()
        
        if expected_inputs:
            for info in expected_inputs:
                # KEY BY EDGE_ID - enables multi-edge same-handle tracking
                self._expected_inputs[info.edge_id] = info
        
        # If no inputs expected, node is immediately ready
        if not self._expected_inputs:
            self._ready_event.set()
            logger.debug("Node %s has no inputs - marked ready", node_id)
    
    @property
    def expected_handles(self) -> Set[str]:
        """Get set of expected input handle names (from all edges)."""
        return {info.handle for info in self._expected_inputs.values()}
    
    @property
    def received_handles(self) -> Set[str]:
        """Get set of handles that have received input."""
        return {info.handle for info in self._expected_inputs.values() if info.received}
    
    @property
    def bypassed_handles(self) -> Set[str]:
        """Get set of handles that were bypassed."""
        return {info.handle for info in self._expected_inputs.values() if info.bypassed}
    
    @property
    def expected_edges(self) -> Set[str]:
        """Get set of expected edge IDs."""
        return set(self._expected_inputs.keys())
    
    @property
    def received_edges(self) -> Set[str]:
        """Get set of edge IDs that have received input."""
        return {edge_id for edge_id, info in self._expected_inputs.items() if info.received}
    
    @property
    def bypassed_edges(self) -> Set[str]:
        """Get set of edge IDs that were bypassed."""
        return {edge_id for edge_id, info in self._expected_inputs.items() if info.bypassed}
    
    @property
    def is_ready(self) -> bool:
        """Check if all expected inputs are accounted for (received or bypassed)."""
        if not self._expected_inputs:
            return True
        return all(
            info.received or info.bypassed 
            for info in self._expected_inputs.values()
        )
    
    @property
    def should_execute(self) -> bool:
        """True if node should execute (at least one real input received)."""
        if not self._expected_inputs:
            return True  # Source nodes always execute
        return self.is_ready and len(self.received_edges) > 0
    
    @property
    def is_bypassed(self) -> bool:
        """True if ALL inputs were bypassed (no real data received)."""
        if not self._expected_inputs:
            return False  # Source nodes are never bypassed
        return self.is_ready and len(self.received_edges) == 0
    
    async def receive_input(self, handle: str, content: Any) -> bool:
        """
        Called when input arrives on a handle.
        
        FAN-IN HANDLING: When multiple edges target the same handle,
        we find the first unreceived edge for that handle and mark it received.
        This enables proper multi-edge tracking.
        
        Args:
            handle: Target handle name
            content: The input content
            
        Returns:
            True if this made the node ready
        """
        async with self._lock:
            # Find matching edges by handle (multiple edges may share same handle)
            matching_edges = [
                (edge_id, info) for edge_id, info in self._expected_inputs.items()
                if info.handle == handle and not info.received
            ]
            
            if not matching_edges:
                # No matching unreceived edge for this handle
                # Could be: already received, wrong handle, or no edges
                all_handles_for_edge = [
                    info.handle for info in self._expected_inputs.values()
                ]
                if handle not in all_handles_for_edge:
                    logger.warning(
                        "Node %s received unexpected input on handle %s (no edge targets this handle)",
                        self.node_id, handle
                    )
                else:
                    logger.debug(
                        "Node %s handle %s already received (fan-in complete)",
                        self.node_id, handle
                    )
                return False
            
            # Mark first unreceived edge as received (fan-in: first-wins semantics)
            edge_id, info = matching_edges[0]
            info.content = content
            info.received = True
            info.bypassed = False  # Clear bypass if it was set
            
            logger.debug(
                "Node %s received input on %s via edge %s from %s.%s",
                self.node_id, handle, edge_id, info.source_node, info.source_handle
            )
            
            was_ready = self._check_ready()
            return was_ready
    
    async def receive_bypass(self, handle: str = None, edge_id: str = None) -> bool:
        """
        Called when input is bypassed.
        
        FAN-IN HANDLING: Can bypass by handle (all edges for that handle)
        or by specific edge_id (single edge).
        
        Args:
            handle: Specific handle to bypass (all edges targeting it)
            edge_id: Specific edge to bypass (single edge)
            If both None: bypass all remaining unaccounted edges
            
        Returns:
            True if this made the node ready
        """
        async with self._lock:
            if edge_id:
                # Bypass specific edge by ID
                if edge_id in self._expected_inputs:
                    info = self._expected_inputs[edge_id]
                    if not info.received:
                        info.bypassed = True
                        logger.debug(
                            "Node %s edge %s (handle %s) marked as bypassed",
                            self.node_id, edge_id, info.handle
                        )
            elif handle:
                # Bypass all edges targeting this handle
                for eid, info in self._expected_inputs.items():
                    if info.handle == handle and not info.received:
                        info.bypassed = True
                        logger.debug(
                            "Node %s edge %s (handle %s) marked as bypassed",
                            self.node_id, eid, handle
                        )
            else:
                # Bypass all remaining unaccounted edges
                for eid, info in self._expected_inputs.items():
                    if not info.received and not info.bypassed:
                        info.bypassed = True
                        logger.debug(
                            "Node %s edge %s (handle %s) marked as bypassed (bulk)",
                            self.node_id, eid, info.handle
                        )
            
            was_ready = self._check_ready()
            return was_ready
    
    def _check_ready(self) -> bool:
        """Check if all inputs are accounted for and signal if ready."""
        if self.is_ready:
            if not self._ready_event.is_set():
                self._ready_event.set()
                logger.debug(
                    "Node %s is now READY (received=%d edges, bypassed=%d edges)",
                    self.node_id,
                    len(self.received_edges),
                    len(self.bypassed_edges)
                )
            return True
        return False
    
    async def wait_ready(self, timeout: float = None) -> bool:
        """
        Wait until node is ready to execute.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if should execute, False if bypassed
            
        Raises:
            asyncio.TimeoutError if timeout reached
        """
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                "Node %s timed out waiting for inputs. "
                "Received edges: %s, Missing edges: %s",
                self.node_id,
                self.received_edges,
                self.expected_edges - self.received_edges - self.bypassed_edges
            )
            raise
        
        return self.should_execute
    
    def get_input(self, handle: str) -> Optional[Any]:
        """
        Get the content received on a specific handle.
        
        FAN-IN: Returns content from first received edge for this handle.
        For multi-edge same-handle scenarios, use get_input_by_edge().
        """
        for info in self._expected_inputs.values():
            if info.handle == handle and info.received:
                return info.content
        return None
    
    def get_input_by_edge(self, edge_id: str) -> Optional[Any]:
        """Get content received on a specific edge."""
        if edge_id in self._expected_inputs:
            return self._expected_inputs[edge_id].content
        return None
    
    def get_all_inputs(self) -> Dict[str, Any]:
        """
        Get all received inputs as a dictionary keyed by handle.
        
        FAN-IN: For same-handle multi-edge, last received wins in dict.
        Use get_all_inputs_by_edge() for edge-level data.
        """
        result = {}
        for info in self._expected_inputs.values():
            if info.received and info.content is not None:
                result[info.handle] = info.content
        return result
    
    def get_all_inputs_by_edge(self) -> Dict[str, Any]:
        """Get all received inputs keyed by edge_id."""
        return {
            edge_id: info.content
            for edge_id, info in self._expected_inputs.items()
            if info.received and info.content is not None
        }
    
    def get_edge_info(self, edge_id: str) -> Optional[InputInfo]:
        """Get InputInfo for a specific edge."""
        return self._expected_inputs.get(edge_id)
    
    def reset(self):
        """Reset tracker for re-execution (e.g., in loops)."""
        self._ready_event.clear()
        for info in self._expected_inputs.values():
            info.content = None
            info.received = False
            info.bypassed = False
        
        # If no inputs, immediately ready again
        if not self._expected_inputs:
            self._ready_event.set()
        
        logger.debug("Node %s tracker reset", self.node_id)
    
    def __repr__(self) -> str:
        status = "READY" if self.is_ready else "PENDING"
        return (
            f"NodeInputTracker({self.node_id}, {status}, "
            f"edges={len(self._expected_inputs)}, "
            f"received={len(self.received_edges)}, bypassed={len(self.bypassed_edges)})"
        )
