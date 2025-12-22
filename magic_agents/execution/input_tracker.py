"""
NodeInputTracker - Tracks which inputs a node is waiting for.

This implements the reactive execution model where nodes wait for all
their expected inputs before becoming ready for execution.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Set, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InputInfo:
    """Information about an expected input."""
    handle: str
    source_node: str
    source_handle: str
    content: Optional[Any] = None
    received: bool = False
    bypassed: bool = False


class NodeInputTracker:
    """
    Tracks which inputs a node is waiting for and their status.
    
    Each node in the graph gets a tracker that:
    1. Knows which input handles are expected (from edges)
    2. Tracks when each input is received or bypassed
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
        
        Args:
            node_id: ID of the node being tracked
            expected_inputs: List of InputInfo objects describing expected inputs
        """
        self.node_id = node_id
        self._expected_inputs: Dict[str, InputInfo] = {}
        self._ready_event = asyncio.Event()
        self._lock = asyncio.Lock()
        
        if expected_inputs:
            for info in expected_inputs:
                self._expected_inputs[info.handle] = info
        
        # If no inputs expected, node is immediately ready
        if not self._expected_inputs:
            self._ready_event.set()
            logger.debug("Node %s has no inputs - marked ready", node_id)
    
    @property
    def expected_handles(self) -> Set[str]:
        """Get set of expected input handle names."""
        return set(self._expected_inputs.keys())
    
    @property
    def received_handles(self) -> Set[str]:
        """Get set of handles that have received input."""
        return {h for h, info in self._expected_inputs.items() if info.received}
    
    @property
    def bypassed_handles(self) -> Set[str]:
        """Get set of handles that were bypassed."""
        return {h for h, info in self._expected_inputs.items() if info.bypassed}
    
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
        return self.is_ready and len(self.received_handles) > 0
    
    @property
    def is_bypassed(self) -> bool:
        """True if ALL inputs were bypassed (no real data received)."""
        if not self._expected_inputs:
            return False  # Source nodes are never bypassed
        return self.is_ready and len(self.received_handles) == 0
    
    async def receive_input(self, handle: str, content: Any) -> bool:
        """
        Called when input arrives on a handle.
        
        Args:
            handle: Target handle name
            content: The input content
            
        Returns:
            True if this made the node ready
        """
        async with self._lock:
            if handle not in self._expected_inputs:
                # Unexpected input - might be optional or misconfigured
                logger.warning(
                    "Node %s received unexpected input on handle %s",
                    self.node_id, handle
                )
                return False
            
            info = self._expected_inputs[handle]
            info.content = content
            info.received = True
            info.bypassed = False  # Clear bypass if it was set
            
            logger.debug(
                "Node %s received input on %s from %s.%s",
                self.node_id, handle, info.source_node, info.source_handle
            )
            
            was_ready = self._check_ready()
            return was_ready
    
    async def receive_bypass(self, handle: str = None) -> bool:
        """
        Called when input handle is bypassed.
        
        Args:
            handle: Specific handle to bypass, or None to bypass all remaining
            
        Returns:
            True if this made the node ready
        """
        async with self._lock:
            if handle:
                if handle in self._expected_inputs:
                    info = self._expected_inputs[handle]
                    if not info.received:  # Don't override received input
                        info.bypassed = True
                        logger.debug(
                            "Node %s handle %s marked as bypassed",
                            self.node_id, handle
                        )
            else:
                # Bypass all remaining unaccounted handles
                for h, info in self._expected_inputs.items():
                    if not info.received and not info.bypassed:
                        info.bypassed = True
                        logger.debug(
                            "Node %s handle %s marked as bypassed (bulk)",
                            self.node_id, h
                        )
            
            was_ready = self._check_ready()
            return was_ready
    
    def _check_ready(self) -> bool:
        """Check if all inputs are accounted for and signal if ready."""
        if self.is_ready:
            if not self._ready_event.is_set():
                self._ready_event.set()
                logger.debug(
                    "Node %s is now READY (received=%d, bypassed=%d)",
                    self.node_id,
                    len(self.received_handles),
                    len(self.bypassed_handles)
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
                "Received: %s, Missing: %s",
                self.node_id,
                self.received_handles,
                self.expected_handles - self.received_handles - self.bypassed_handles
            )
            raise
        
        return self.should_execute
    
    def get_input(self, handle: str) -> Optional[Any]:
        """Get the content received on a specific handle."""
        if handle in self._expected_inputs:
            return self._expected_inputs[handle].content
        return None
    
    def get_all_inputs(self) -> Dict[str, Any]:
        """Get all received inputs as a dictionary."""
        return {
            handle: info.content
            for handle, info in self._expected_inputs.items()
            if info.received and info.content is not None
        }
    
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
            f"received={self.received_handles}, bypassed={self.bypassed_handles})"
        )
