"""
Debug Event Type Definitions and Data Structures.

This module defines the unified event model for all debug events in the system.
All debug events conform to the DebugEvent structure, enabling consistent
handling, filtering, storage, and backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class DebugEventType(Enum):
    """
    Enumeration of all debug event types.
    
    Organized into categories:
    - Lifecycle: Node/Graph lifecycle transitions
    - Data Flow: Data movement between nodes
    - State: Internal state transitions
    - Error: Error conditions
    - Diagnostic: Debugging/analysis events
    """
    
    # Lifecycle events
    NODE_INIT = "node_init"
    NODE_START = "node_start"
    NODE_END = "node_end"
    NODE_ERROR = "node_error"
    NODE_BYPASS = "node_bypass"
    GRAPH_START = "graph_start"
    GRAPH_END = "graph_end"
    ITERATION_START = "iteration_start"
    ITERATION_END = "iteration_end"
    
    # Data flow events
    INPUT_RECEIVED = "input_received"
    OUTPUT_PRODUCED = "output_produced"
    EDGE_TRAVERSED = "edge_traversed"
    DATA_TRANSFORMED = "data_transformed"
    
    # State events
    STATE_CHANGE = "state_change"
    
    # Error events
    VALIDATION_ERROR = "validation_error"
    ROUTING_ERROR = "routing_error"
    TIMEOUT_ERROR = "timeout_error"
    INPUT_ERROR = "input_error"
    TEMPLATE_ERROR = "template_error"
    PARSE_ERROR = "parse_error"
    
    # Diagnostic events
    CONDITION_EVALUATED = "condition_evaluated"
    TEMPLATE_RENDERED = "template_rendered"
    LLM_GENERATION = "llm_generation"
    TIMING_CHECKPOINT = "timing_checkpoint"


class DebugEventSeverity(Enum):
    """
    Severity levels for debug events.
    
    Used for filtering and display purposes:
    - TRACE: Very detailed, typically disabled in production
    - DEBUG: Development-time debugging information
    - INFO: General execution information
    - WARN: Potential issues that don't stop execution
    - ERROR: Errors that affect execution
    """
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass
class DebugEvent:
    """
    Unified debug event structure.
    
    All debug events in the system use this structure, enabling
    consistent handling, filtering, and storage.
    
    Attributes:
        event_id: Unique identifier for this event
        event_type: Type of debug event
        severity: Severity level for filtering
        timestamp: When the event occurred
        execution_id: Unique ID for the graph execution
        sequence_number: Order within execution
        node_id: ID of the node (if node-specific)
        node_type: Type of the node (if node-specific)
        node_class: Class name of the node (if node-specific)
        payload: Event-specific data
        parent_event_id: ID of parent event (for nested events)
        related_event_ids: IDs of related events
        tags: Additional tags for categorization
    """
    
    # Identity
    event_id: str = field(default_factory=lambda: uuid4().hex)
    event_type: DebugEventType = DebugEventType.NODE_START
    severity: DebugEventSeverity = DebugEventSeverity.INFO
    
    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # Execution context
    execution_id: str = ""
    sequence_number: int = 0
    
    # Node context (optional - not all events are node-specific)
    node_id: Optional[str] = None
    node_type: Optional[str] = None
    node_class: Optional[str] = None
    
    # Payload - event-specific data
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Relationships
    parent_event_id: Optional[str] = None
    related_event_ids: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the event
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "execution_id": self.execution_id,
            "sequence_number": self.sequence_number,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "node_class": self.node_class,
            "payload": self.payload,
            "parent_event_id": self.parent_event_id,
            "related_event_ids": self.related_event_ids,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DebugEvent:
        """
        Create a DebugEvent from a dictionary.
        
        Args:
            data: Dictionary with event data
            
        Returns:
            DebugEvent instance
        """
        return cls(
            event_id=data.get("event_id", uuid4().hex),
            event_type=DebugEventType(data.get("event_type", "node_start")),
            severity=DebugEventSeverity(data.get("severity", "info")),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(UTC)
            ),
            execution_id=data.get("execution_id", ""),
            sequence_number=data.get("sequence_number", 0),
            node_id=data.get("node_id"),
            node_type=data.get("node_type"),
            node_class=data.get("node_class"),
            payload=data.get("payload", {}),
            parent_event_id=data.get("parent_event_id"),
            related_event_ids=data.get("related_event_ids", []),
            tags=data.get("tags", []),
        )
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert to legacy debug format for backward compatibility.
        
        Maps new event structure to existing NodeDebugInfo-like format
        used by the current debug system.
        
        Returns:
            Dictionary in legacy format
        """
        legacy = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "node_class": self.node_class,
        }
        
        # Map event-specific payload to legacy fields
        if self.event_type == DebugEventType.NODE_END:
            legacy.update({
                "start_time": self.payload.get("start_time"),
                "end_time": self.payload.get("end_time"),
                "execution_duration_ms": self.payload.get("duration_ms"),
                "inputs": self.payload.get("inputs", {}),
                "outputs": self.payload.get("outputs", {}),
                "internal_variables": self.payload.get("internal_state", {}),
                "was_executed": True,
                "was_bypassed": False,
                "error": None,
            })
        elif self.event_type == DebugEventType.NODE_BYPASS:
            legacy.update({
                "inputs": self.payload.get("inputs_at_bypass", {}),
                "outputs": {},
                "internal_variables": {},
                "was_executed": False,
                "was_bypassed": True,
                "bypass_reason": self.payload.get("reason"),
                "bypass_source": self.payload.get("bypass_source"),
                "error": None,
            })
        elif self.event_type == DebugEventType.NODE_START:
            legacy.update({
                "start_time": self.payload.get("start_time"),
                "inputs": self.payload.get("inputs", {}),
                "was_executed": False,
                "was_bypassed": False,
            })
        elif self.event_type in (
            DebugEventType.NODE_ERROR,
            DebugEventType.INPUT_ERROR,
            DebugEventType.TIMEOUT_ERROR,
            DebugEventType.VALIDATION_ERROR,
            DebugEventType.ROUTING_ERROR,
            DebugEventType.TEMPLATE_ERROR,
            DebugEventType.PARSE_ERROR,
        ):
            legacy.update({
                "error_type": self.payload.get("error_type"),
                "error_message": self.payload.get("error_message"),
                "error": self.payload.get("error_message"),
                "timestamp": self.timestamp.isoformat(),
                "context": self.payload.get("context"),
                "was_executed": False,
                "was_bypassed": False,
            })
        elif self.event_type == DebugEventType.GRAPH_START:
            legacy.update({
                "graph_type": self.payload.get("graph_type"),
                "start_time": self.payload.get("start_time"),
            })
        elif self.event_type == DebugEventType.GRAPH_END:
            legacy.update({
                "total_nodes": self.payload.get("total_nodes"),
                "executed_nodes": self.payload.get("executed_nodes"),
                "bypassed_nodes": self.payload.get("bypassed_nodes"),
                "failed_nodes": self.payload.get("failed_nodes"),
                "start_time": self.payload.get("start_time"),
                "end_time": self.payload.get("end_time"),
                "total_duration_ms": self.payload.get("total_duration_ms"),
            })
        elif self.event_type == DebugEventType.CONDITION_EVALUATED:
            legacy.update({
                "condition": self.payload.get("condition"),
                "result": self.payload.get("result"),
                "selected_handle": self.payload.get("selected_handle"),
            })
        elif self.event_type == DebugEventType.LLM_GENERATION:
            legacy.update({
                "model": self.payload.get("model"),
                "prompt_tokens": self.payload.get("prompt_tokens"),
                "completion_tokens": self.payload.get("completion_tokens"),
                "total_tokens": self.payload.get("total_tokens"),
                "response_preview": self.payload.get("response_preview"),
            })
        else:
            # For other events, just include the payload
            legacy.update(self.payload)
        
        return legacy
    
    def with_payload(self, **kwargs) -> DebugEvent:
        """
        Create a new event with updated payload.
        
        Args:
            **kwargs: Key-value pairs to add/update in payload
            
        Returns:
            New DebugEvent with updated payload
        """
        new_payload = {**self.payload, **kwargs}
        return DebugEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            severity=self.severity,
            timestamp=self.timestamp,
            execution_id=self.execution_id,
            sequence_number=self.sequence_number,
            node_id=self.node_id,
            node_type=self.node_type,
            node_class=self.node_class,
            payload=new_payload,
            parent_event_id=self.parent_event_id,
            related_event_ids=self.related_event_ids.copy(),
            tags=self.tags.copy(),
        )
    
    def with_tags(self, *tags: str) -> DebugEvent:
        """
        Create a new event with additional tags.
        
        Args:
            *tags: Tags to add
            
        Returns:
            New DebugEvent with additional tags
        """
        new_tags = list(set(self.tags + list(tags)))
        return DebugEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            severity=self.severity,
            timestamp=self.timestamp,
            execution_id=self.execution_id,
            sequence_number=self.sequence_number,
            node_id=self.node_id,
            node_type=self.node_type,
            node_class=self.node_class,
            payload=self.payload.copy(),
            parent_event_id=self.parent_event_id,
            related_event_ids=self.related_event_ids.copy(),
            tags=new_tags,
        )
    
    def is_error(self) -> bool:
        """Check if this is an error event."""
        return self.event_type in (
            DebugEventType.NODE_ERROR,
            DebugEventType.VALIDATION_ERROR,
            DebugEventType.ROUTING_ERROR,
            DebugEventType.TIMEOUT_ERROR,
            DebugEventType.INPUT_ERROR,
            DebugEventType.TEMPLATE_ERROR,
            DebugEventType.PARSE_ERROR,
        )
    
    def is_lifecycle(self) -> bool:
        """Check if this is a lifecycle event."""
        return self.event_type in (
            DebugEventType.NODE_INIT,
            DebugEventType.NODE_START,
            DebugEventType.NODE_END,
            DebugEventType.NODE_ERROR,
            DebugEventType.NODE_BYPASS,
            DebugEventType.GRAPH_START,
            DebugEventType.GRAPH_END,
            DebugEventType.ITERATION_START,
            DebugEventType.ITERATION_END,
        )
    
    def is_data_flow(self) -> bool:
        """Check if this is a data flow event."""
        return self.event_type in (
            DebugEventType.INPUT_RECEIVED,
            DebugEventType.OUTPUT_PRODUCED,
            DebugEventType.EDGE_TRAVERSED,
            DebugEventType.DATA_TRANSFORMED,
        )


# Convenience factory functions for common event types
def node_start_event(
    execution_id: str,
    node_id: str,
    node_type: str,
    node_class: str,
    inputs: Dict[str, Any],
    sequence: int = 0,
) -> DebugEvent:
    """Create a NODE_START event."""
    return DebugEvent(
        event_type=DebugEventType.NODE_START,
        execution_id=execution_id,
        sequence_number=sequence,
        node_id=node_id,
        node_type=node_type,
        node_class=node_class,
        payload={
            "inputs": inputs,
            "start_time": datetime.now(UTC).isoformat(),
        },
    )


def node_end_event(
    execution_id: str,
    node_id: str,
    node_type: str,
    node_class: str,
    outputs: Dict[str, Any],
    internal_state: Dict[str, Any],
    duration_ms: float,
    start_time: datetime,
    sequence: int = 0,
) -> DebugEvent:
    """Create a NODE_END event."""
    return DebugEvent(
        event_type=DebugEventType.NODE_END,
        execution_id=execution_id,
        sequence_number=sequence,
        node_id=node_id,
        node_type=node_type,
        node_class=node_class,
        payload={
            "outputs": outputs,
            "internal_state": internal_state,
            "duration_ms": duration_ms,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now(UTC).isoformat(),
        },
    )


def node_error_event(
    execution_id: str,
    node_id: str,
    node_type: str,
    node_class: str,
    error: Exception,
    context: Dict[str, Any] = None,
    sequence: int = 0,
) -> DebugEvent:
    """Create a NODE_ERROR event."""
    return DebugEvent(
        event_type=DebugEventType.NODE_ERROR,
        severity=DebugEventSeverity.ERROR,
        execution_id=execution_id,
        sequence_number=sequence,
        node_id=node_id,
        node_type=node_type,
        node_class=node_class,
        payload={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        },
    )


def node_bypass_event(
    execution_id: str,
    node_id: str,
    node_type: str,
    node_class: str,
    reason: str,
    bypass_source: str,
    inputs_at_bypass: Dict[str, Any] = None,
    sequence: int = 0,
) -> DebugEvent:
    """Create a NODE_BYPASS event."""
    return DebugEvent(
        event_type=DebugEventType.NODE_BYPASS,
        execution_id=execution_id,
        sequence_number=sequence,
        node_id=node_id,
        node_type=node_type,
        node_class=node_class,
        payload={
            "reason": reason,
            "bypass_source": bypass_source,
            "inputs_at_bypass": inputs_at_bypass or {},
        },
    )


def graph_start_event(
    execution_id: str,
    graph_type: str,
    total_nodes: int = 0,
    sequence: int = 0,
) -> DebugEvent:
    """Create a GRAPH_START event."""
    return DebugEvent(
        event_type=DebugEventType.GRAPH_START,
        execution_id=execution_id,
        sequence_number=sequence,
        payload={
            "graph_type": graph_type,
            "total_nodes": total_nodes,
            "start_time": datetime.now(UTC).isoformat(),
        },
    )


def graph_end_event(
    execution_id: str,
    total_nodes: int,
    executed_nodes: int,
    bypassed_nodes: int,
    failed_nodes: int,
    start_time: datetime,
    sequence: int = 0,
) -> DebugEvent:
    """Create a GRAPH_END event."""
    end_time = datetime.now(UTC)
    return DebugEvent(
        event_type=DebugEventType.GRAPH_END,
        execution_id=execution_id,
        sequence_number=sequence,
        payload={
            "total_nodes": total_nodes,
            "executed_nodes": executed_nodes,
            "bypassed_nodes": bypassed_nodes,
            "failed_nodes": failed_nodes,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_ms": (end_time - start_time).total_seconds() * 1000,
        },
    )
