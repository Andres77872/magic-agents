"""
Debug Event Collection and Aggregation.

This module provides collectors that aggregate debug events into
summaries, compatible with the existing GraphDebugFeedback format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from .events import DebugEvent, DebugEventType, DebugEventSeverity


@dataclass
class NodeExecutionSummary:
    """
    Summary of a single node's execution.
    
    Compatible with the existing NodeDebugInfo format.
    """
    node_id: str
    node_type: str
    node_class: str
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Data
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    internal_state: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    was_executed: bool = False
    was_bypassed: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    # Events
    events: List[DebugEvent] = field(default_factory=list)
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert to NodeDebugInfo-compatible format.
        
        Returns:
            Dictionary matching NodeDebugInfo structure
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "node_class": self.node_class,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "execution_duration_ms": self.duration_ms,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "internal_variables": self.internal_state,
            "was_executed": self.was_executed,
            "was_bypassed": self.was_bypassed,
            "error": self.error,
            "extra_info": {
                "error_type": self.error_type,
                "event_count": len(self.events),
            } if self.error_type else {},
        }


@dataclass
class GraphExecutionSummary:
    """
    Summary of a complete graph execution.
    
    Compatible with the existing GraphDebugFeedback format.
    """
    execution_id: str
    graph_type: str
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    
    # Node summaries
    nodes: Dict[str, NodeExecutionSummary] = field(default_factory=dict)
    
    # Counts
    total_nodes: int = 0
    executed_nodes: int = 0
    bypassed_nodes: int = 0
    failed_nodes: int = 0
    
    # Edges
    edges_processed: List[Dict[str, str]] = field(default_factory=list)
    
    # Events
    all_events: List[DebugEvent] = field(default_factory=list)
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert to GraphDebugFeedback-compatible format.
        
        Returns:
            Dictionary matching GraphDebugFeedback structure
        """
        return {
            "execution_id": self.execution_id,
            "graph_type": self.graph_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "nodes": [
                node.to_legacy_format()
                for node in self.nodes.values()
            ],
            "total_nodes": self.total_nodes,
            "executed_nodes": self.executed_nodes,
            "bypassed_nodes": self.bypassed_nodes,
            "failed_nodes": self.failed_nodes,
            "edges_processed": self.edges_processed,
        }


class DebugCollector:
    """
    Collects and aggregates debug events into summaries.
    
    This class receives debug events and builds up node and graph
    summaries that can be converted to the legacy format.
    
    Example:
        collector = DebugCollector(execution_id="abc123", graph_type="agent")
        
        # As events come in
        collector.collect(node_start_event)
        collector.collect(node_end_event)
        
        # Get the summary
        summary = collector.get_summary()
        legacy = summary.to_legacy_format()
    """
    
    def __init__(
        self,
        execution_id: str,
        graph_type: str,
        total_nodes: int = 0,
    ):
        """
        Initialize the collector.
        
        Args:
            execution_id: Unique identifier for this execution
            graph_type: Type of graph being executed
            total_nodes: Total number of nodes in the graph
        """
        self._execution_id = execution_id
        self._graph_type = graph_type
        self._total_nodes = total_nodes
        
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        
        self._nodes: Dict[str, NodeExecutionSummary] = {}
        self._events: List[DebugEvent] = []
        self._edges: List[Dict[str, str]] = []
        
        self._executed_count = 0
        self._bypassed_count = 0
        self._failed_count = 0
    
    @property
    def execution_id(self) -> str:
        """Get the execution ID."""
        return self._execution_id
    
    def collect(self, event: DebugEvent) -> None:
        """
        Collect a debug event.
        
        Updates internal summaries based on the event type.
        
        Args:
            event: The event to collect
        """
        self._events.append(event)
        
        # Handle graph-level events
        if event.event_type == DebugEventType.GRAPH_START:
            self._start_time = event.timestamp
            if "total_nodes" in event.payload:
                self._total_nodes = event.payload["total_nodes"]
        
        elif event.event_type == DebugEventType.GRAPH_END:
            self._end_time = event.timestamp
        
        # Handle node-level events
        elif event.node_id:
            self._collect_node_event(event)
        
        # Handle edge traversal
        elif event.event_type == DebugEventType.EDGE_TRAVERSED:
            self._edges.append({
                "source": event.payload.get("source_node", ""),
                "target": event.payload.get("target_node", ""),
                "source_handle": event.payload.get("source_handle", ""),
                "target_handle": event.payload.get("target_handle", ""),
            })
    
    def _collect_node_event(self, event: DebugEvent) -> None:
        """Collect a node-specific event."""
        node_id = event.node_id
        
        # Ensure node summary exists
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeExecutionSummary(
                node_id=node_id,
                node_type=event.node_type or "unknown",
                node_class=event.node_class or "unknown",
            )
        
        summary = self._nodes[node_id]
        summary.events.append(event)
        
        # Update summary based on event type
        if event.event_type == DebugEventType.NODE_START:
            summary.start_time = event.timestamp
            if "inputs" in event.payload:
                summary.inputs = event.payload["inputs"]
        
        elif event.event_type == DebugEventType.NODE_END:
            summary.end_time = event.timestamp
            summary.was_executed = True
            
            if "outputs" in event.payload:
                summary.outputs = event.payload["outputs"]
            if "internal_state" in event.payload:
                summary.internal_state = event.payload["internal_state"]
            if "duration_ms" in event.payload:
                summary.duration_ms = event.payload["duration_ms"]
            
            self._executed_count += 1
        
        elif event.event_type == DebugEventType.NODE_BYPASS:
            summary.was_bypassed = True
            if "inputs_at_bypass" in event.payload:
                summary.inputs = event.payload["inputs_at_bypass"]
            
            self._bypassed_count += 1
        
        elif event.event_type == DebugEventType.NODE_ERROR:
            summary.error = event.payload.get("error_message")
            summary.error_type = event.payload.get("error_type")
            
            self._failed_count += 1
        
        elif event.event_type == DebugEventType.INPUT_RECEIVED:
            # Merge input data
            if "data_preview" in event.payload:
                handle = event.payload.get("handle", "input")
                summary.inputs[handle] = event.payload["data_preview"]
        
        elif event.event_type == DebugEventType.OUTPUT_PRODUCED:
            # Merge output data
            if "data_preview" in event.payload:
                handle = event.payload.get("handle", "output")
                summary.outputs[handle] = event.payload["data_preview"]
    
    def get_summary(self) -> GraphExecutionSummary:
        """
        Get the collected execution summary.
        
        Returns:
            GraphExecutionSummary with all collected data
        """
        # Calculate total duration
        total_duration_ms = None
        if self._start_time and self._end_time:
            total_duration_ms = (
                self._end_time - self._start_time
            ).total_seconds() * 1000
        
        return GraphExecutionSummary(
            execution_id=self._execution_id,
            graph_type=self._graph_type,
            start_time=self._start_time,
            end_time=self._end_time,
            total_duration_ms=total_duration_ms,
            nodes=self._nodes.copy(),
            total_nodes=self._total_nodes or len(self._nodes),
            executed_nodes=self._executed_count,
            bypassed_nodes=self._bypassed_count,
            failed_nodes=self._failed_count,
            edges_processed=self._edges.copy(),
            all_events=self._events.copy(),
        )
    
    def get_node_summary(self, node_id: str) -> Optional[NodeExecutionSummary]:
        """
        Get the summary for a specific node.
        
        Args:
            node_id: The node ID
            
        Returns:
            NodeExecutionSummary or None if not found
        """
        return self._nodes.get(node_id)
    
    def get_events(
        self,
        event_type: Optional[DebugEventType] = None,
        node_id: Optional[str] = None,
        min_severity: Optional[DebugEventSeverity] = None,
    ) -> List[DebugEvent]:
        """
        Get filtered events.
        
        Args:
            event_type: Filter by event type
            node_id: Filter by node ID
            min_severity: Minimum severity to include
            
        Returns:
            Filtered list of events
        """
        result = []
        
        severity_order = list(DebugEventSeverity)
        min_idx = severity_order.index(min_severity) if min_severity else 0
        
        for event in self._events:
            if event_type and event.event_type != event_type:
                continue
            
            if node_id and event.node_id != node_id:
                continue
            
            if min_severity:
                event_idx = severity_order.index(event.severity)
                if event_idx < min_idx:
                    continue
            
            result.append(event)
        
        return result
    
    def get_errors(self) -> List[DebugEvent]:
        """
        Get all error events.
        
        Returns:
            List of error events
        """
        error_types = {
            DebugEventType.NODE_ERROR,
            DebugEventType.VALIDATION_ERROR,
            DebugEventType.ROUTING_ERROR,
            DebugEventType.TIMEOUT_ERROR,
            DebugEventType.INPUT_ERROR,
            DebugEventType.TEMPLATE_ERROR,
            DebugEventType.PARSE_ERROR,
        }
        
        return [e for e in self._events if e.event_type in error_types]
    
    def finalize(self) -> GraphExecutionSummary:
        """
        Finalize collection and get the summary.
        
        Sets the end time if not already set and returns
        the complete summary.
        
        Returns:
            Final GraphExecutionSummary
        """
        if self._end_time is None:
            self._end_time = datetime.now(UTC)
        
        return self.get_summary()
    
    def reset(self) -> None:
        """
        Reset the collector for reuse.
        
        Clears all collected events and summaries.
        """
        self._start_time = None
        self._end_time = None
        self._nodes.clear()
        self._events.clear()
        self._edges.clear()
        self._executed_count = 0
        self._bypassed_count = 0
        self._failed_count = 0


class StreamingCollector(DebugCollector):
    """
    Collector that can emit events while collecting.
    
    Useful for streaming debug output while still building
    a complete summary.
    
    Example:
        async def on_event(event):
            await queue.put(event.to_dict())
        
        collector = StreamingCollector(
            execution_id="abc123",
            graph_type="agent",
            on_event=on_event
        )
        
        await collector.collect_async(event)  # Calls on_event and stores
    """
    
    def __init__(
        self,
        execution_id: str,
        graph_type: str,
        total_nodes: int = 0,
        on_event: Optional[callable] = None,
    ):
        """
        Initialize the streaming collector.
        
        Args:
            execution_id: Unique identifier for this execution
            graph_type: Type of graph being executed
            total_nodes: Total number of nodes in the graph
            on_event: Callback for each event (sync or async)
        """
        super().__init__(execution_id, graph_type, total_nodes)
        self._on_event = on_event
    
    async def collect_async(self, event: DebugEvent) -> None:
        """
        Collect an event and optionally emit it.
        
        Args:
            event: The event to collect
        """
        # Call the event handler if set
        if self._on_event:
            import asyncio
            if asyncio.iscoroutinefunction(self._on_event):
                await self._on_event(event)
            else:
                self._on_event(event)
        
        # Still collect for summary
        self.collect(event)
