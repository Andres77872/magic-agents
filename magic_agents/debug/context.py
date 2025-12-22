"""
Debug Context Management.

This module provides a context manager for coordinating debug operations
across an execution lifecycle. It ties together capture, transform,
emit, and collection into a cohesive debugging experience.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

from .events import DebugEvent, DebugEventType, DebugEventSeverity
from .capture import DefaultDebugCapture
from .transform import TransformPipeline, create_default_pipeline
from .emitter import EmitterRegistry, QueueEmitter, DebugEmitter
from .collector import DebugCollector, GraphExecutionSummary

if TYPE_CHECKING:
    from .config import DebugConfig


class DebugContext:
    """
    Context for managing debug operations during graph execution.
    
    This class coordinates:
    - Event capture via hooks
    - Event transformation via pipeline
    - Event emission to registered emitters
    - Event collection for summaries
    
    Example:
        ctx = DebugContext(
            execution_id="abc123",
            graph_type="agent",
            enabled=True
        )
        
        await ctx.start()
        
        # During execution
        event = ctx.capture.on_node_start(...)
        await ctx.emit(event)
        
        # After execution
        summary = await ctx.finish()
    """
    
    def __init__(
        self,
        execution_id: Optional[str] = None,
        graph_type: str = "unknown",
        enabled: bool = True,
        total_nodes: int = 0,
        config: Optional["DebugConfig"] = None,
    ):
        """
        Initialize the debug context.
        
        Args:
            execution_id: Unique ID for this execution (auto-generated if None)
            graph_type: Type of graph being executed
            enabled: Whether debug is enabled
            total_nodes: Total number of nodes in the graph
            config: Optional configuration (uses defaults if None)
        """
        self._execution_id = execution_id or uuid.uuid4().hex
        self._graph_type = graph_type
        self._enabled = enabled
        self._total_nodes = total_nodes
        
        # Load config
        if config is None:
            from .config import default_config
            config = default_config()
        self._config = config
        
        # Create components
        self._capture = DefaultDebugCapture(
            execution_id=self._execution_id,
            enabled=enabled
        )
        
        self._pipeline = create_default_pipeline(
            redact=config.redact_sensitive,
            min_severity=config.min_severity,
            max_length=config.max_payload_length,
        )
        
        self._emitters = EmitterRegistry()
        self._collector = DebugCollector(
            execution_id=self._execution_id,
            graph_type=graph_type,
            total_nodes=total_nodes
        )
        
        # State
        self._started = False
        self._finished = False
        self._start_time: Optional[datetime] = None
    
    @property
    def execution_id(self) -> str:
        """Get the execution ID."""
        return self._execution_id
    
    @property
    def enabled(self) -> bool:
        """Check if debug is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set whether debug is enabled."""
        self._enabled = value
        self._capture.enabled = value
    
    @property
    def capture(self) -> DefaultDebugCapture:
        """Get the capture hooks."""
        return self._capture
    
    @property
    def pipeline(self) -> TransformPipeline:
        """Get the transform pipeline."""
        return self._pipeline
    
    @property
    def emitters(self) -> EmitterRegistry:
        """Get the emitter registry."""
        return self._emitters
    
    @property
    def collector(self) -> DebugCollector:
        """Get the event collector."""
        return self._collector
    
    def add_emitter(self, emitter: DebugEmitter) -> "DebugContext":
        """
        Add an emitter.
        
        Args:
            emitter: Emitter to add
            
        Returns:
            Self for chaining
        """
        self._emitters.register(emitter)
        return self
    
    def add_queue_emitter(
        self,
        queue: asyncio.Queue,
        use_legacy_format: bool = True
    ) -> "DebugContext":
        """
        Add a queue emitter.
        
        Args:
            queue: Queue to emit to
            use_legacy_format: Use NodeDebugInfo-compatible format
            
        Returns:
            Self for chaining
        """
        emitter = QueueEmitter(queue, use_legacy_format=use_legacy_format)
        return self.add_emitter(emitter)
    
    async def start(self) -> DebugEvent:
        """
        Start the debug context.
        
        Emits a GRAPH_START event.
        
        Returns:
            The graph start event
        """
        if self._started:
            raise RuntimeError("Context already started")
        
        self._started = True
        self._start_time = datetime.now(UTC)
        
        event = self._capture.on_graph_start(
            graph_type=self._graph_type,
            total_nodes=self._total_nodes
        )
        
        await self.emit(event)
        return event
    
    async def finish(self) -> GraphExecutionSummary:
        """
        Finish the debug context.
        
        Emits a GRAPH_END event and returns the execution summary.
        
        Returns:
            The graph execution summary
        """
        if self._finished:
            raise RuntimeError("Context already finished")
        
        self._finished = True
        
        summary = self._collector.get_summary()
        
        event = self._capture.on_graph_end(
            total_nodes=summary.total_nodes,
            executed_nodes=summary.executed_nodes,
            bypassed_nodes=summary.bypassed_nodes,
            failed_nodes=summary.failed_nodes,
            start_time=self._start_time
        )
        
        await self.emit(event)
        await self._emitters.flush_all()
        
        return self._collector.finalize()
    
    async def emit(self, event: DebugEvent) -> None:
        """
        Process and emit an event.
        
        The event passes through the transform pipeline, gets collected,
        and is emitted to all registered emitters.
        
        Args:
            event: The event to emit
        """
        if not self._enabled:
            return
        
        # Transform
        processed = self._pipeline.process(event)
        if processed is None:
            return
        
        # Collect
        self._collector.collect(processed)
        
        # Emit
        await self._emitters.emit(processed)
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        """
        Process and emit multiple events.
        
        Args:
            events: The events to emit
        """
        if not self._enabled:
            return
        
        # Transform
        processed = self._pipeline.process_batch(events)
        if not processed:
            return
        
        # Collect
        for event in processed:
            self._collector.collect(event)
        
        # Emit
        await self._emitters.emit_batch(processed)
    
    # Convenience methods for common events
    
    async def node_start(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        inputs: Dict[str, Any],
    ) -> DebugEvent:
        """
        Capture and emit a node start event.
        
        Returns:
            The node start event
        """
        event = self._capture.on_node_start(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            inputs=inputs
        )
        await self.emit(event)
        return event
    
    async def node_end(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        outputs: Dict[str, Any],
        internal_state: Dict[str, Any],
        duration_ms: float,
        start_time: datetime,
    ) -> DebugEvent:
        """
        Capture and emit a node end event.
        
        Returns:
            The node end event
        """
        event = self._capture.on_node_end(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            outputs=outputs,
            internal_state=internal_state,
            duration_ms=duration_ms,
            start_time=start_time
        )
        await self.emit(event)
        return event
    
    async def node_error(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        error: Exception,
        context: Dict[str, Any] = None,
    ) -> DebugEvent:
        """
        Capture and emit a node error event.
        
        Returns:
            The node error event
        """
        event = self._capture.on_node_error(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            error=error,
            context=context or {}
        )
        await self.emit(event)
        return event
    
    async def node_bypass(
        self,
        node_id: str,
        node_type: str,
        node_class: str,
        reason: str,
        bypass_source: str,
        inputs_at_bypass: Dict[str, Any] = None,
    ) -> DebugEvent:
        """
        Capture and emit a node bypass event.
        
        Returns:
            The node bypass event
        """
        event = self._capture.on_node_bypass(
            node_id=node_id,
            node_type=node_type,
            node_class=node_class,
            reason=reason,
            bypass_source=bypass_source,
            inputs_at_bypass=inputs_at_bypass or {}
        )
        await self.emit(event)
        return event
    
    async def close(self) -> None:
        """
        Close the context and release resources.
        
        Called automatically when using as context manager.
        """
        await self._emitters.close_all()


@asynccontextmanager
async def debug_context(
    graph_type: str = "unknown",
    enabled: bool = True,
    total_nodes: int = 0,
    output_queue: Optional[asyncio.Queue] = None,
    use_legacy_format: bool = True,
    config: Optional["DebugConfig"] = None,
) -> AsyncGenerator[DebugContext, None]:
    """
    Context manager for debug operations.
    
    Handles setup and teardown of debug context, including:
    - Generating execution ID
    - Setting up emitters
    - Emitting start/end events
    - Cleaning up resources
    
    Example:
        queue = asyncio.Queue()
        
        async with debug_context(
            graph_type="agent",
            output_queue=queue
        ) as ctx:
            # Use ctx.capture, ctx.emit, etc.
            event = ctx.capture.on_node_start(...)
            await ctx.emit(event)
        
        # Summary is available after the context exits
    
    Args:
        graph_type: Type of graph being executed
        enabled: Whether debug is enabled
        total_nodes: Total number of nodes
        output_queue: Optional queue for streaming events
        use_legacy_format: Use NodeDebugInfo-compatible format
        config: Optional debug configuration
        
    Yields:
        The debug context
    """
    ctx = DebugContext(
        graph_type=graph_type,
        enabled=enabled,
        total_nodes=total_nodes,
        config=config
    )
    
    # Add queue emitter if provided
    if output_queue:
        ctx.add_queue_emitter(output_queue, use_legacy_format=use_legacy_format)
    
    try:
        if enabled:
            await ctx.start()
        yield ctx
    finally:
        if enabled and ctx._started and not ctx._finished:
            try:
                await ctx.finish()
            except Exception:
                pass  # Don't let cleanup errors propagate
        
        await ctx.close()


class NoopDebugContext:
    """
    A no-op debug context for when debug is disabled.
    
    All methods are no-ops, making it safe to use without
    checking if debug is enabled.
    """
    
    @property
    def execution_id(self) -> str:
        return ""
    
    @property
    def enabled(self) -> bool:
        return False
    
    async def start(self) -> None:
        pass
    
    async def finish(self) -> Optional[GraphExecutionSummary]:
        return None
    
    async def emit(self, event: DebugEvent) -> None:
        pass
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        pass
    
    async def node_start(self, *args, **kwargs) -> None:
        pass
    
    async def node_end(self, *args, **kwargs) -> None:
        pass
    
    async def node_error(self, *args, **kwargs) -> None:
        pass
    
    async def node_bypass(self, *args, **kwargs) -> None:
        pass
    
    async def close(self) -> None:
        pass


def create_debug_context(
    enabled: bool = True,
    **kwargs
) -> DebugContext:
    """
    Factory function to create the appropriate debug context.
    
    Returns a NoopDebugContext when disabled for efficiency.
    
    Args:
        enabled: Whether debug is enabled
        **kwargs: Arguments passed to DebugContext
        
    Returns:
        DebugContext or NoopDebugContext
    """
    if not enabled:
        return NoopDebugContext()
    
    return DebugContext(enabled=enabled, **kwargs)
