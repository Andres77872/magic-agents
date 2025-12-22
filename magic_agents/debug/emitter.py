"""
Debug Event Emission Layer.

This module provides emitters for delivering debug events to various
destinations: async queues, logging, callbacks, and more.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .events import DebugEvent, DebugEventSeverity


@runtime_checkable
class DebugEmitter(Protocol):
    """
    Protocol for debug event emitters.
    
    Emitters are responsible for delivering debug events to their
    final destination (queue, log, callback, etc.).
    """
    
    @property
    def name(self) -> str:
        """Unique name for this emitter."""
        ...
    
    async def emit(self, event: DebugEvent) -> None:
        """
        Emit a single debug event.
        
        Args:
            event: The event to emit
        """
        ...
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        """
        Emit multiple debug events.
        
        Args:
            events: The events to emit
        """
        ...
    
    async def flush(self) -> None:
        """Flush any buffered events."""
        ...
    
    async def close(self) -> None:
        """Close the emitter and release resources."""
        ...


class EmitterRegistry:
    """
    Registry for managing multiple emitters.
    
    Events are emitted to all registered emitters in parallel.
    Errors in individual emitters don't affect others.
    
    Example:
        registry = EmitterRegistry()
        registry.register(QueueEmitter(output_queue))
        registry.register(LogEmitter())
        
        await registry.emit(event)  # Emits to both
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._emitters: Dict[str, DebugEmitter] = {}
    
    def register(self, emitter: DebugEmitter) -> "EmitterRegistry":
        """
        Register an emitter.
        
        Args:
            emitter: Emitter to register
            
        Returns:
            Self for chaining
        """
        self._emitters[emitter.name] = emitter
        return self
    
    def unregister(self, name: str) -> "EmitterRegistry":
        """
        Unregister an emitter by name.
        
        Args:
            name: Name of the emitter to remove
            
        Returns:
            Self for chaining
        """
        self._emitters.pop(name, None)
        return self
    
    def get(self, name: str) -> Optional[DebugEmitter]:
        """
        Get an emitter by name.
        
        Args:
            name: Name of the emitter
            
        Returns:
            The emitter, or None if not found
        """
        return self._emitters.get(name)
    
    @property
    def emitters(self) -> List[DebugEmitter]:
        """Get all registered emitters."""
        return list(self._emitters.values())
    
    async def emit(self, event: DebugEvent) -> None:
        """
        Emit an event to all registered emitters.
        
        Emitters are called in parallel. Exceptions are caught
        and logged, but don't prevent other emitters from receiving
        the event.
        
        Args:
            event: Event to emit
        """
        if not self._emitters:
            return
        
        results = await asyncio.gather(
            *(emitter.emit(event) for emitter in self._emitters.values()),
            return_exceptions=True
        )
        
        # Log any errors
        for emitter, result in zip(self._emitters.values(), results):
            if isinstance(result, Exception):
                logging.getLogger(__name__).warning(
                    "Emitter %s failed: %s", emitter.name, result
                )
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        """
        Emit multiple events to all registered emitters.
        
        Args:
            events: Events to emit
        """
        if not self._emitters or not events:
            return
        
        await asyncio.gather(
            *(emitter.emit_batch(events) for emitter in self._emitters.values()),
            return_exceptions=True
        )
    
    async def flush_all(self) -> None:
        """Flush all emitters."""
        await asyncio.gather(
            *(emitter.flush() for emitter in self._emitters.values()),
            return_exceptions=True
        )
    
    async def close_all(self) -> None:
        """Close all emitters."""
        await asyncio.gather(
            *(emitter.close() for emitter in self._emitters.values()),
            return_exceptions=True
        )


class QueueEmitter:
    """
    Emit events to an async queue.
    
    This is the primary emitter for streaming debug output.
    Events are put on a queue that can be consumed by the caller.
    
    Example:
        queue = asyncio.Queue()
        emitter = QueueEmitter(queue)
        await emitter.emit(event)
        
        # Consumer
        item = await queue.get()  # {"type": "debug", "content": {...}}
    """
    
    name = "queue"
    
    def __init__(
        self,
        queue: asyncio.Queue,
        use_legacy_format: bool = True,
        include_event_type: bool = True,
    ):
        """
        Initialize the queue emitter.
        
        Args:
            queue: The async queue to emit to
            use_legacy_format: Use NodeDebugInfo-compatible format
            include_event_type: Include event_type in the message
        """
        self._queue = queue
        self._use_legacy = use_legacy_format
        self._include_event_type = include_event_type
        self._closed = False
    
    async def emit(self, event: DebugEvent) -> None:
        """Put an event on the queue."""
        if self._closed:
            return
        
        if self._use_legacy:
            content = event.to_legacy_format()
        else:
            content = event.to_dict()
        
        message = {"type": "debug", "content": content}
        
        if self._include_event_type:
            message["event_type"] = event.event_type.value
        
        await self._queue.put(message)
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        """Put multiple events on the queue."""
        for event in events:
            await self.emit(event)
    
    async def flush(self) -> None:
        """No-op for queue emitter."""
        pass
    
    async def close(self) -> None:
        """Mark the emitter as closed."""
        self._closed = True


class LogEmitter:
    """
    Emit events to the logging system.
    
    Events are logged at levels corresponding to their severity.
    
    Example:
        emitter = LogEmitter(logger_name="myapp.debug")
        await emitter.emit(event)  # Logs to myapp.debug
    """
    
    name = "log"
    
    SEVERITY_TO_LEVEL = {
        DebugEventSeverity.TRACE: logging.DEBUG,
        DebugEventSeverity.DEBUG: logging.DEBUG,
        DebugEventSeverity.INFO: logging.INFO,
        DebugEventSeverity.WARN: logging.WARNING,
        DebugEventSeverity.ERROR: logging.ERROR,
    }
    
    def __init__(
        self,
        logger_name: str = "magic_agents.debug",
        format_json: bool = False,
    ):
        """
        Initialize the log emitter.
        
        Args:
            logger_name: Name of the logger to use
            format_json: If True, log events as JSON
        """
        self._logger = logging.getLogger(logger_name)
        self._format_json = format_json
        self._closed = False
    
    async def emit(self, event: DebugEvent) -> None:
        """Log an event."""
        if self._closed:
            return
        
        level = self.SEVERITY_TO_LEVEL.get(event.severity, logging.INFO)
        
        if self._format_json:
            message = json.dumps(event.to_dict())
        else:
            message = self._format_event(event)
        
        self._logger.log(level, message)
    
    def _format_event(self, event: DebugEvent) -> str:
        """Format an event for human-readable logging."""
        parts = [f"[{event.event_type.value}]"]
        
        if event.node_id:
            parts.append(f"node={event.node_id}")
        
        if event.node_type:
            parts.append(f"type={event.node_type}")
        
        # Add key payload info
        if "duration_ms" in event.payload:
            parts.append(f"duration={event.payload['duration_ms']:.2f}ms")
        
        if "error_message" in event.payload:
            parts.append(f"error={event.payload['error_message']}")
        
        return " ".join(parts)
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        """Log multiple events."""
        for event in events:
            await self.emit(event)
    
    async def flush(self) -> None:
        """Flush the logging handlers."""
        for handler in self._logger.handlers:
            handler.flush()
    
    async def close(self) -> None:
        """Mark the emitter as closed."""
        self._closed = True


class CallbackEmitter:
    """
    Emit events to registered callbacks.
    
    Useful for custom handling of debug events, like sending
    to external systems or updating UI.
    
    Example:
        async def my_handler(event):
            print(f"Event: {event.event_type}")
        
        emitter = CallbackEmitter()
        emitter.add_callback(my_handler)
        await emitter.emit(event)  # Calls my_handler
    """
    
    name = "callback"
    
    def __init__(self):
        """Initialize the callback emitter."""
        self._callbacks: List[Callable[[DebugEvent], Awaitable[None]]] = []
        self._sync_callbacks: List[Callable[[DebugEvent], None]] = []
        self._closed = False
    
    def add_callback(
        self,
        callback: Callable[[DebugEvent], Awaitable[None]]
    ) -> "CallbackEmitter":
        """
        Add an async callback.
        
        Args:
            callback: Async function to call with each event
            
        Returns:
            Self for chaining
        """
        self._callbacks.append(callback)
        return self
    
    def add_sync_callback(
        self,
        callback: Callable[[DebugEvent], None]
    ) -> "CallbackEmitter":
        """
        Add a synchronous callback.
        
        Args:
            callback: Sync function to call with each event
            
        Returns:
            Self for chaining
        """
        self._sync_callbacks.append(callback)
        return self
    
    def remove_callback(
        self,
        callback: Callable
    ) -> "CallbackEmitter":
        """
        Remove a callback.
        
        Args:
            callback: The callback to remove
            
        Returns:
            Self for chaining
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
        if callback in self._sync_callbacks:
            self._sync_callbacks.remove(callback)
        return self
    
    async def emit(self, event: DebugEvent) -> None:
        """Emit an event to all callbacks."""
        if self._closed:
            return
        
        # Call sync callbacks
        for callback in self._sync_callbacks:
            try:
                callback(event)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Sync callback failed: %s", e
                )
        
        # Call async callbacks in parallel
        if self._callbacks:
            await asyncio.gather(
                *(self._safe_call(cb, event) for cb in self._callbacks),
                return_exceptions=True
            )
    
    async def _safe_call(
        self,
        callback: Callable[[DebugEvent], Awaitable[None]],
        event: DebugEvent
    ) -> None:
        """Call a callback with error handling."""
        try:
            await callback(event)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Async callback failed: %s", e
            )
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        """Emit multiple events to all callbacks."""
        for event in events:
            await self.emit(event)
    
    async def flush(self) -> None:
        """No-op for callback emitter."""
        pass
    
    async def close(self) -> None:
        """Mark the emitter as closed and clear callbacks."""
        self._closed = True
        self._callbacks.clear()
        self._sync_callbacks.clear()


class BufferedEmitter:
    """
    Buffer events and emit in batches.
    
    Useful for reducing overhead when many events are generated.
    Events are flushed when the buffer is full or flush() is called.
    
    Example:
        inner = QueueEmitter(queue)
        emitter = BufferedEmitter(inner, buffer_size=100)
        await emitter.emit(event)  # Buffered
        await emitter.flush()  # Sends to inner emitter
    """
    
    def __init__(
        self,
        inner: DebugEmitter,
        buffer_size: int = 100,
        flush_interval_seconds: Optional[float] = None,
    ):
        """
        Initialize the buffered emitter.
        
        Args:
            inner: The emitter to send batches to
            buffer_size: Maximum events to buffer before auto-flush
            flush_interval_seconds: Auto-flush interval (None to disable)
        """
        self._inner = inner
        self._buffer: List[DebugEvent] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds
        self._closed = False
        self._flush_task: Optional[asyncio.Task] = None
        
        if flush_interval_seconds:
            self._start_flush_task()
    
    @property
    def name(self) -> str:
        return f"buffered_{self._inner.name}"
    
    def _start_flush_task(self) -> None:
        """Start the periodic flush task."""
        async def flush_periodically():
            while not self._closed:
                await asyncio.sleep(self._flush_interval)
                if self._buffer:
                    await self.flush()
        
        self._flush_task = asyncio.create_task(flush_periodically())
    
    async def emit(self, event: DebugEvent) -> None:
        """Buffer an event."""
        if self._closed:
            return
        
        self._buffer.append(event)
        
        if len(self._buffer) >= self._buffer_size:
            await self.flush()
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        """Buffer multiple events."""
        for event in events:
            await self.emit(event)
    
    async def flush(self) -> None:
        """Flush buffered events to the inner emitter."""
        if not self._buffer:
            return
        
        events = self._buffer
        self._buffer = []
        
        await self._inner.emit_batch(events)
        await self._inner.flush()
    
    async def close(self) -> None:
        """Flush remaining events and close."""
        self._closed = True
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        await self.flush()
        await self._inner.close()


class NullEmitter:
    """
    Emitter that discards all events.
    
    Useful for disabling debug output without changing code.
    """
    
    name = "null"
    
    async def emit(self, event: DebugEvent) -> None:
        """Discard the event."""
        pass
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        """Discard the events."""
        pass
    
    async def flush(self) -> None:
        """No-op."""
        pass
    
    async def close(self) -> None:
        """No-op."""
        pass


class FilteredEmitter:
    """
    Wrapper that filters events before emitting.
    
    Combines an emitter with a transform pipeline for
    per-emitter filtering.
    
    Example:
        from .transform import FilterTransformer, TransformPipeline
        
        pipeline = TransformPipeline()
        pipeline.add(FilterTransformer(min_severity=DebugEventSeverity.WARN))
        
        emitter = FilteredEmitter(LogEmitter(), pipeline)
        # Only warnings and errors go to the log
    """
    
    def __init__(
        self,
        inner: DebugEmitter,
        pipeline: "TransformPipeline",
    ):
        """
        Initialize the filtered emitter.
        
        Args:
            inner: The emitter to send filtered events to
            pipeline: The transform pipeline to filter with
        """
        self._inner = inner
        self._pipeline = pipeline
    
    @property
    def name(self) -> str:
        return f"filtered_{self._inner.name}"
    
    async def emit(self, event: DebugEvent) -> None:
        """Filter and emit an event."""
        processed = self._pipeline.process(event)
        if processed:
            await self._inner.emit(processed)
    
    async def emit_batch(self, events: List[DebugEvent]) -> None:
        """Filter and emit multiple events."""
        processed = self._pipeline.process_batch(events)
        if processed:
            await self._inner.emit_batch(processed)
    
    async def flush(self) -> None:
        """Flush the inner emitter."""
        await self._inner.flush()
    
    async def close(self) -> None:
        """Close the inner emitter."""
        await self._inner.close()
