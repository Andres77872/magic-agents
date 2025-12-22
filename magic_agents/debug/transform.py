"""
Debug Event Transformation Pipeline.

This module provides transformers that can filter, redact, truncate,
or otherwise modify debug events before they are emitted.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from .events import DebugEvent, DebugEventType, DebugEventSeverity


@runtime_checkable
class DebugTransformer(Protocol):
    """
    Protocol for debug event transformers.
    
    Transformers can modify, filter, or enrich debug events before
    they are emitted. Return None to filter out an event.
    """
    
    @property
    def name(self) -> str:
        """Unique name for this transformer."""
        ...
    
    @property
    def order(self) -> int:
        """
        Order in which this transformer runs.
        
        Lower numbers run first. Typical ranges:
        - 0-9: Pre-processing (validation, normalization)
        - 10-19: Redaction and security
        - 20-29: Filtering
        - 30-39: Truncation and size limits
        - 40-49: Enrichment
        - 50+: Post-processing
        """
        ...
    
    def transform(self, event: DebugEvent) -> Optional[DebugEvent]:
        """
        Transform a debug event.
        
        Args:
            event: The event to transform
            
        Returns:
            Transformed event, or None to filter it out
        """
        ...


class TransformPipeline:
    """
    Chain of transformers applied in order.
    
    Events pass through each transformer in sequence. If any transformer
    returns None, the event is filtered out and no subsequent transformers
    are called.
    
    Example:
        pipeline = TransformPipeline()
        pipeline.add(RedactTransformer())
        pipeline.add(FilterTransformer(min_severity=DebugEventSeverity.INFO))
        pipeline.add(TruncateTransformer(max_length=500))
        
        event = pipeline.process(raw_event)
        if event:
            # Event passed all transformers
            emit(event)
    """
    
    def __init__(self):
        """Initialize an empty pipeline."""
        self._transformers: List[DebugTransformer] = []
    
    def add(self, transformer: DebugTransformer) -> "TransformPipeline":
        """
        Add a transformer to the pipeline.
        
        Transformers are automatically sorted by their order property.
        
        Args:
            transformer: Transformer to add
            
        Returns:
            Self for chaining
        """
        self._transformers.append(transformer)
        self._transformers.sort(key=lambda t: t.order)
        return self
    
    def remove(self, name: str) -> "TransformPipeline":
        """
        Remove a transformer by name.
        
        Args:
            name: Name of the transformer to remove
            
        Returns:
            Self for chaining
        """
        self._transformers = [t for t in self._transformers if t.name != name]
        return self
    
    def get(self, name: str) -> Optional[DebugTransformer]:
        """
        Get a transformer by name.
        
        Args:
            name: Name of the transformer
            
        Returns:
            The transformer, or None if not found
        """
        for t in self._transformers:
            if t.name == name:
                return t
        return None
    
    def clear(self) -> "TransformPipeline":
        """
        Remove all transformers.
        
        Returns:
            Self for chaining
        """
        self._transformers.clear()
        return self
    
    def process(self, event: DebugEvent) -> Optional[DebugEvent]:
        """
        Process an event through all transformers.
        
        Args:
            event: Event to process
            
        Returns:
            Transformed event, or None if filtered out
        """
        current = event
        for transformer in self._transformers:
            current = transformer.transform(current)
            if current is None:
                return None
        return current
    
    def process_batch(self, events: List[DebugEvent]) -> List[DebugEvent]:
        """
        Process multiple events through the pipeline.
        
        Filtered events are not included in the result.
        
        Args:
            events: Events to process
            
        Returns:
            List of events that passed all transformers
        """
        result = []
        for event in events:
            processed = self.process(event)
            if processed is not None:
                result.append(processed)
        return result
    
    @property
    def transformers(self) -> List[DebugTransformer]:
        """Get the list of transformers in order."""
        return self._transformers.copy()


class RedactTransformer:
    """
    Redact sensitive data from events.
    
    Replaces values of keys matching sensitive patterns with
    a redaction marker. This helps prevent accidental exposure
    of secrets in debug output.
    
    Example:
        transformer = RedactTransformer()
        event.payload = {"api_key": "secret123", "data": "normal"}
        transformed = transformer.transform(event)
        # payload is now {"api_key": "***REDACTED***", "data": "normal"}
    """
    
    SENSITIVE_KEYS = {
        "api_key", "apikey", "api-key",
        "private_key", "privatekey", "private-key",
        "authorization", "auth",
        "password", "passwd", "pwd",
        "token", "access_token", "refresh_token",
        "bearer", "secret", "credential", "credentials",
        "client_secret", "client_id",
        "aws_access_key", "aws_secret_key",
        "openai_api_key", "anthropic_api_key",
    }
    
    name = "redact"
    order = 10
    
    def __init__(
        self,
        additional_keys: Optional[Set[str]] = None,
        redaction_marker: str = "***REDACTED***",
    ):
        """
        Initialize the redactor.
        
        Args:
            additional_keys: Additional key patterns to redact
            redaction_marker: String to replace sensitive values with
        """
        self._sensitive_keys = self.SENSITIVE_KEYS.copy()
        if additional_keys:
            self._sensitive_keys.update(k.lower() for k in additional_keys)
        self._redaction_marker = redaction_marker
    
    def transform(self, event: DebugEvent) -> DebugEvent:
        """Redact sensitive data from the event payload."""
        # Create a copy to avoid modifying the original
        new_event = copy.copy(event)
        new_event.payload = self._redact_dict(event.payload)
        return new_event
    
    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact sensitive keys in a dictionary."""
        result = {}
        for key, value in data.items():
            if self._is_sensitive_key(key):
                result[key] = self._redaction_marker
            elif isinstance(value, dict):
                result[key] = self._redact_dict(value)
            elif isinstance(value, list):
                result[key] = self._redact_list(value)
            else:
                result[key] = value
        return result
    
    def _redact_list(self, data: List[Any]) -> List[Any]:
        """Recursively redact sensitive keys in a list."""
        result = []
        for item in data:
            if isinstance(item, dict):
                result.append(self._redact_dict(item))
            elif isinstance(item, list):
                result.append(self._redact_list(item))
            else:
                result.append(item)
        return result
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key matches any sensitive pattern."""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self._sensitive_keys)


class FilterTransformer:
    """
    Filter events based on criteria.
    
    Events can be filtered by:
    - Event type (include or exclude specific types)
    - Severity level (minimum severity)
    - Node ID (include or exclude specific nodes)
    
    Example:
        # Only include errors and warnings
        transformer = FilterTransformer(
            min_severity=DebugEventSeverity.WARN
        )
        
        # Exclude debug and trace events
        transformer = FilterTransformer(
            exclude_types={DebugEventType.STATE_CHANGE}
        )
    """
    
    name = "filter"
    order = 20
    
    def __init__(
        self,
        include_types: Optional[Set[DebugEventType]] = None,
        exclude_types: Optional[Set[DebugEventType]] = None,
        min_severity: DebugEventSeverity = DebugEventSeverity.DEBUG,
        include_nodes: Optional[Set[str]] = None,
        exclude_nodes: Optional[Set[str]] = None,
    ):
        """
        Initialize the filter.
        
        Args:
            include_types: If set, only include these event types
            exclude_types: Exclude these event types
            min_severity: Minimum severity level to include
            include_nodes: If set, only include events from these nodes
            exclude_nodes: Exclude events from these nodes
        """
        self.include_types = include_types
        self.exclude_types = exclude_types or set()
        self.min_severity = min_severity
        self.include_nodes = include_nodes
        self.exclude_nodes = exclude_nodes or set()
    
    def transform(self, event: DebugEvent) -> Optional[DebugEvent]:
        """Filter events based on configured criteria."""
        # Check event type inclusion
        if self.include_types and event.event_type not in self.include_types:
            return None
        
        # Check event type exclusion
        if event.event_type in self.exclude_types:
            return None
        
        # Check minimum severity
        severity_order = list(DebugEventSeverity)
        event_severity_idx = severity_order.index(event.severity)
        min_severity_idx = severity_order.index(self.min_severity)
        if event_severity_idx < min_severity_idx:
            return None
        
        # Check node filters (only if event has a node_id)
        if event.node_id:
            if self.include_nodes and event.node_id not in self.include_nodes:
                return None
            if event.node_id in self.exclude_nodes:
                return None
        
        return event


class TruncateTransformer:
    """
    Truncate large values in event payloads.
    
    This helps prevent overly large debug output while preserving
    enough information to be useful. Truncated strings get a suffix
    appended to indicate truncation.
    
    Example:
        transformer = TruncateTransformer(max_length=100)
        # Long strings in payload are truncated to 100 chars + "..."
    """
    
    name = "truncate"
    order = 30
    
    def __init__(
        self,
        max_length: int = 1000,
        suffix: str = "...[truncated]",
        max_list_items: int = 20,
    ):
        """
        Initialize the truncator.
        
        Args:
            max_length: Maximum length for string values
            suffix: Suffix to append to truncated strings
            max_list_items: Maximum number of items in lists
        """
        self.max_length = max_length
        self.suffix = suffix
        self.max_list_items = max_list_items
    
    def transform(self, event: DebugEvent) -> DebugEvent:
        """Truncate large values in the event payload."""
        new_event = copy.copy(event)
        new_event.payload = self._truncate_dict(event.payload)
        return new_event
    
    def _truncate_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively truncate values in a dictionary."""
        result = {}
        for key, value in data.items():
            result[key] = self._truncate_value(value)
        return result
    
    def _truncate_list(self, data: List[Any]) -> List[Any]:
        """Truncate a list, limiting items and recursively truncating values."""
        truncated = data[:self.max_list_items]
        result = [self._truncate_value(item) for item in truncated]
        
        if len(data) > self.max_list_items:
            result.append(f"...[{len(data) - self.max_list_items} more items]")
        
        return result
    
    def _truncate_value(self, value: Any) -> Any:
        """Truncate a single value."""
        if isinstance(value, str) and len(value) > self.max_length:
            return value[:self.max_length] + self.suffix
        elif isinstance(value, dict):
            return self._truncate_dict(value)
        elif isinstance(value, list):
            return self._truncate_list(value)
        else:
            return value


class EnrichTransformer:
    """
    Enrich events with additional context.
    
    Can add computed fields, environment info, or other
    contextual data to events.
    
    Example:
        transformer = EnrichTransformer(
            static_fields={"environment": "production"},
            computed_fields={"timestamp_unix": lambda e: e.timestamp.timestamp()}
        )
    """
    
    name = "enrich"
    order = 40
    
    def __init__(
        self,
        static_fields: Optional[Dict[str, Any]] = None,
        computed_fields: Optional[Dict[str, callable]] = None,
        add_tags: Optional[List[str]] = None,
    ):
        """
        Initialize the enricher.
        
        Args:
            static_fields: Fields to add to all event payloads
            computed_fields: Functions that compute field values from events
            add_tags: Tags to add to all events
        """
        self.static_fields = static_fields or {}
        self.computed_fields = computed_fields or {}
        self.add_tags = add_tags or []
    
    def transform(self, event: DebugEvent) -> DebugEvent:
        """Enrich the event with additional fields."""
        new_event = copy.copy(event)
        
        # Add static fields
        new_payload = {**event.payload, **self.static_fields}
        
        # Add computed fields
        for key, func in self.computed_fields.items():
            try:
                new_payload[key] = func(event)
            except Exception:
                pass  # Skip failed computations
        
        new_event.payload = new_payload
        
        # Add tags
        if self.add_tags:
            new_event.tags = list(set(event.tags + self.add_tags))
        
        return new_event


class TagFilterTransformer:
    """
    Filter events based on tags.
    
    Example:
        # Only include events tagged with 'performance'
        transformer = TagFilterTransformer(include_tags={"performance"})
    """
    
    name = "tag_filter"
    order = 25
    
    def __init__(
        self,
        include_tags: Optional[Set[str]] = None,
        exclude_tags: Optional[Set[str]] = None,
        require_all_include_tags: bool = False,
    ):
        """
        Initialize the tag filter.
        
        Args:
            include_tags: If set, only include events with these tags
            exclude_tags: Exclude events with any of these tags
            require_all_include_tags: If True, event must have all include_tags
        """
        self.include_tags = include_tags
        self.exclude_tags = exclude_tags or set()
        self.require_all_include_tags = require_all_include_tags
    
    def transform(self, event: DebugEvent) -> Optional[DebugEvent]:
        """Filter events based on tags."""
        event_tags = set(event.tags)
        
        # Check exclusions first
        if self.exclude_tags and event_tags & self.exclude_tags:
            return None
        
        # Check inclusions
        if self.include_tags:
            if self.require_all_include_tags:
                if not self.include_tags <= event_tags:
                    return None
            else:
                if not (self.include_tags & event_tags):
                    return None
        
        return event


class SamplingTransformer:
    """
    Sample events to reduce volume.
    
    Useful for high-volume debug output where seeing every event
    isn't necessary. Error events are never sampled (always included).
    
    Example:
        # Include only 10% of non-error events
        transformer = SamplingTransformer(sample_rate=0.1)
    """
    
    name = "sampling"
    order = 15
    
    def __init__(
        self,
        sample_rate: float = 1.0,
        never_sample_types: Optional[Set[DebugEventType]] = None,
    ):
        """
        Initialize the sampler.
        
        Args:
            sample_rate: Fraction of events to include (0.0 to 1.0)
            never_sample_types: Event types to always include
        """
        import random
        self._random = random
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.never_sample_types = never_sample_types or {
            DebugEventType.NODE_ERROR,
            DebugEventType.VALIDATION_ERROR,
            DebugEventType.ROUTING_ERROR,
            DebugEventType.TIMEOUT_ERROR,
            DebugEventType.INPUT_ERROR,
            DebugEventType.TEMPLATE_ERROR,
            DebugEventType.PARSE_ERROR,
        }
    
    def transform(self, event: DebugEvent) -> Optional[DebugEvent]:
        """Sample events based on configured rate."""
        # Never sample certain event types
        if event.event_type in self.never_sample_types:
            return event
        
        # Sample based on rate
        if self._random.random() <= self.sample_rate:
            return event
        
        return None


def create_default_pipeline(
    redact: bool = True,
    min_severity: DebugEventSeverity = DebugEventSeverity.DEBUG,
    max_length: int = 1000,
) -> TransformPipeline:
    """
    Create a pipeline with common defaults.
    
    Args:
        redact: Whether to redact sensitive data
        min_severity: Minimum severity to include
        max_length: Maximum string length before truncation
        
    Returns:
        Configured TransformPipeline
    """
    pipeline = TransformPipeline()
    
    if redact:
        pipeline.add(RedactTransformer())
    
    pipeline.add(FilterTransformer(min_severity=min_severity))
    pipeline.add(TruncateTransformer(max_length=max_length))
    
    return pipeline
