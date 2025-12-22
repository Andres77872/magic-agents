"""
Debug System Module

A comprehensive debug event capture, transformation, and emission system
for Magic Agents. This module provides a unified abstraction layer for
all debug-related functionality.

Key Components:
- events: Event type definitions and data structures
- capture: Event capture hooks and interfaces
- transform: Transformation pipeline for filtering/redacting
- emitter: Event emission dispatchers
- collector: Event aggregation for summaries
- context: Debug context manager for execution lifecycle
- config: Debug configuration options
"""

from .events import (
    DebugEvent,
    DebugEventType,
    DebugEventSeverity,
)
from .capture import (
    DebugCaptureHook,
    DefaultDebugCapture,
)
from .transform import (
    DebugTransformer,
    TransformPipeline,
    RedactTransformer,
    FilterTransformer,
    TruncateTransformer,
)
from .emitter import (
    DebugEmitter,
    EmitterRegistry,
    QueueEmitter,
    LogEmitter,
    CallbackEmitter,
)
from .collector import (
    DebugCollector,
    GraphExecutionSummary,
)
from .context import (
    DebugContext,
    debug_context,
)
from .config import (
    DebugConfig,
    default_config,
    get_preset,
    PRESETS,
)

__all__ = [
    # Events
    "DebugEvent",
    "DebugEventType",
    "DebugEventSeverity",
    # Capture
    "DebugCaptureHook",
    "DefaultDebugCapture",
    # Transform
    "DebugTransformer",
    "TransformPipeline",
    "RedactTransformer",
    "FilterTransformer",
    "TruncateTransformer",
    # Emitter
    "DebugEmitter",
    "EmitterRegistry",
    "QueueEmitter",
    "LogEmitter",
    "CallbackEmitter",
    # Collector
    "DebugCollector",
    "GraphExecutionSummary",
    # Context
    "DebugContext",
    "debug_context",
    # Config
    "DebugConfig",
    "default_config",
    "get_preset",
    "PRESETS",
]
