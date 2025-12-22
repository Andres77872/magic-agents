"""
Debug Configuration.

This module provides configuration options for the debug system,
allowing customization of capture, transformation, and emission behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .events import DebugEventType, DebugEventSeverity


@dataclass
class DebugConfig:
    """
    Configuration for the debug system.
    
    This class provides all configuration options for customizing
    debug behavior. Use default_config() for sensible defaults.
    
    Attributes:
        enabled: Master switch for debug functionality
        min_severity: Minimum severity level to capture
        redact_sensitive: Whether to redact sensitive data
        max_payload_length: Maximum length for string values
        max_list_items: Maximum items in lists
        use_legacy_format: Use NodeDebugInfo-compatible format
        include_event_types: If set, only include these event types
        exclude_event_types: Exclude these event types
        include_nodes: If set, only debug these nodes
        exclude_nodes: Don't debug these nodes
        capture_inputs: Whether to capture input data
        capture_outputs: Whether to capture output data
        capture_internal_state: Whether to capture internal state
        emit_to_log: Also emit events to logging
        log_level: Log level for emitted events
        sample_rate: Fraction of events to include (1.0 = all)
        additional_redact_keys: Additional keys to redact
    """
    
    # Master switch
    enabled: bool = True
    
    # Filtering
    min_severity: DebugEventSeverity = DebugEventSeverity.DEBUG
    include_event_types: Optional[Set[DebugEventType]] = None
    exclude_event_types: Set[DebugEventType] = field(default_factory=set)
    include_nodes: Optional[Set[str]] = None
    exclude_nodes: Set[str] = field(default_factory=set)
    
    # Redaction
    redact_sensitive: bool = True
    additional_redact_keys: Set[str] = field(default_factory=set)
    
    # Truncation
    max_payload_length: int = 1000
    max_list_items: int = 20
    
    # Data capture
    capture_inputs: bool = True
    capture_outputs: bool = True
    capture_internal_state: bool = True
    
    # Format
    use_legacy_format: bool = True
    
    # Logging
    emit_to_log: bool = False
    log_level: str = "DEBUG"
    log_format_json: bool = False
    
    # Sampling
    sample_rate: float = 1.0
    
    # Extra tags to add to all events
    default_tags: List[str] = field(default_factory=list)
    
    # Custom metadata to include in all events
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebugConfig":
        """
        Create a DebugConfig from a dictionary (e.g., from JSON).
        
        This allows debug configuration to be specified in the graph JSON:
        
        ```json
        {
          "type": "chat",
          "debug": true,
          "debug_config": {
            "preset": "verbose",
            "redact_sensitive": true,
            "max_payload_length": 2000
          },
          "nodes": [...],
          "edges": [...]
        }
        ```
        
        Args:
            data: Dictionary with configuration values
            
        Returns:
            DebugConfig instance
        """
        if not data:
            return cls()
        
        # Check for preset first
        preset_name = data.pop("preset", None) if isinstance(data, dict) else None
        
        if preset_name:
            # Start from preset, then override with remaining values
            base_config = get_preset(preset_name)
        else:
            base_config = cls()
        
        # Apply remaining overrides
        if isinstance(data, dict):
            # Convert event type strings to enums if present
            if "include_event_types" in data and data["include_event_types"]:
                data["include_event_types"] = {
                    DebugEventType(et) if isinstance(et, str) else et
                    for et in data["include_event_types"]
                }
            if "exclude_event_types" in data and data["exclude_event_types"]:
                data["exclude_event_types"] = {
                    DebugEventType(et) if isinstance(et, str) else et
                    for et in data["exclude_event_types"]
                }
            
            # Convert severity string to enum if present
            if "min_severity" in data and isinstance(data["min_severity"], str):
                data["min_severity"] = DebugEventSeverity(data["min_severity"])
            
            # Convert sets
            if "include_nodes" in data and data["include_nodes"]:
                data["include_nodes"] = set(data["include_nodes"])
            if "exclude_nodes" in data and data["exclude_nodes"]:
                data["exclude_nodes"] = set(data["exclude_nodes"])
            if "additional_redact_keys" in data and data["additional_redact_keys"]:
                data["additional_redact_keys"] = set(data["additional_redact_keys"])
            
            # Create new config with overrides
            return cls(
                enabled=data.get("enabled", base_config.enabled),
                min_severity=data.get("min_severity", base_config.min_severity),
                include_event_types=data.get("include_event_types", base_config.include_event_types),
                exclude_event_types=data.get("exclude_event_types", base_config.exclude_event_types),
                include_nodes=data.get("include_nodes", base_config.include_nodes),
                exclude_nodes=data.get("exclude_nodes", base_config.exclude_nodes),
                redact_sensitive=data.get("redact_sensitive", base_config.redact_sensitive),
                additional_redact_keys=data.get("additional_redact_keys", base_config.additional_redact_keys),
                max_payload_length=data.get("max_payload_length", base_config.max_payload_length),
                max_list_items=data.get("max_list_items", base_config.max_list_items),
                capture_inputs=data.get("capture_inputs", base_config.capture_inputs),
                capture_outputs=data.get("capture_outputs", base_config.capture_outputs),
                capture_internal_state=data.get("capture_internal_state", base_config.capture_internal_state),
                use_legacy_format=data.get("use_legacy_format", base_config.use_legacy_format),
                emit_to_log=data.get("emit_to_log", base_config.emit_to_log),
                log_level=data.get("log_level", base_config.log_level),
                log_format_json=data.get("log_format_json", base_config.log_format_json),
                sample_rate=data.get("sample_rate", base_config.sample_rate),
                default_tags=data.get("default_tags", base_config.default_tags),
                metadata=data.get("metadata", base_config.metadata),
            )
        
        return base_config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DebugConfig to a dictionary (for JSON serialization).
        
        Returns:
            Dictionary representation of the config
        """
        return {
            "enabled": self.enabled,
            "min_severity": self.min_severity.value if self.min_severity else None,
            "include_event_types": [et.value for et in self.include_event_types] if self.include_event_types else None,
            "exclude_event_types": [et.value for et in self.exclude_event_types] if self.exclude_event_types else [],
            "include_nodes": list(self.include_nodes) if self.include_nodes else None,
            "exclude_nodes": list(self.exclude_nodes) if self.exclude_nodes else [],
            "redact_sensitive": self.redact_sensitive,
            "additional_redact_keys": list(self.additional_redact_keys) if self.additional_redact_keys else [],
            "max_payload_length": self.max_payload_length,
            "max_list_items": self.max_list_items,
            "capture_inputs": self.capture_inputs,
            "capture_outputs": self.capture_outputs,
            "capture_internal_state": self.capture_internal_state,
            "use_legacy_format": self.use_legacy_format,
            "emit_to_log": self.emit_to_log,
            "log_level": self.log_level,
            "log_format_json": self.log_format_json,
            "sample_rate": self.sample_rate,
            "default_tags": self.default_tags,
            "metadata": self.metadata,
        }
    
    def with_severity(self, severity: DebugEventSeverity) -> "DebugConfig":
        """
        Create a copy with different minimum severity.
        
        Args:
            severity: New minimum severity
            
        Returns:
            New DebugConfig with updated severity
        """
        return DebugConfig(
            enabled=self.enabled,
            min_severity=severity,
            include_event_types=self.include_event_types,
            exclude_event_types=self.exclude_event_types.copy(),
            include_nodes=self.include_nodes,
            exclude_nodes=self.exclude_nodes.copy(),
            redact_sensitive=self.redact_sensitive,
            additional_redact_keys=self.additional_redact_keys.copy(),
            max_payload_length=self.max_payload_length,
            max_list_items=self.max_list_items,
            capture_inputs=self.capture_inputs,
            capture_outputs=self.capture_outputs,
            capture_internal_state=self.capture_internal_state,
            use_legacy_format=self.use_legacy_format,
            emit_to_log=self.emit_to_log,
            log_level=self.log_level,
            log_format_json=self.log_format_json,
            sample_rate=self.sample_rate,
            default_tags=self.default_tags.copy(),
            metadata=self.metadata.copy(),
        )
    
    def with_nodes(
        self,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None
    ) -> "DebugConfig":
        """
        Create a copy with different node filters.
        
        Args:
            include: Only include these nodes
            exclude: Exclude these nodes
            
        Returns:
            New DebugConfig with updated node filters
        """
        new_config = self.with_severity(self.min_severity)
        if include is not None:
            new_config.include_nodes = include
        if exclude is not None:
            new_config.exclude_nodes = exclude
        return new_config
    
    def with_event_types(
        self,
        include: Optional[Set[DebugEventType]] = None,
        exclude: Optional[Set[DebugEventType]] = None
    ) -> "DebugConfig":
        """
        Create a copy with different event type filters.
        
        Args:
            include: Only include these event types
            exclude: Exclude these event types
            
        Returns:
            New DebugConfig with updated event type filters
        """
        new_config = self.with_severity(self.min_severity)
        if include is not None:
            new_config.include_event_types = include
        if exclude is not None:
            new_config.exclude_event_types = exclude
        return new_config


def default_config() -> DebugConfig:
    """
    Get the default debug configuration.
    
    Returns:
        DebugConfig with sensible defaults
    """
    return DebugConfig()


def minimal_config() -> DebugConfig:
    """
    Get a minimal debug configuration.
    
    Only captures errors and warnings, no input/output data.
    
    Returns:
        DebugConfig for minimal debugging
    """
    return DebugConfig(
        min_severity=DebugEventSeverity.WARN,
        capture_inputs=False,
        capture_outputs=False,
        capture_internal_state=False,
        max_payload_length=200,
    )


def verbose_config() -> DebugConfig:
    """
    Get a verbose debug configuration.
    
    Captures everything including trace events.
    
    Returns:
        DebugConfig for verbose debugging
    """
    return DebugConfig(
        min_severity=DebugEventSeverity.TRACE,
        max_payload_length=5000,
        max_list_items=100,
        emit_to_log=True,
    )


def production_config() -> DebugConfig:
    """
    Get a production-safe debug configuration.
    
    Redacts sensitive data, samples events, only includes
    errors and key lifecycle events.
    
    Returns:
        DebugConfig for production use
    """
    return DebugConfig(
        min_severity=DebugEventSeverity.INFO,
        redact_sensitive=True,
        max_payload_length=500,
        sample_rate=0.1,
        include_event_types={
            DebugEventType.GRAPH_START,
            DebugEventType.GRAPH_END,
            DebugEventType.NODE_ERROR,
            DebugEventType.VALIDATION_ERROR,
            DebugEventType.ROUTING_ERROR,
        },
    )


def errors_only_config() -> DebugConfig:
    """
    Get a configuration that only captures errors.
    
    Returns:
        DebugConfig for error-only debugging
    """
    return DebugConfig(
        min_severity=DebugEventSeverity.ERROR,
        include_event_types={
            DebugEventType.NODE_ERROR,
            DebugEventType.VALIDATION_ERROR,
            DebugEventType.ROUTING_ERROR,
            DebugEventType.TIMEOUT_ERROR,
            DebugEventType.INPUT_ERROR,
            DebugEventType.TEMPLATE_ERROR,
            DebugEventType.PARSE_ERROR,
        },
    )


# Preset configurations
PRESETS = {
    "default": default_config,
    "minimal": minimal_config,
    "verbose": verbose_config,
    "production": production_config,
    "errors_only": errors_only_config,
}


def get_preset(name: str) -> DebugConfig:
    """
    Get a preset configuration by name.
    
    Available presets:
    - default: Standard debugging
    - minimal: Errors and warnings only
    - verbose: Everything including trace
    - production: Production-safe with sampling
    - errors_only: Only error events
    
    Args:
        name: Name of the preset
        
    Returns:
        DebugConfig for the preset
        
    Raises:
        ValueError: If preset name is unknown
    """
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}"
        )
    return PRESETS[name]()
