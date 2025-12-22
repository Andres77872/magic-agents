"""
ConditionalNodeModel - Pydantic validation model for NodeConditional configuration.

This model validates conditional node configuration at build time,
including Jinja2 template syntax and output handle declarations.
"""

from typing import Optional, Dict, List, Literal

import jinja2
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ConditionalNodeModel(BaseModel):
    """
    Validation model for NodeConditional configuration.
    
    Validates:
    - Jinja2 template syntax in condition
    - Valid merge strategy values
    - Output handle name validity
    - Default handle configuration
    
    Example usage in JSON:
        {
            "id": "my_conditional",
            "type": "conditional",
            "data": {
                "condition": "{{ 'yes' if approved else 'no' }}",
                "merge_strategy": "flat",
                "output_handles": ["yes", "no"],
                "default_handle": "no"
            }
        }
    """
    model_config = ConfigDict(extra='allow')  # Allow extra fields from JSON
    
    condition: str = Field(
        ...,
        description="Jinja2 template that evaluates to output handle name",
        min_length=1
    )
    
    merge_strategy: Literal["flat", "namespaced"] = Field(
        default="flat",
        description="How to merge multiple inputs: 'flat' or 'namespaced'"
    )
    
    handles: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom handle name mappings. Example: {'input': 'my_input'}"
    )
    
    output_handles: Optional[List[str]] = Field(
        default=None,
        description="Declared output handle names for graph validation. "
                    "When specified, build-time validation ensures edges exist for each handle."
    )
    
    default_handle: Optional[str] = Field(
        default=None,
        description="Fallback handle if condition evaluates to empty/invalid string. "
                    "Must match one of the declared output_handles if specified."
    )

    @field_validator('condition')
    @classmethod
    def validate_jinja2_syntax(cls, v: str) -> str:
        """Pre-validate Jinja2 template syntax at build time."""
        try:
            jinja2.Environment().parse(v)
        except jinja2.TemplateSyntaxError as e:
            raise ValueError(f"Invalid Jinja2 syntax in condition: {e}")
        return v
    
    @field_validator('output_handles')
    @classmethod
    def validate_output_handles(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Ensure output handles are valid identifiers."""
        if v is not None:
            for handle in v:
                if not handle:
                    raise ValueError("Output handle name cannot be empty")
                if handle.startswith('__') and handle.endswith('__'):
                    raise ValueError(
                        f"Invalid output handle name: '{handle}'. "
                        "System handles (with double underscores) are reserved."
                    )
        return v
    
    @field_validator('default_handle')
    @classmethod
    def validate_default_handle(cls, v: Optional[str], info) -> Optional[str]:
        """Validate default_handle is in output_handles if both specified."""
        if v is not None:
            # Check against reserved system handles
            if v.startswith('__') and v.endswith('__'):
                raise ValueError(
                    f"Invalid default_handle: '{v}'. "
                    "System handles (with double underscores) are reserved."
                )
            
            # Check consistency with output_handles if available
            output_handles = info.data.get('output_handles')
            if output_handles is not None and v not in output_handles:
                raise ValueError(
                    f"default_handle '{v}' must be one of the declared "
                    f"output_handles: {output_handles}"
                )
        return v


class ConditionalSignalTypes:
    """
    Standard signal types for conditional routing.
    
    These are system-reserved signals used internally by the conditional
    node and executor for error handling and bypass propagation.
    """
    # Bypass all downstream paths (error case)
    BYPASS_ALL = "__bypass_all__"
    
    # Use default handle for routing
    DEFAULT = "__default__"
    
    # Error signal with details
    ERROR = "__error__"
    
    # Timeout signal
    TIMEOUT = "__timeout__"
    
    @staticmethod
    def is_system_signal(handle: str) -> bool:
        """Check if handle is a system signal (not user-defined)."""
        return handle.startswith("__") and handle.endswith("__")
