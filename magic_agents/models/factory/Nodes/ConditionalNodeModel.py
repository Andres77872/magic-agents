"""
ConditionalNodeModel - Pydantic validation model for NodeConditional configuration.

This model validates conditional node configuration at build time,
including Jinja2 template syntax and output handle declarations.
"""

import json
from urllib.parse import urlsplit
from typing import Annotated, Optional, Dict, List, Literal, Union

import jinja2
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, JsonValue
from magic_agents.util.env_resolver import ENV_PLACEHOLDER_PATTERN


Description = Union[str, Dict[str, JsonValue], List[JsonValue]]


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    instructions: Description

    @field_validator('instructions')
    @classmethod
    def validate_instructions(cls, value):
        if not value or isinstance(value, str) and not value.strip():
            raise ValueError('Question instructions cannot be empty')
        json.dumps(value, allow_nan=False)
        return value


class ChoiceQuestion(QuestionBase):
    type: Literal['choice']
    criteria: Dict[str, Optional[Description]] = Field(min_length=1, max_length=255)

    @field_validator('criteria')
    @classmethod
    def validate_criteria(cls, value):
        if any(not label.strip() for label in value):
            raise ValueError('Choice labels cannot be empty')
        json.dumps(value, allow_nan=False)
        return value


class ScoreQuestion(QuestionBase):
    type: Literal['score']
    criteria: List[Description] = Field(min_length=2, max_length=10)

    @field_validator('criteria')
    @classmethod
    def validate_criteria(cls, value):
        json.dumps(value, allow_nan=False)
        return value


class NoulQuestion(QuestionBase):
    type: Literal['noul']
    criteria: Optional[Dict[Literal['true', 'false'], Optional[Description]]] = None

    @field_validator('criteria')
    @classmethod
    def validate_criteria(cls, value):
        json.dumps(value, allow_nan=False)
        return value


ConditionalQuestion = Annotated[
    Union[ChoiceQuestion, ScoreQuestion, NoulQuestion], Field(discriminator='type')
]


class JevConnection(BaseModel):
    """Only transport settings; Jev provides its own judgment model."""
    model_config = ConfigDict(extra='forbid', strict=True)
    base_url: str = 'https://api.typesafe.ai/v1'
    api_key: Optional[str] = Field(default=None, repr=False)

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, value):
        value = value.strip().rstrip('/')
        if ENV_PLACEHOLDER_PATTERN.fullmatch(value):
            return value  # Resolve only connection settings immediately before the request.
        parsed = urlsplit(value)
        try:
            parsed.port
        except ValueError:
            raise ValueError('Jev base_url must contain a valid port') from None
        if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or '?' in value or '#' in value or any(char.isspace() for char in value)):
            raise ValueError('Jev base_url must be an HTTP(S) URL without credentials, query or fragment')
        return value


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
    model_config = ConfigDict(extra='forbid')  # Reject unknown fields - strict validation

    evaluation_mode: Literal['jinja', 'llm', 'jev'] = Field(
        default='jinja', description='Exact routing, or Jev-compatible typed judgments followed by deterministic routing'
    )
    questions: Optional[Dict[str, ConditionalQuestion]] = Field(default=None, min_length=1)
    evaluation_timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False, strict=True)
    evaluation_error_policy: Literal['fail', 'default'] = Field(
        default='fail', description='Fail judgment errors, or route original state to the declared default branch'
    )
    jev: Optional[JevConnection] = None
    
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

    @field_validator('questions')
    @classmethod
    def validate_questions(cls, value):
        if value is not None and any(not key.strip() for key in value):
            raise ValueError('Question IDs cannot be empty')
        return value

    @model_validator(mode='after')
    def validate_evaluation_configuration(self):
        if self.questions is not None and 'evaluation_mode' not in self.model_fields_set:
            raise ValueError('Specify evaluation_mode when configuring questions')
        if self.evaluation_mode in {'llm', 'jev'}:
            if not self.questions:
                raise ValueError('Judgment evaluation requires at least one typed question')
            if not self.output_handles:
                raise ValueError('Judgment evaluation requires declared output_handles')
        if self.evaluation_mode in {'llm', 'jev'} and self.handles:
            client_handle = self.handles.get('client_provider', self.handles.get('client', 'handle-client-provider'))
            state_handle = self.handles.get('input', self.handles.get('context', 'handle_input'))
            if state_handle in {client_handle, 'handle-client-provider'}:
                raise ValueError('Client and state input handles must be different')
        if self.evaluation_error_policy == 'default' and (
                not self.default_handle or not self.output_handles or self.default_handle not in self.output_handles):
            raise ValueError('Default evaluation error policy requires a declared default_handle')
        return self

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
            if len(set(v)) != len(v):
                raise ValueError('Output handle names must be unique')
            for handle in v:
                if not handle or not handle.strip():
                    raise ValueError("Output handle name cannot be empty")
                if handle in {'debug', 'debug_summary', 'content', 'end'}:
                    raise ValueError(f"Output handle '{handle}' is reserved for runtime events")
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
