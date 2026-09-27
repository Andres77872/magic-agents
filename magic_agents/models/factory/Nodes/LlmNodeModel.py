from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


def normalize_reasoning_effort(value: Any) -> Optional[str]:
    """Effort names belong to providers; only validate type and trim whitespace."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("reasoning_effort must be a string or null")
    return value.strip() or None


class AgentExecutionConfig(BaseModel):
    """Serializable limits for the agent loop, separate from provider kwargs."""

    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)

    max_iterations: int = Field(default=150, ge=1)
    wall_clock_timeout: Optional[float] = Field(default=None, gt=0)
    per_tool_timeout: float = Field(default=30.0, gt=0)
    max_parallel_tools: int = Field(default=8, ge=1)
    max_output_chars: int = Field(default=50000, ge=1)
    deduplicate: bool = False


class LlmNodeModel(BaseNodeModel):
    """
    LLM node model - accepts various field names from JSON.
    The JSON definition is the source of truth.
    """
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    json_output: Optional[bool] = False
    json_mode: Optional[bool] = None  # alias for json_output
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Provider-defined reasoning effort. Empty/null inherits defaults; "
                    "OpenAI suggestions are none, minimal, low, medium, high, xhigh, max."
    )

    _normalize_reasoning_effort = field_validator('reasoning_effort', mode='before')(normalize_reasoning_effort)

    max_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None  # alias for max_tokens
    iterate: Optional[bool] = False  # if true, rerun this LLM node on each Loop iteration
    history_messages: Optional[list[dict[str, Any]]] = None  # Backend-injected history for no-CHAT graph path
    agent_config: Optional[AgentExecutionConfig] = None

    # STM windowing fields (no-CHAT fallback path parity)
    max_messages: Optional[int] = Field(
        default=None, ge=1,
        description="Maximum non-system conversation messages to keep. "
                    "System messages are always preserved. "
                    "Default 30 applied at runtime in no-CHAT fallback path."
    )
    max_input_tokens: Optional[int] = Field(
        default=None, ge=1,
        description="Maximum estimated tokens for the assembled chat. "
                    "No legacy fallback on LlmNodeModel."
    )
    truncation_strategy: Literal['tail', 'token_budget'] = Field(
        default='tail',
        description="Windowing strategy: 'tail' (keep newest N), "
                    "'token_budget' (token-aware selection)."
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name for usage/logging tracking. "
                    "Does NOT affect STM windowing token estimation, "
                    "which uses a hardcoded GPT-5 tokenizer. "
                    "Optional — purely informational."
    )

    @model_validator(mode='after')
    def resolve_aliases(self):
        """Resolve fields from alternative names (JSON-first approach)."""
        if self.json_output is False and self.json_mode is True:
            self.json_output = True
        if self.max_tokens is None and self.max_output_tokens is not None:
            self.max_tokens = self.max_output_tokens
        return self
