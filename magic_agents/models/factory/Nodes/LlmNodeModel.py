from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


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
    max_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None  # alias for max_tokens
    iterate: Optional[bool] = False  # if true, rerun this LLM node on each Loop iteration
    history_messages: Optional[list[dict[str, Any]]] = None  # Backend-injected history for no-CHAT graph path

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
