from typing import Optional

from pydantic import model_validator

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

    @model_validator(mode='after')
    def resolve_aliases(self):
        """Resolve fields from alternative names (JSON-first approach)."""
        if self.json_output is False and self.json_mode is True:
            self.json_output = True
        if self.max_tokens is None and self.max_output_tokens is not None:
            self.max_tokens = self.max_output_tokens
        return self
