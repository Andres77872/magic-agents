from typing import Optional

from pydantic import model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class SendMessageNodeModel(BaseNodeModel):
    """
    SendMessage node model - accepts various field names from JSON.
    The JSON definition is the source of truth.
    """
    message: Optional[str] = ''
    content: Optional[str] = None
    json_extras: Optional[str] = ''
    extras: Optional[str] = None

    @model_validator(mode='after')
    def resolve_fields(self):
        """Resolve fields from alternative names (JSON-first approach)."""
        if not self.message and self.content:
            self.message = self.content
        if not self.json_extras and self.extras:
            self.json_extras = self.extras
        return self
