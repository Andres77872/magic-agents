from typing import Optional

from pydantic import model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class TextNodeModel(BaseNodeModel):
    """
    Text node model - accepts 'text' or 'content' from JSON.
    The JSON definition is the source of truth.
    """
    text: Optional[str] = None
    content: Optional[str] = None

    @model_validator(mode='after')
    def resolve_text_content(self):
        """Resolve text from either 'text' or 'content' field (JSON-first approach)."""
        if self.text is None and self.content is not None:
            self.text = self.content
        elif self.text is None and self.content is None:
            self.text = ""
        return self
