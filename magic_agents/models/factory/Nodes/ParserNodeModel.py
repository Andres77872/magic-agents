from typing import Optional

from pydantic import model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class ParserNodeModel(BaseNodeModel):
    """
    Parser node model - accepts 'text', 'content', or 'template' from JSON.
    The JSON definition is the source of truth.
    """
    text: Optional[str] = None
    content: Optional[str] = None
    template: Optional[str] = None

    @model_validator(mode='after')
    def resolve_text_content(self):
        """Resolve text from 'text', 'content', or 'template' field (JSON-first approach)."""
        if self.text is None:
            if self.content is not None:
                self.text = self.content
            elif self.template is not None:
                self.text = self.template
            else:
                self.text = ""
        return self
