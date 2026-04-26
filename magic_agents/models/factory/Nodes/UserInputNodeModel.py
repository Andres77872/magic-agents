from typing import Optional, Any

from pydantic import model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class UserInputNodeModel(BaseNodeModel):
    """
    UserInput node model - accepts various field names from JSON.
    The JSON definition is the source of truth.
    
    Session management aligned with backend thread persistence contract.
    """
    template: Optional[str] = None
    text: Optional[str] = None
    content: Optional[str] = None
    message: Optional[str] = None
    files: Optional[list[Any] | Any] = None
    images: Optional[list[Any] | Any] = None
    extras: Optional[dict[str, Any]] = None  # Client-provided contextual data
    
    # Session configuration for thread persistence
    session_id: Optional[str] = None  # External thread/conversation ID (reuse from backend)
    session_required: bool = False  # If True, enforce session presence (auto-create fallback)

    @model_validator(mode='after')
    def resolve_text_content(self):
        """Resolve text from 'text', 'content', or 'message' field (JSON-first approach)."""
        if self.text is None:
            if self.content is not None:
                self.text = self.content
            elif self.message is not None:
                self.text = self.message
        return self
