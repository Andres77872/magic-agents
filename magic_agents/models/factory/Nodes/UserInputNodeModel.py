from typing import Optional, Any

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class UserInputNodeModel(BaseNodeModel):
    template: Optional[str] = None
    text: Optional[str] = None
    files: Optional[list[Any] | Any] = None
    images: Optional[list[Any] | Any] = None
