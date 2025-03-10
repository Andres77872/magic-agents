from typing import Optional

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class UserInputNodeModel(BaseNodeModel):
    template: Optional[str] = None
    text: Optional[str] = None
