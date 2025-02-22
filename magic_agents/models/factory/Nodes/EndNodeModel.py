from typing import Optional

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class EndNodeModel(BaseNodeModel):
    end: Optional[str] = None
