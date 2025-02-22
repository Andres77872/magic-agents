from typing import Optional

from pydantic import BaseModel

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel
from magic_agents.models.factory.Nodes.EndNodeModel import EndNodeModel


class EdgeNodeModel(BaseModel):
    id: str = "chat"
    source: BaseNodeModel
    target: Optional[BaseNodeModel] = EndNodeModel
    sourceHandle: str
    targetHandle: str
