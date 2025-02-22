from pydantic import BaseModel

from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class AgentFlowModel(BaseModel):
    type: str = "chat"
    debug: bool = False
    nodes: list[BaseNodeModel]
    edges: list[EdgeNodeModel]
