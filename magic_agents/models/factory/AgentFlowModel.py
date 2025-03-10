from typing import Any

from pydantic import BaseModel

from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel


class AgentFlowModel(BaseModel):
    type: str = "chat"
    debug: bool = False
    nodes: dict[str, Any]
    edges: list[EdgeNodeModel]
