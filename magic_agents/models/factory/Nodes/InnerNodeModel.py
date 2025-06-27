from typing import Any

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class InnerNodeModel(BaseNodeModel):
    """
    Node model for nested graph execution.

    magic_flow: graph specification dict for an AgentFlowModel to execute.
    """
    magic_flow: dict[str, Any]