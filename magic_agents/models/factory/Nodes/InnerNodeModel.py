from typing import Any, Optional

from pydantic import model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class InnerNodeModel(BaseNodeModel):
    """
    Node model for nested graph execution.
    The JSON definition is the source of truth.

    magic_flow/flow/graph: graph specification dict for an AgentFlowModel to execute.
    """
    magic_flow: Optional[dict[str, Any]] = None
    flow: Optional[dict[str, Any]] = None  # alias for magic_flow
    graph: Optional[dict[str, Any]] = None  # alias for magic_flow
    subgraph: Optional[dict[str, Any]] = None  # alias for magic_flow

    @model_validator(mode='after')
    def resolve_aliases(self):
        """Resolve fields from alternative names (JSON-first approach)."""
        if self.magic_flow is None:
            if self.flow is not None:
                self.magic_flow = self.flow
            elif self.graph is not None:
                self.magic_flow = self.graph
            elif self.subgraph is not None:
                self.magic_flow = self.subgraph
        return self