from typing import Optional

from pydantic import Field

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class LoopNodeModel(BaseNodeModel):
    """
    Node model for loop control. Expects two inputs:
      - 'list': a JSON or Python list of items to iterate over.
      - 'loop': (optional) result from each iteration to aggregate.
    Produces:
      - iteration outputs via handle 'item'.
      - final aggregation via handle 'end'.

    ``max_iterations`` caps how many items are processed (executor default: 100).
    """
    max_iterations: Optional[int] = Field(default=None, ge=1, strict=True) 