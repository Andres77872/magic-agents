from typing import Optional

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class FetchNodeModel(BaseNodeModel):
    url: str
    method: str = "GET"
    headers: Optional[dict[str, str]] = None
    params: Optional[dict[str, str]] = None
    body: Optional[dict[str, str]] = None
    data: Optional[dict[str, str]] = None
    json: Optional[dict[str, str]] = None
