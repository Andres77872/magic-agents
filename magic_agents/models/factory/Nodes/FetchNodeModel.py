from typing import Optional, Any

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class FetchNodeModel(BaseNodeModel):
    url: str
    method: str = "GET"
    headers: Optional[dict[str, Any] | str] = None
    params: Optional[dict[str, Any] | str] = None
    body: Optional[dict[str, Any] | str] = None
    data: Optional[dict[str, Any] | str] = None
    json_data: Optional[dict[str, Any] | str] = None
