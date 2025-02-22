from typing import Optional

from pydantic import BaseModel

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class FetchDataModel(BaseModel):
    url: str
    method: str = "GET"
    headers: Optional[dict[str, str]] = None
    params: Optional[dict[str, str]] = None
    body: Optional[dict[str, str]] = None


class FetchNodeModel(BaseNodeModel):
    data: FetchDataModel
