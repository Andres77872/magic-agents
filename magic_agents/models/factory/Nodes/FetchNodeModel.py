from typing import Optional, Any

from pydantic import model_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class FetchNodeModel(BaseNodeModel):
    """
    Fetch node model - accepts various field names from JSON.
    The JSON definition is the source of truth.
    """
    url: Optional[str] = None
    endpoint: Optional[str] = None  # alias for url
    method: str = "GET"
    headers: Optional[dict[str, Any] | str] = None
    params: Optional[dict[str, Any] | str] = None
    query: Optional[dict[str, Any] | str] = None  # alias for params
    body: Optional[dict[str, Any] | str] = None
    data: Optional[dict[str, Any] | str] = None
    json_data: Optional[dict[str, Any] | str] = None
    json_body: Optional[dict[str, Any] | str] = None  # alias for json_data

    @model_validator(mode='after')
    def resolve_aliases(self):
        """Resolve fields from alternative names (JSON-first approach)."""
        if self.url is None and self.endpoint is not None:
            self.url = self.endpoint
        if self.params is None and self.query is not None:
            self.params = self.query
        if self.json_data is None and self.json_body is not None:
            self.json_data = self.json_body
        return self
