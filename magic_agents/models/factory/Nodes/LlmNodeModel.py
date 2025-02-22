from typing import Optional

from pydantic import BaseModel

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class LlmDataModel(BaseModel):
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    json_output: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class LlmNodeModel(BaseNodeModel):
    data: LlmDataModel
