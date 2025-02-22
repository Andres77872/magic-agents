from pydantic import BaseModel

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class ParserDataModel(BaseModel):
    text: str


class ParserNodeModel(BaseNodeModel):
    data: ParserDataModel
