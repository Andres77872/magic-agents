from pydantic import BaseModel

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class SendMessageDataModel(BaseModel):
    json_extras: str


class SendMessageNodeModel(BaseNodeModel):
    data: SendMessageDataModel
