from typing import Optional

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class SendMessageNodeModel(BaseNodeModel):
    message: Optional[str] = ''
    json_extras: str
