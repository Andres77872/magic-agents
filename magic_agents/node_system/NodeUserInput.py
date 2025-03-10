import uuid

from magic_agents.models.factory.Nodes import UserInputNodeModel
from magic_agents.node_system.Node import Node


class NodeUserInput(Node):
    def __init__(self, data: UserInputNodeModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = data.text

    async def process(self, chat_log):
        if not chat_log.id_chat:
            chat_log.id_chat = str(uuid.uuid4())
        if not chat_log.id_thread:
            chat_log.id_thread = str(uuid.uuid4())
        yield {
            'type': 'end',
            'content': super().prep(self._text)
        }
