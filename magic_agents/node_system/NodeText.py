from magic_agents.models.factory.Nodes.TextNodeModel import TextNodeModel
from magic_agents.node_system.Node import Node


class NodeText(Node):
    def __init__(self, data: TextNodeModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = data.text

    async def process(self, chat_log):
        yield {
            'type': 'end',
            'content': super().prep(self._text)
        }
