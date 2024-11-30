from magic_agents.node_system.Node import Node


class NodeText(Node):
    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text

    async def __call__(self, chat_log) -> dict:
        print('Node text')
        return super().prep(self._text)
