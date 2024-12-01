from magic_agents.node_system.Node import Node
from magic_agents.util.template_parser import template_parse


class NodeParser(Node):
    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = text

    async def __call__(self, chat_log) -> dict:
        output = template_parse(template=self._text, params=self.parents)
        yield {
            'type': 'end',
            'content': super().prep(output)
        }
