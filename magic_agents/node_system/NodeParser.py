from magic_agents.node_system.Node import Node
from magic_agents.util.template_parser import template_parse


class NodeParser(Node):
    def __init__(self, text: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.text = text

    async def process(self, chat_log):
        output = template_parse(template=self.text, params=self.inputs)
        yield {
            'type': 'end',
            'content': super().prep(output)
        }
