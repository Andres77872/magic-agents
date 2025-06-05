import json

from magic_agents.models.factory.Nodes import ParserNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.util.template_parser import template_parse


class NodeParser(Node):
    def __init__(self, data: ParserNodeModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.text = data.text

    async def process(self, chat_log):

        def safe_json_parse(value):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        rp_inputs = {
            k: safe_json_parse(v)
            for k, v in self.inputs.items()
        }

        output = template_parse(template=self.text, params=rp_inputs)
        yield self.yield_static(output)
