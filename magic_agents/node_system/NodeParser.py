import json
import logging

from magic_agents.models.factory.Nodes import ParserNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.util.template_parser import template_parse

logger = logging.getLogger(__name__)


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
        
        logger.debug("NodeParser:%s parsing template with %d inputs", self.node_id, len(rp_inputs))
        output = template_parse(template=self.text, params=rp_inputs)
        logger.info("NodeParser:%s template parsed successfully (output_len=%d)", self.node_id, len(str(output)))
        yield self.yield_static(output)
