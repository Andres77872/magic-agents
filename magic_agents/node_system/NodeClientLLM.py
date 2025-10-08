import json
import logging

from magic_agents.models.factory.Nodes import ClientNodeModel
from magic_agents.node_system.Node import Node
from magic_llm import MagicLLM

logger = logging.getLogger(__name__)


class NodeClientLLM(Node):
    def __init__(self,
                 data: ClientNodeModel,
                 node_id: str,
                 debug: bool = False,
                 **kwargs) -> None:
        super().__init__(
            node_id=node_id,
            debug=debug,
            **kwargs
        )

        api_info = data.api_info if type(data.api_info) is dict else json.loads(data.api_info)

        args = {
            'engine': data.engine,
            'model': data.model,
            **api_info,
            **data.extra_data
        }

        if 'api_key' in args:
            args['private_key'] = args['api_key']

        if self.debug:
            logger.debug("NodeClientLLM:%s initializing client engine=%s model=%s", self.node_id, data.engine, data.model)
        self.client = MagicLLM(**args)
        logger.info("NodeClientLLM:%s client initialized", self.node_id)

    async def process(self, chat_log):
        if self.debug:
            logger.debug("NodeClientLLM:%s yielding MagicLLM client", self.node_id)
        yield self.yield_static(self.client)
