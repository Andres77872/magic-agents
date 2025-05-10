from magic_llm import MagicLLM

from magic_agents.models.factory.Nodes import ClientNodeModel
from magic_agents.node_system.Node import Node


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

        args = {
            'private_key': data.api_key,
            'base_url': data.base_url,
            'engine': data.engine,
            'model': data.model,
            **data.extra_data
        }

        self.client = MagicLLM(**args)

    async def process(self, chat_log):
        yield self.yield_static(self.client)
