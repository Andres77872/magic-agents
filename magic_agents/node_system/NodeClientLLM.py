from magic_llm import MagicLLM

from magic_agents.node_system.Node import Node


class NodeClientLLM(Node):
    def __init__(self,
                 engine: str,
                 api_key: str,
                 base_url: str,
                 node_id: str,
                 debug: bool = False,
                 **kwargs) -> None:
        super().__init__(
            node_id=node_id,
            debug=debug,
            **kwargs
        )

        kwargs['private_key'] = api_key
        kwargs['base_url'] = base_url
        kwargs['engine'] = engine

        self.client = MagicLLM(**kwargs)

    async def process(self, chat_log):
        yield {
            'type': 'end',
            'content': super().prep(self.client)
        }
