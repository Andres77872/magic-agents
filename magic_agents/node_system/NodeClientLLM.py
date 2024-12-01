from magic_llm import MagicLLM

from magic_agents.node_system.Node import Node


class NodeClientLLM(Node):
    def __init__(self,
                 engine: str,
                 api_key: str,
                 base_url: str,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = MagicLLM(engine=engine,
                               private_key=api_key,
                               base_url=base_url,
                               **kwargs)

    async def __call__(self, chat_log) -> dict:
        yield {
            'type': 'end',
            'content': super().prep(self.client)
        }
