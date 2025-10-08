import logging

from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeEND(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    async def process(self, chat_log):
        logger.info("NodeEND:%s execution completed", self.node_id)
        yield self.yield_static(ChatCompletionModel(id='', model='', choices=[ChoiceModel()]))
