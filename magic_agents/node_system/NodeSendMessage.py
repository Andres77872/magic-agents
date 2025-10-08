import json
import logging

from magic_agents.models.factory.Nodes import SendMessageNodeModel
from magic_agents.node_system.Node import Node
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

logger = logging.getLogger(__name__)


class NodeSendMessage(Node):
    def __init__(self,
                 data: SendMessageNodeModel,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.message = data.message
        self.json_extras = data.json_extras

    async def process(self, chat_log):
        output = self.get_input('handle_send_extra')
        if output:
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                    logger.debug("NodeSendMessage:%s parsed extra output from JSON", self.node_id)
                except json.JSONDecodeError:
                    output = {'text': output}
                    logger.debug("NodeSendMessage:%s using raw string as extra output", self.node_id)
        else:
            output = {}
        logger.info("NodeSendMessage:%s sending message with extras", self.node_id)
        yield self.yield_static(ChatCompletionModel(id='',
                                                    model='',
                                                    choices=[ChoiceModel(delta=DeltaModel(content=self.json_extras))],
                                                    extras=output),
                                content_type='content')
