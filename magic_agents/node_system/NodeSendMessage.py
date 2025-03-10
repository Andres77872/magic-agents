import json

from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.models.factory.Nodes import SendMessageNodeModel
from magic_agents.node_system.Node import Node


class NodeSendMessage(Node):
    def __init__(self,
                 data: SendMessageNodeModel,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.message = data.message
        self.json_extras = data.json_extras

    async def process(self, chat_log):
        output = self.inputs['handle_send_extra']
        if self.json_extras:
            output = json.loads(output)
        yield {
            'type': 'content',
            'content': ChatCompletionModel(id='', model='', choices=[ChoiceModel()], extras=output)
        }
