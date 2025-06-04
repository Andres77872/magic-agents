import json

from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

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
        output = self.get_input('handle_send_extra')
        output = json.loads(output)
        yield self.yield_static(ChatCompletionModel(id='',
                                                    model='',
                                                    choices=[ChoiceModel(delta=DeltaModel(content=self.json_extras))],
                                                    extras=output),
                                content_type='content')
