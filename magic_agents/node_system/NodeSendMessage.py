import json

from magic_agents.models.factory.Nodes import SendMessageNodeModel
from magic_agents.node_system.Node import Node
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel


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
                except json.JSONDecodeError:
                    output = {'text': output}
        else:
            output = {}
        yield self.yield_static(ChatCompletionModel(id='',
                                                    model='',
                                                    choices=[ChoiceModel(delta=DeltaModel(content=self.json_extras))],
                                                    extras=output),
                                content_type='content')
