from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.node_system.Node import Node


class NodeEND(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    async def process(self, chat_log):
        yield self.yield_static(ChatCompletionModel(id='', model='', choices=[ChoiceModel()]))
