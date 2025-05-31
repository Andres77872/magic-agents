import json
import uuid

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.models.factory.Nodes import LlmNodeModel
from magic_agents.node_system.Node import Node


class NodeLLM(Node):
    INPUT_HANDLER_CLIENT_PROVIDER = 'handle-client-provider'
    INPUT_HANDLER_CHAT = 'handle-chat'
    INPUT_HANDLER_SYSTEM_CONTEXT = 'handle-system-context'
    INPUT_HANDLER_USER_MESSAGE = 'handle_user_message'

    def __init__(self,
                 data: LlmNodeModel,
                 node_id: str,
                 debug: bool = False,
                 **kwargs):
        super().__init__(
            debug=debug,
            node_id=node_id,
            **kwargs)
        self.stream = data.stream
        self.json_output = data.json_output
        self.extra_data = data.extra_data
        self.generated = ''

    async def process(self, chat_log):
        params = self.inputs
        client: MagicLLM = self.get_input(self.INPUT_HANDLER_CLIENT_PROVIDER, required=True)
        if c := params.get(self.INPUT_HANDLER_CHAT):
            chat = c
        else:
            chat = ModelChat(params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT))
            if k := params.get(self.INPUT_HANDLER_USER_MESSAGE):
                chat.add_user_message(k)
            else:
                raise ValueError('No message provided')

        if not self.stream:
            intention = await client.llm.async_generate(chat, **self.extra_data)
            self.generated = intention.content
            yield self.yield_static(ChatCompletionModel(
                id=uuid.uuid4().hex,
                model=client.llm.model,
                choices=[ChoiceModel()],
                usage=intention.usage),
                content_type='content')
        else:
            async for i in client.llm.async_stream_generate(chat, **self.extra_data):
                self.generated += i.choices[0].delta.content
                yield self.yield_static(i, content_type='content')
        if self.json_output:
            self.generated = json.loads(self.generated)
        yield self.yield_static(self.generated)
