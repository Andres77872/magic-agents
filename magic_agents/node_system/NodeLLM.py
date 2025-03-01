import json

from magic_llm import MagicLLM
from magic_llm.model import ModelChat

from magic_agents.node_system.Node import Node


class NodeLLM(Node):
    INPUT_HANDLER_CLIENT_PROVIDER = 'handle-client-provider'
    INPUT_HANDLER_CHAT = 'handle-chat'
    INPUT_HANDLER_SYSTEM_CONTEXT = 'handle-system-context'
    INPUT_HANDLER_USER_MESSAGE = 'handle_user_message'

    def __init__(self,
                 node_id: str,
                 stream: bool = True,
                 json_output: bool = False,
                 debug: bool = False,
                 **kwargs):
        super().__init__(
            debug=debug,
            node_id=node_id,
            **kwargs)
        self.stream = stream
        self.json_output = json_output
        self.extra_data = kwargs
        self.generated = ''

    async def process(self, chat_log):
        params = self.inputs
        client: MagicLLM = self.inputs[self.INPUT_HANDLER_CLIENT_PROVIDER]
        if c := params.get(self.INPUT_HANDLER_CHAT):
            chat = c
        else:
            chat = ModelChat(params.get(self.INPUT_HANDLER_SYSTEM_CONTEXT))
            if k := params.get(self.INPUT_HANDLER_USER_MESSAGE):
                chat.add_user_message(k)
            else:
                raise ValueError('No message provided')

        if not self.stream:
            if chat.messages[0]['role'] == 'system':
                chat.messages = chat.messages[-5:]
            else:
                chat.messages = chat.messages[-4:]
            intention = await client.llm.async_generate(chat, **self.extra_data)
            print('INTENTION', intention)
            self.generated = intention.content
        else:
            async for i in client.llm.async_stream_generate(chat, **self.extra_data):
                self.generated += i.choices[0].delta.content
                yield {
                    'type': 'content',
                    'content': i
                }
        if self.json_output:
            self.generated = json.loads(self.generated)
        yield {
            'type': 'end',
            'content': super().prep(self.generated)
        }
