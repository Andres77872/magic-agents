import json

from magic_llm.model import ModelChat

from magic_agents.node_system.Node import Node
from magic_agents.util.const import HANDLE_CHAT, HANDLE_SYSTEM_CONTEXT, HANDLE_USER_MESSAGE


class NodeLLM(Node):
    def __init__(self,
                 stream: bool = True,
                 json_output: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.stream = stream
        self.json_output = json_output
        self.extra_data = kwargs
        self.generated = ''

    async def __call__(self, chat_log):
        params = self.parents
        client = self.parents['handle-client-provider']
        if c := params.get(HANDLE_CHAT):
            chat = c['chat']
        else:
            chat = ModelChat(params.get(HANDLE_SYSTEM_CONTEXT))
            if k := params.get(HANDLE_USER_MESSAGE):
                chat.add_user_message(k)

        if not self.stream:
            if chat.messages[0]['role'] == 'system':
                chat.messages = chat.messages[-5:]
            else:
                chat.messages = chat.messages[-4:]
            intention = await client.llm.async_generate(chat, **self.extra_data)
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
