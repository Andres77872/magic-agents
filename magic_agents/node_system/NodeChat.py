from typing import Callable

from magic_llm.model import ModelChat

from magic_agents.node_system.Node import Node
from magic_agents.util.const import HANDLE_SYSTEM_CONTEXT, HANDLE_USER_MESSAGE, HANDLE_USER_MESSAGE_CONTEXT


class NodeChat(Node):
    def __init__(self,
                 memory: dict,
                 message: str,
                 load_chat: Callable,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._memory = memory
        if load_chat:
            self.chat = load_chat(
                message=message,
                memory_chat=memory['stm'],
                long_memory_chat=memory['ltm'])
        else:
            self.chat = ModelChat()

    async def __call__(self, chat_log):
        print('Node chat')
        params = self.parents
        print('PARENTS', params)
        if c := params.get(HANDLE_SYSTEM_CONTEXT):
            # chat_log.chat_system = c
            self.chat.set_system(c)
            print('CHAT_EVAL_TEST CONTEXT : ', self.chat)
        if c := params.get(HANDLE_USER_MESSAGE):
            self.chat.add_user_message(c)
            print('CHAT_EVAL_TEST MESSAGE : ', self.chat)
        if c := params.get(HANDLE_USER_MESSAGE_CONTEXT):
            k = self.chat.messages.pop(-1)
            c += '\n' + k['content']
            self.chat.add_user_message(c)
            print('CHAT_EVAL_TEST MESSAGE CONTEXT : ', self.chat)

        return super().prep({
            'chat': self.chat
        })
