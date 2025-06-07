import json
from typing import Callable

from magic_llm.model import ModelChat

from magic_agents.node_system.Node import Node


class NodeChat(Node):
    INPUT_HANDLER_SYSTEM_CONTEXT = 'handle-system-context'
    INPUT_HANDLER_USER_MESSAGE = 'handle_user_message'
    INPUT_HANDLER_MESSAGES = 'handle_messages'
    INPUT_HANDLER_USER_FILES = 'handle_user_files'
    INPUT_HANDLER_USER_IMAGES = 'handle_user_images'

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
                memory_chat=memory.get('stm', 0),
                long_memory_chat=memory.get('ltm', 0))
        else:
            self.chat = ModelChat(max_input_tokens=memory.get('max_input_tokens'))

    async def process(self, chat_log):
        if c := self.get_input(self.INPUT_HANDLER_MESSAGES):
            self.chat.messages = c
        else:
            if c := self.get_input(self.INPUT_HANDLER_SYSTEM_CONTEXT):
                self.chat.set_system(c)
            if c := self.get_input(self.INPUT_HANDLER_USER_MESSAGE):
                if im := self.get_input(self.INPUT_HANDLER_USER_IMAGES):
                    if isinstance(im, str):
                        im = json.loads(im)
                    self.chat.add_user_message(c, im)
                else:
                    self.chat.add_user_message(c)

        yield self.yield_static(self.chat)
