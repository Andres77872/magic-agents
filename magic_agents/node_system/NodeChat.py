import json
import logging
from typing import Callable

from magic_llm.model import ModelChat

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeChat(Node):
    INPUT_HANDLER_SYSTEM_CONTEXT = 'handle-system-context'
    INPUT_HANDLER_USER_MESSAGE = 'handle_user_message'
    INPUT_HANDLER_MESSAGES = 'handle_messages'
    INPUT_HANDLER_USER_FILES = 'handle_user_files'
    INPUT_HANDLER_USER_IMAGES = 'handle_user_images'

    def __init__(self,
                 message: str,
                 load_chat: Callable,
                 memory: dict = {},
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
            logger.debug("NodeChat:%s loading messages directly", self.node_id)
            self.chat.messages = c
        else:
            if c := self.get_input(self.INPUT_HANDLER_SYSTEM_CONTEXT):
                logger.debug("NodeChat:%s setting system context", self.node_id)
                self.chat.set_system(c)
            if c := self.get_input(self.INPUT_HANDLER_USER_MESSAGE):
                if im := self.get_input(self.INPUT_HANDLER_USER_IMAGES):
                    if isinstance(im, str):
                        im = json.loads(im)
                    is_list_single = False
                    is_list_pair = False
                    for i in im:
                        if isinstance(i, str):
                            is_list_single = True
                        elif isinstance(i, list):
                            is_list_pair = True
                    if is_list_single and is_list_pair:
                        logger.error("NodeChat:%s UserImage and UserFile cannot be used together", self.node_id)
                        raise ValueError("UserImage and UserFile cannot be used together")
                    if is_list_single:
                        logger.debug("NodeChat:%s adding user message with images (single list)", self.node_id)
                        self.chat.add_user_message(c, im)
                    elif is_list_pair:
                        logger.debug("NodeChat:%s adding user message with images (pair list)", self.node_id)
                        for i in im:
                            self.chat.add_user_message(i[0], i[1])
                        self.chat.add_user_message(c)
                else:
                    logger.debug("NodeChat:%s adding user message", self.node_id)
                    self.chat.add_user_message(c)
        logger.info("NodeChat:%s chat prepared with %d messages", self.node_id, len(self.chat.messages))
        yield self.yield_static(self.chat)
