import uuid
import logging

from magic_agents.models.factory.Nodes import UserInputNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeUserInput(Node):
    HANDLER_USER_MESSAGE = 'handle_user_message'
    HANDLER_USER_FILES = 'handle_user_files'
    HANDLER_USER_IMAGES = 'handle_user_images'

    def __init__(self, data: UserInputNodeModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = data.text
        self.files = data.files
        self.images = data.images

    async def process(self, chat_log):
        if not chat_log.id_chat:
            chat_log.id_chat = str(uuid.uuid4())
            logger.debug("NodeUserInput:%s generated new chat_id=%s", self.node_id, chat_log.id_chat)
        if not chat_log.id_thread:
            chat_log.id_thread = str(uuid.uuid4())
            logger.debug("NodeUserInput:%s generated new thread_id=%s", self.node_id, chat_log.id_thread)
        logger.info("NodeUserInput:%s processing user input (text_len=%d, files=%d, images=%d)", 
                   self.node_id, len(self._text) if self._text else 0, 
                   len(self.files) if self.files else 0, 
                   len(self.images) if self.images else 0)
        yield self.yield_static(self._text, content_type=self.HANDLER_USER_MESSAGE)
        yield self.yield_static(self.files, content_type=self.HANDLER_USER_FILES)
        yield self.yield_static(self.images, content_type=self.HANDLER_USER_IMAGES)
