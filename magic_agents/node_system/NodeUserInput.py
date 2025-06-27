import uuid

from magic_agents.models.factory.Nodes import UserInputNodeModel
from magic_agents.node_system.Node import Node


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
        if not chat_log.id_thread:
            chat_log.id_thread = str(uuid.uuid4())
        yield self.yield_static(self._text, content_type=self.HANDLER_USER_MESSAGE)
        yield self.yield_static(self.files, content_type=self.HANDLER_USER_FILES)
        yield self.yield_static(self.images, content_type=self.HANDLER_USER_IMAGES)
