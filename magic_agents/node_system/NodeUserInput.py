import uuid
import logging
from typing import Optional

from magic_agents.models.factory.Nodes import UserInputNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeUserInput(Node):
    """
    UserInput node - output handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    """
    # Default output handle names - can be overridden by JSON data.handles
    DEFAULT_OUTPUT_USER_MESSAGE = 'handle_user_message'
    DEFAULT_OUTPUT_USER_FILES = 'handle_user_files'
    DEFAULT_OUTPUT_USER_IMAGES = 'handle_user_images'

    def __init__(self, data: UserInputNodeModel, handles: Optional[dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = data.text
        self.files = data.files
        self.images = data.images
        # Allow JSON to override handle names
        handles = handles or {}
        self.HANDLER_USER_MESSAGE = handles.get('user_message', handles.get('message', self.DEFAULT_OUTPUT_USER_MESSAGE))
        self.HANDLER_USER_FILES = handles.get('user_files', handles.get('files', self.DEFAULT_OUTPUT_USER_FILES))
        self.HANDLER_USER_IMAGES = handles.get('user_images', handles.get('images', self.DEFAULT_OUTPUT_USER_IMAGES))

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

    def _capture_internal_state(self):
        """Capture UserInput-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add UserInput-specific variables as documented
        state['text'] = self._text
        state['images'] = self.images if self.images else []
        state['files'] = self.files if self.files else []
        
        return state
