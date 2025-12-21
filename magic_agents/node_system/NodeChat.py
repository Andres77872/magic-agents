import json
import logging
from typing import Callable, Optional

from magic_llm.model import ModelChat

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeChat(Node):
    """
    Chat node - handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    """
    # Default handle names - can be overridden by JSON data.handles
    DEFAULT_INPUT_SYSTEM_CONTEXT = 'handle-system-context'
    DEFAULT_INPUT_USER_MESSAGE = 'handle_user_message'
    DEFAULT_INPUT_MESSAGES = 'handle_messages'
    DEFAULT_INPUT_USER_FILES = 'handle_user_files'
    DEFAULT_INPUT_USER_IMAGES = 'handle_user_images'
    # Output handle
    DEFAULT_OUTPUT_HANDLE = 'handle_chat_output'

    def __init__(self,
                 message: str,
                 load_chat: Callable,
                 memory: Optional[dict] = None,
                 handles: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._memory = memory or {}
        # Allow JSON to override handle names
        handles = handles or {}
        self.INPUT_HANDLER_SYSTEM_CONTEXT = handles.get('system_context', handles.get('system', self.DEFAULT_INPUT_SYSTEM_CONTEXT))
        self.INPUT_HANDLER_USER_MESSAGE = handles.get('user_message', handles.get('message', self.DEFAULT_INPUT_USER_MESSAGE))
        self.INPUT_HANDLER_MESSAGES = handles.get('messages', self.DEFAULT_INPUT_MESSAGES)
        self.INPUT_HANDLER_USER_FILES = handles.get('user_files', handles.get('files', self.DEFAULT_INPUT_USER_FILES))
        self.INPUT_HANDLER_USER_IMAGES = handles.get('user_images', handles.get('images', self.DEFAULT_INPUT_USER_IMAGES))
        # Output handle
        self.OUTPUT_HANDLE = handles.get('output', handles.get('chat', self.DEFAULT_OUTPUT_HANDLE))
        if load_chat:
            self.chat = load_chat(
                message=message,
                memory_chat=self._memory.get('stm', 0),
                long_memory_chat=self._memory.get('ltm', 0))
        else:
            self.chat = ModelChat(max_input_tokens=self._memory.get('max_input_tokens'))

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
                        yield self.yield_debug_error(
                            error_type="ValidationError",
                            error_message="UserImage and UserFile cannot be used together. Images must be either all single strings or all pairs.",
                            context={
                                "images_input": im,
                                "has_single_strings": is_list_single,
                                "has_pairs": is_list_pair
                            }
                        )
                        return
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
        yield self.yield_static(self.chat, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture Chat-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add Chat-specific variables
        if hasattr(self, 'chat') and self.chat:
            state['messages_count'] = len(self.chat.messages) if hasattr(self.chat, 'messages') else 0
            state['has_system_message'] = hasattr(self.chat, 'system') and self.chat.system is not None
        
        # Capture memory configuration
        state['memory'] = self._memory
        
        return state
