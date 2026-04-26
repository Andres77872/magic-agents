import json
import logging
from typing import Optional

from magic_llm.model import ModelChat

from magic_agents.models.factory.Nodes.ChatNodeModel import ChatNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeChat(Node):
    """
    Chat node - handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    
    Merge order (per backend contract):
    1. Persisted session history (from backend via chat_log.id_chat)
    2. Runtime messages input (handle_messages) - APPEND or REPLACE based on mode
    3. custom_messages from config - APPEND
    4. System context - INSERT at index 0
    5. User message - APPEND (final slot)
    """
    # Default handle names - can be overridden by JSON data.handles
    DEFAULT_INPUT_SYSTEM_CONTEXT = 'handle-system-context'
    DEFAULT_INPUT_USER_MESSAGE = 'handle_user_message'
    DEFAULT_INPUT_MESSAGES = 'handle_messages'
    DEFAULT_INPUT_USER_FILES = 'handle_user_files'
    DEFAULT_INPUT_USER_IMAGES = 'handle_user_images'
    # Output handle
    DEFAULT_OUTPUT_HANDLE = 'handle_chat_output'

    def __init__(self, data: ChatNodeModel, **kwargs) -> None:
        """
        Initialize Chat node with validated ChatNodeModel.
        
        Args:
            data: ChatNodeModel instance with validated session configuration
            
        BACKEND-AUTHORITATIVE ARCHITECTURE:
        - Backend injects persisted + runtime history via `data.history_messages`
        - NodeChat does NOT load persisted history from DB - backend prepares it
        - NodeChat composes additional layers on top of backend-provided base
        """
        super().__init__(**kwargs)
        
        # Session configuration from validated model
        self._session_id = data.session_id
        self._session_required = data.session_required
        self._messages_append_mode = data.messages_append_mode
        self._custom_messages = data.custom_messages or []
        
        # BACKEND-AUTHORITATIVE: History messages from backend (Slot 1)
        # Backend prepares persisted + runtime history and passes via build()
        self._history_messages = data.history_messages or []
        
        # Legacy fields (backward compatibility)
        self._memory = data.memory or {}
        
        # Handle name overrides from validated model
        handles = data.handles or {}
        self.INPUT_HANDLER_SYSTEM_CONTEXT = handles.get('system_context', handles.get('system', self.DEFAULT_INPUT_SYSTEM_CONTEXT))
        self.INPUT_HANDLER_USER_MESSAGE = handles.get('user_message', handles.get('message', self.DEFAULT_INPUT_USER_MESSAGE))
        self.INPUT_HANDLER_MESSAGES = handles.get('messages', self.DEFAULT_INPUT_MESSAGES)
        self.INPUT_HANDLER_USER_FILES = handles.get('user_files', handles.get('files', self.DEFAULT_INPUT_USER_FILES))
        self.INPUT_HANDLER_USER_IMAGES = handles.get('user_images', handles.get('images', self.DEFAULT_INPUT_USER_IMAGES))
        # Output handle
        self.OUTPUT_HANDLE = handles.get('output', handles.get('chat', self.DEFAULT_OUTPUT_HANDLE))
        
        # Initialize empty ModelChat - backend loads history via chat_log.id_chat
        self.chat = ModelChat(max_input_tokens=self._memory.get('max_input_tokens'))

    async def process(self, chat_log):
        """
        Process chat node with merge logic aligned with backend contract.
        
        Merge order (BACKEND-AUTHORITATIVE ARCHITECTURE):
        1. history_messages from backend (persisted + runtime) - injected via build()
        2. Runtime messages input (APPEND or REPLACE based on mode) - from input handles
        3. custom_messages from config (APPEND)
        4. System context (INSERT at index 0)
        5. User message (APPEND, final slot)
        
        Args:
            chat_log: ModelAgentRunLog with id_chat for session context
        """
        # Slot 1: Base messages from backend-authoritative history
        # Backend loads persisted history + runtime messages and passes via build()
        # NodeChat does NOT load from DB - backend is authoritative source
        base_messages = list(self._history_messages)  # Copy to avoid mutation
        
        if base_messages:
            logger.debug("NodeChat:%s starting with %d backend-injected history messages", 
                        self.node_id, len(base_messages))
        
        # Log session context for debugging
        session_id = self._session_id or (chat_log.id_chat if chat_log else None)
        if session_id:
            logger.debug("NodeChat:%s using session_id=%s for thread context", self.node_id, session_id)
        
        # Slot 2: Handle runtime messages input
        if c := self.get_input(self.INPUT_HANDLER_MESSAGES):
            if self._messages_append_mode:
                # APPEND mode: extend base messages (preserves history)
                logger.debug("NodeChat:%s appending runtime messages (append_mode=True)", self.node_id)
                base_messages.extend(c)
            else:
                # REPLACE mode: legacy semantics (overwrite history)
                logger.debug("NodeChat:%s replacing messages with runtime input (append_mode=False)", self.node_id)
                base_messages = list(c)  # Create new list to avoid mutation
        
        # Slot 3: Append custom_messages from config
        if self._custom_messages:
            logger.debug("NodeChat:%s appending %d custom_messages", self.node_id, len(self._custom_messages))
            base_messages.extend(self._custom_messages)
        
        # Build ModelChat with merged messages
        self.chat.messages = base_messages
        
        # Slot 4: System context (INSERT at index 0)
        if c := self.get_input(self.INPUT_HANDLER_SYSTEM_CONTEXT):
            logger.debug("NodeChat:%s setting system context", self.node_id)
            self.chat.set_system(c)
        
        # Slot 5: User message (final slot)
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
        
        logger.info("NodeChat:%s chat prepared with %d messages (session=%s, append_mode=%s)", 
                   self.node_id, len(self.chat.messages), session_id, self._messages_append_mode)
        yield self.yield_static(self.chat, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture Chat-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add Chat-specific variables
        if hasattr(self, 'chat') and self.chat:
            messages = getattr(self.chat, 'messages', [])
            state['messages_count'] = len(messages)
            # Check if system message exists in messages list
            state['has_system_message'] = any(
                msg.get('role') == 'system' for msg in messages if isinstance(msg, dict)
            )
        
        # Capture session configuration
        state['session_id'] = self._session_id
        state['session_required'] = self._session_required
        state['messages_append_mode'] = self._messages_append_mode
        state['custom_messages_count'] = len(self._custom_messages)
        
        # Capture memory configuration
        state['memory'] = self._memory
        
        return state