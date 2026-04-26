import logging
from typing import Optional

from magic_agents.models.factory.Nodes import UserInputNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeUserInput(Node):
    """
    UserInput node - output handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    
    Session management aligned with backend thread persistence contract:
    - Uses external session_id from configuration (no ephemeral UUID generation)
    - Reuses chat_log.id_chat from backend response if available
    - Backend auto-creates thread when session_required=True and no session exists
    """
    # Default output handle names - can be overridden by JSON data.handles
    DEFAULT_OUTPUT_USER_MESSAGE = 'handle_user_message'
    DEFAULT_OUTPUT_USER_FILES = 'handle_user_files'
    DEFAULT_OUTPUT_USER_IMAGES = 'handle_user_images'
    DEFAULT_OUTPUT_CLIENT_EXTRAS = 'handle_client_extras'  # Client extras output handle

    def __init__(self, data: UserInputNodeModel, handles: Optional[dict] = None, **kwargs) -> None:
        """
        Initialize UserInput node with validated UserInputNodeModel.
        
        Args:
            data: UserInputNodeModel instance with validated session configuration
        """
        super().__init__(**kwargs)
        self._text = data.text
        self.files = data.files
        self.images = data.images
        self._extras = data.extras  # Client-provided extras
        
        # Session configuration from validated model
        self._session_id = data.session_id
        self._session_required = data.session_required
        
        # Allow JSON to override handle names
        handles = handles or {}
        self.HANDLER_USER_MESSAGE = handles.get('user_message', handles.get('message', self.DEFAULT_OUTPUT_USER_MESSAGE))
        self.HANDLER_USER_FILES = handles.get('user_files', handles.get('files', self.DEFAULT_OUTPUT_USER_FILES))
        self.HANDLER_USER_IMAGES = handles.get('user_images', handles.get('images', self.DEFAULT_OUTPUT_USER_IMAGES))
        self.HANDLER_CLIENT_EXTRAS = handles.get('client_extras', self.DEFAULT_OUTPUT_CLIENT_EXTRAS)

    async def process(self, chat_log):
        """
        Process user input with session-aware behavior.
        
        Session handling:
        - If session_id configured: use it (thread reuse)
        - If chat_log.id_chat exists: use it (backend-provided)
        - If session_required and no session: backend auto-creates (no frontend generation)
        - If no session required: proceed without session (backward compatible)
        
        Args:
            chat_log: ModelAgentRunLog with id_chat for session context
        """
        # Use external session_id from configuration if provided
        if self._session_id:
            chat_log.id_chat = self._session_id
            logger.debug("NodeUserInput:%s using configured session_id=%s", self.node_id, chat_log.id_chat)
        elif chat_log.id_chat:
            # Reuse existing session from backend response
            logger.debug("NodeUserInput:%s reusing backend session_id=%s", self.node_id, chat_log.id_chat)
        elif self._session_required:
            # Backend will auto-create thread when session_required=True
            # No frontend UUID generation - backend is authoritative
            logger.debug("NodeUserInput:%s session_required=True, backend will auto-create thread", self.node_id)
            # Note: chat_log.id_chat may remain None here; backend creates thread on first turn
        else:
            # No session requirement - backward compatible path
            # chat_log.id_chat remains as-is (may be None or externally provided)
            logger.debug("NodeUserInput:%s no session required, proceeding without persistence", self.node_id)
        
        # Thread ID handling (same logic as session_id)
        if self._session_id and not chat_log.id_thread:
            chat_log.id_thread = self._session_id  # Use same ID for thread context
            logger.debug("NodeUserInput:%s setting thread_id=%s", self.node_id, chat_log.id_thread)
        
        logger.info("NodeUserInput:%s processing user input (text_len=%d, files=%d, images=%d, session=%s)", 
                   self.node_id, len(self._text) if self._text else 0, 
                   len(self.files) if self.files else 0, 
                   len(self.images) if self.images else 0,
                   chat_log.id_chat)
        yield self.yield_static(self._text, content_type=self.HANDLER_USER_MESSAGE)
        yield self.yield_static(self.files, content_type=self.HANDLER_USER_FILES)
        yield self.yield_static(self.images, content_type=self.HANDLER_USER_IMAGES)
        # Yield client extras only if present (backward compatible)
        if self._extras is not None:
            yield self.yield_static(self._extras, content_type=self.HANDLER_CLIENT_EXTRAS)

    def _capture_internal_state(self):
        """Capture UserInput-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add UserInput-specific variables as documented
        state['text'] = self._text
        state['images'] = self.images if self.images else []
        state['files'] = self.files if self.files else []
        state['extras'] = self._extras  # Client extras for debugging visibility
        
        # Capture session configuration
        state['session_id'] = self._session_id
        state['session_required'] = self._session_required
        
        return state