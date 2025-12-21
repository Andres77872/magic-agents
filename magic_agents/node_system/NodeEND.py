import logging
from typing import Optional

from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeEND(Node):
    """
    END node - output handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    """
    # Default output handle name - can be overridden by JSON data.handles
    DEFAULT_OUTPUT_HANDLE = 'handle_end_output'

    def __init__(self, handles: Optional[dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # Allow JSON to override handle names
        handles = handles or {}
        self.OUTPUT_HANDLE = handles.get('output', handles.get('end', self.DEFAULT_OUTPUT_HANDLE))

    async def process(self, chat_log):
        logger.info("NodeEND:%s execution completed", self.node_id)
        yield self.yield_static(ChatCompletionModel(id='', model='', choices=[ChoiceModel()]), content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture END-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add END-specific marker
        state['is_terminal_node'] = True
        
        return state
