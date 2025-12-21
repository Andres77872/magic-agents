import logging
from typing import Optional

from magic_agents.models.factory.Nodes import TextNodeModel
from magic_agents.node_system.Node import Node

logger = logging.getLogger(__name__)


class NodeText(Node):
    """
    Text node - output handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    """
    # Default output handle name - can be overridden by JSON data.handles
    DEFAULT_OUTPUT_HANDLE = 'handle_text_output'

    def __init__(self, data: TextNodeModel, handles: Optional[dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._text = data.text
        # Allow JSON to override handle names
        handles = handles or {}
        self.OUTPUT_HANDLE = handles.get('output', handles.get('text', self.DEFAULT_OUTPUT_HANDLE))

    async def process(self, chat_log):
        logger.info("NodeText:%s yielding static text (len=%d)", self.node_id, len(self._text) if self._text else 0)
        yield self.yield_static(self._text, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture Text-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add Text-specific variables
        state['text'] = self._text[:500] if self._text and len(self._text) > 500 else self._text
        state['text_length'] = len(self._text) if self._text else 0
        
        return state
