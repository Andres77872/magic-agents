import json
import logging
from typing import Optional

from magic_agents.models.factory.Nodes import ParserNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.util.template_parser import template_parse

logger = logging.getLogger(__name__)


class NodeParser(Node):
    """
    Parser node - output handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    """
    # Default output handle name - can be overridden by JSON data.handles
    DEFAULT_OUTPUT_HANDLE = 'handle_parser_output'

    def __init__(self, data: ParserNodeModel, handles: Optional[dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.text = data.text
        # Allow JSON to override handle names
        handles = handles or {}
        self.OUTPUT_HANDLE = handles.get('output', handles.get('result', self.DEFAULT_OUTPUT_HANDLE))

    async def process(self, chat_log):

        def safe_json_parse(value):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        rp_inputs = {
            k: safe_json_parse(v)
            for k, v in self.inputs.items()
        }
        
        logger.debug("NodeParser:%s parsing template with %d inputs", self.node_id, len(rp_inputs))
        output = template_parse(template=self.text, params=rp_inputs)
        logger.info("NodeParser:%s template parsed successfully (output_len=%d)", self.node_id, len(str(output)))
        yield self.yield_static(output, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture Parser-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add Parser-specific variables
        state['template'] = self.text[:500] if len(self.text) > 500 else self.text  # Truncate long templates
        state['template_length'] = len(self.text)
        
        return state
