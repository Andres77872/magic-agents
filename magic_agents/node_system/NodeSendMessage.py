import json
import logging
from typing import Optional

from magic_agents.models.factory.Nodes import SendMessageNodeModel
from magic_agents.node_system.Node import Node
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

logger = logging.getLogger(__name__)


class NodeSendMessage(Node):
    """
    SendMessage node - handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    
    Emits content directly to the user stream via OUTPUT_HANDLE_CONTENT.
    """
    # Default handle names - can be overridden by JSON data.handles
    DEFAULT_INPUT_SEND_EXTRA = 'handle_send_extra'
    # Output handle for routing to downstream nodes
    DEFAULT_OUTPUT_HANDLE = 'handle_message_output'
    # Streaming content handle - used by executor to forward to user
    OUTPUT_HANDLE_CONTENT = 'content'

    def __init__(self,
                 data: SendMessageNodeModel,
                 handles: Optional[dict] = None,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.message = data.message
        self.json_extras = data.json_extras
        # Allow JSON to override handle names
        handles = handles or {}
        self.INPUT_HANDLE_SEND_EXTRA = handles.get('send_extra', handles.get('extra', self.DEFAULT_INPUT_SEND_EXTRA))
        # Output handle
        self.OUTPUT_HANDLE = handles.get('output', handles.get('message', self.DEFAULT_OUTPUT_HANDLE))

    async def process(self, chat_log):
        output = self.get_input(self.INPUT_HANDLE_SEND_EXTRA)
        if output:
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                    logger.debug("NodeSendMessage:%s parsed extra output from JSON", self.node_id)
                except json.JSONDecodeError:
                    output = {'text': output}
                    logger.debug("NodeSendMessage:%s using raw string as extra output", self.node_id)
        else:
            output = {}
        logger.info("NodeSendMessage:%s sending message with extras", self.node_id)
        
        message = ChatCompletionModel(
            id='',
            model='',
            choices=[ChoiceModel(delta=DeltaModel(content=self.json_extras))],
            extras=output
        )
        
        # Yield for streaming to user (executor checks OUTPUT_HANDLE_CONTENT)
        yield self.yield_static(message, content_type=self.OUTPUT_HANDLE_CONTENT)
        
        # Yield for routing to downstream nodes
        yield self.yield_static(message, content_type=self.OUTPUT_HANDLE)

    def _capture_internal_state(self):
        """Capture SendMessage-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add SendMessage-specific variables
        state['message'] = self.message
        state['json_extras'] = self.json_extras
        
        return state
