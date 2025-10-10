from typing import Callable, TYPE_CHECKING
import logging

# from magic_agents.agt_flow import build, execute_graph
from magic_agents.models.factory.Nodes import InnerNodeModel
from magic_agents.node_system.Node import Node

if TYPE_CHECKING:
    from magic_agents.models.factory.AgentFlowModel import AgentFlowModel

logger = logging.getLogger(__name__)


class NodeInner(Node):
    """
    Node to execute a nested agent flow graph.

    Takes a single input (via handle_user_message) and a magic_flow property
    providing an AgentFlowModel spec. Streams all inner graph content events
    to the outer flow.
    """
    INPUT_HANDLE = 'handle_user_message'
    HANDLER_EXECUTION_CONTENT = 'handle_execution_content'
    HANDLER_EXECUTION_EXTRAS = 'handle_execution_extras'

    def __init__(self, data: InnerNodeModel, load_chat: Callable, **kwargs) -> None:
        super().__init__(**kwargs)
        self.magic_flow = data.magic_flow
        self._load_chat = load_chat
        self.inner_graph: 'AgentFlowModel' = None  # Will be set by build()

    async def process(self, chat_log):
        input_message = self.inputs.get(self.INPUT_HANDLE)
        if input_message is None:
            yield self.yield_debug_error(
                error_type="InputError",
                error_message=f"NodeInner requires input '{self.INPUT_HANDLE}'",
                context={
                    "available_inputs": list(self.inputs.keys()),
                    "required_input": self.INPUT_HANDLE
                }
            )
            return
        
        if self.inner_graph is None:
            yield self.yield_debug_error(
                error_type="ConfigurationError",
                error_message="NodeInner has no inner_graph set. The inner graph was not built correctly.",
                context={
                    "has_magic_flow": self.magic_flow is not None,
                    "magic_flow_keys": list(self.magic_flow.keys()) if isinstance(self.magic_flow, dict) else None
                }
            )
            return
        
        # Update input nodes in the inner graph with the current message
        from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
        for node_id, node in self.inner_graph.nodes.items():
            node_type = getattr(node, 'node_type', None)
            if node_type in [ModelAgentFlowTypesModel.USER_INPUT, ModelAgentFlowTypesModel.CHAT]:
                if node_type == ModelAgentFlowTypesModel.USER_INPUT:
                    node._text = input_message  # Update the internal text directly
                elif node_type == ModelAgentFlowTypesModel.CHAT:
                    node.message = input_message
        
        # Execute the inner graph
        from magic_agents.agt_flow import execute_graph
        content = ''
        extras = []
        async for evt in execute_graph(
                self.inner_graph,
                id_chat=chat_log.id_chat,
                id_thread=chat_log.id_thread,
                id_user=chat_log.id_user
        ):
            event = evt['content']
            # Check if event is a ChatCompletionModel
            if hasattr(event, 'choices') and event.choices:
                # It's a ChatCompletionModel
                event_content = event
                if event_content.choices[0].delta.content:
                    content += event_content.choices[0].delta.content
                if hasattr(event_content, 'extras') and event_content.extras:
                    extras.append(event_content.extras)
            else:
                # It's some other type of output - try to convert to string
                if self.debug:
                    logger.debug("NodeInner:%s received non-ChatCompletionModel: %s", self.node_id, type(event))
                # For now, we'll skip non-ChatCompletionModel outputs
                # In a full implementation, you might want to handle these differently
                pass
        
        yield self.yield_static(content, content_type=self.HANDLER_EXECUTION_CONTENT)
        if extras:
            yield self.yield_static(extras, content_type=self.HANDLER_EXECUTION_EXTRAS)
