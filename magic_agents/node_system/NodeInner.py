from typing import Any, Dict, AsyncGenerator, Callable

from magic_agents.node_system.Node import Node
from magic_agents.models.factory.Nodes import InnerNodeModel
from magic_agents.agt_flow import build, execute_graph


class NodeInner(Node):
    """
    Node to execute a nested agent flow graph.

    Takes a single input (via handle_user_message) and a magic_flow property
    providing an AgentFlowModel spec. Streams all inner graph content events
    to the outer flow.
    """
    INPUT_HANDLE = 'handle_user_message'

    def __init__(self, data: InnerNodeModel, load_chat: Callable, **kwargs) -> None:
        super().__init__(**kwargs)
        self.magic_flow = data.magic_flow
        self._load_chat = load_chat

    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:
        input_message = self.inputs.get(self.INPUT_HANDLE)
        if input_message is None:
            raise ValueError(f"NodeInner '{self.node_id}' requires input '{self.INPUT_HANDLE}'")

        # Build and execute the inner graph with the same chat context
        inner_graph = build(
            self.magic_flow,
            message=input_message,
            load_chat=self._load_chat
        )
        async for event in execute_graph(
            inner_graph,
            id_chat=chat_log.id_chat,
            id_thread=chat_log.id_thread,
            id_user=chat_log.id_user
        ):
            yield {'type': 'content', 'content': event}