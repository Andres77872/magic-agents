import logging
import uuid
from typing import Callable, Dict, Any, AsyncGenerator, Optional, Union

from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes import (ModelAgentFlowTypesModel,
                                               LlmNodeModel,
                                               TextNodeModel,
                                               UserInputNodeModel,
                                               ParserNodeModel,
                                               FetchNodeModel,
                                               ClientNodeModel,
                                               SendMessageNodeModel)
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system import (
    NodeChat,
    NodeLLM,
    NodeEND,
    NodeText,
    NodeUserInput,
    NodeFetch,
    NodeClientLLM,
    NodeSendMessage,
    NodeParser,
    sort_nodes
)
from magic_agents.util.const import HANDLE_VOID

logger = logging.getLogger(__name__)


def create_node(node: dict, load_chat: Callable, debug: bool = False):
    """Factory method to create node instances."""
    extra = {'debug': debug, 'node_id': node['id']}
    match node['type']:
        case ModelAgentFlowTypesModel.CHAT:
            return NodeChat(load_chat=load_chat, **extra, **node['data'])
        case ModelAgentFlowTypesModel.LLM:
            return NodeLLM(**extra, data=LlmNodeModel(**extra, **node['data']))
        case ModelAgentFlowTypesModel.END:
            return NodeEND(**extra)
        case ModelAgentFlowTypesModel.TEXT:
            return NodeText(**extra, data=TextNodeModel(**extra, **node['data']))
        case ModelAgentFlowTypesModel.USER_INPUT:
            return NodeUserInput(**extra, data=UserInputNodeModel(**extra, **node.get('data', {})))
        case ModelAgentFlowTypesModel.PARSER:
            return NodeParser(**extra, data=ParserNodeModel(**extra, **node['data']))
        case ModelAgentFlowTypesModel.FETCH:
            return NodeFetch(**extra, data=FetchNodeModel(**extra, **node['data']))
        case ModelAgentFlowTypesModel.CLIENT:
            return NodeClientLLM(**extra, data=ClientNodeModel(**extra, **node['data']))
        case ModelAgentFlowTypesModel.SEND_MESSAGE:
            return NodeSendMessage(**extra, data=SendMessageNodeModel(**extra, **node['data']))
        case ModelAgentFlowTypesModel.VOID:
            return NodeEND(**extra)
        case _:
            raise ValueError(f"Unsupported node type: {node['type']}")


async def execute_graph(graph: AgentFlowModel,
                        id_chat: Optional[Union[int, str]] = None,
                        id_thread: Optional[Union[int, str]] = None,
                        id_user: Optional[Union[int, str]] = None
                        ) -> AsyncGenerator[str, None]:
    nodes = graph.nodes

    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app='magic-research'
    )

    async def process_edge(edge: EdgeNodeModel):
        source_node = nodes[edge.source]
        target_node = nodes[edge.target]

        if not source_node:
            logger.error(f"Source node {edge.source} not found.")
            return
        if not target_node:
            logger.error(f"Target node {edge.target} not found.")
            return

        # Execute source node (only if outputs not already computed)
        if not source_node.outputs:
            async for item in source_node(chat_log):
                if item["type"] == "end":
                    source_node.outputs[edge.sourceHandle] = item["content"]  # outputs must be structured as dict
                elif item["type"] == "content":
                    yield item["content"]

        # Pass output at source_handle to target_handle input
        source_handle = edge.sourceHandle
        target_handle = edge.targetHandle
        target_node.add_parent(source_node.outputs, source_handle, target_handle)

    for edge in graph.edges:
        async for result in process_edge(edge):
            yield result


def build(agt_data, message: str, load_chat=None) -> AgentFlowModel:
    nodes, edges = sort_nodes(agt_data['nodes'], agt_data['edges'])
    agt_data['nodes'] = nodes
    agt_data['edges'] = edges

    void_id = uuid.uuid4().hex
    agt_data['nodes'].append({'type': ModelAgentFlowTypesModel.VOID, 'id': void_id})

    # Prepare graph data
    for edge in agt_data['edges']:
        edge.setdefault('targetHandle', HANDLE_VOID)
        if edge['targetHandle'] == HANDLE_VOID:
            edge['target'] = void_id

    for node in agt_data['nodes']:
        if node['type'] in [ModelAgentFlowTypesModel.USER_INPUT, ModelAgentFlowTypesModel.CHAT]:
            node['data'] = node.get('data', {})
            node['data']['text' if node['type'] == ModelAgentFlowTypesModel.USER_INPUT else 'message'] = message
        elif node['type'] == ModelAgentFlowTypesModel.END:
            agt_data['edges'].append({
                "id": uuid.uuid4().hex,
                "source": node['id'],
                "target": void_id
            })

    nodes: Dict[str, Any] = {
        node['id']: create_node(node, load_chat, agt_data['debug']) for node in agt_data['nodes']
    }
    agt_data['nodes'] = nodes
    agt = AgentFlowModel(**agt_data)
    return agt


async def run_agent(
        graph: AgentFlowModel,
        id_chat: Optional[Union[int, str]] = None,
        id_thread: Optional[Union[int, str]] = None,
        id_user: Optional[Union[int, str]] = None) -> AsyncGenerator[str, None]:
    async for result in execute_graph(
            graph=graph,
            id_chat=id_chat,
            id_thread=id_thread,
            id_user=id_user):
        yield result
