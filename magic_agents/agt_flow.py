import logging
import uuid
from typing import Callable, Dict, Any, AsyncGenerator, Optional, Union

from magic_llm.model.ModelChatStream import ChatCompletionModel

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


def create_node(node: dict, load_chat: Callable, debug: bool = False) -> Any:
    """
    Factory method to create node instances.
    
    Args:
    node (dict): Node data.
    load_chat (Callable): Load chat function.
    debug (bool): Debug mode. Defaults to False.
    
    Returns:
    Any: Node instance.
    """
    extra = {'debug': debug, 'node_id': node['id']}
    node_type = node['type']
    node_data = node.get('data', {})
    # Mapping of node types to (constructor, model)
    node_map = {
        ModelAgentFlowTypesModel.CHAT:   (NodeChat, None),
        ModelAgentFlowTypesModel.LLM:    (NodeLLM, LlmNodeModel),
        ModelAgentFlowTypesModel.END:    (NodeEND, None),
        ModelAgentFlowTypesModel.TEXT:   (NodeText, TextNodeModel),
        ModelAgentFlowTypesModel.USER_INPUT: (NodeUserInput, UserInputNodeModel),
        ModelAgentFlowTypesModel.PARSER: (NodeParser, ParserNodeModel),
        ModelAgentFlowTypesModel.FETCH:  (NodeFetch, FetchNodeModel),
        ModelAgentFlowTypesModel.CLIENT: (NodeClientLLM, ClientNodeModel),
        ModelAgentFlowTypesModel.SEND_MESSAGE: (NodeSendMessage, SendMessageNodeModel),
        ModelAgentFlowTypesModel.VOID:   (NodeEND, None),
    }
    if node_type not in node_map:
        raise ValueError(f"Unsupported node type: {node_type}")
    constructor, model_cls = node_map[node_type]
    if node_type == ModelAgentFlowTypesModel.CHAT:
        return constructor(load_chat=load_chat, **extra, **node_data)
    elif model_cls:
        return constructor(**extra, data=model_cls(**extra, **node_data))
    else:
        return constructor(**extra)


async def execute_graph(graph: AgentFlowModel,
                       id_chat: Optional[Union[int, str]] = None,
                       id_thread: Optional[Union[int, str]] = None,
                       id_user: Optional[Union[int, str]] = None
                       ) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Execute the agent flow graph asynchronously, yielding ChatCompletionModel results as generated.
    
    Args:
    graph (AgentFlowModel): Agent flow graph.
    id_chat (Optional[Union[int, str]]): Chat ID. Defaults to None.
    id_thread (Optional[Union[int, str]]): Thread ID. Defaults to None.
    id_user (Optional[Union[int, str]]): User ID. Defaults to None.
    
    Yields:
    AsyncGenerator[ChatCompletionModel, None]: ChatCompletionModel results.
    """
    nodes = graph.nodes
    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app='magic-research'
    )
    async def process_edge(edge: EdgeNodeModel):
        source_node = nodes.get(edge.source)
        target_node = nodes.get(edge.target)
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
    """
    Prepare and build the agent flow graph from input data and message.
    
    Args:
    agt_data: Agent data.
    message (str): Message.
    load_chat: Load chat function. Defaults to None.
    
    Returns:
    AgentFlowModel: Agent flow graph.
    """
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
        node['id']: create_node(node, load_chat, agt_data.get('debug', False)) for node in agt_data['nodes']
    }
    agt_data['nodes'] = nodes
    agt = AgentFlowModel(**agt_data)
    return agt


async def run_agent(
        graph: AgentFlowModel,
        id_chat: Optional[Union[int, str]] = None,
        id_thread: Optional[Union[int, str]] = None,
        id_user: Optional[Union[int, str]] = None) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Run the agent flow and yield ChatCompletionModel results as they are generated.
    
    Args:
    graph (AgentFlowModel): Agent flow graph.
    id_chat (Optional[Union[int, str]]): Chat ID. Defaults to None.
    id_thread (Optional[Union[int, str]]): Thread ID. Defaults to None.
    id_user (Optional[Union[int, str]]): User ID. Defaults to None.
    
    Yields:
    AsyncGenerator[ChatCompletionModel, None]: ChatCompletionModel results.
    """
    async for result in execute_graph(
            graph=graph,
            id_chat=id_chat,
            id_thread=id_thread,
            id_user=id_user):
        yield result
