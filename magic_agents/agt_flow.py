import logging
import uuid
from typing import Callable, Dict, Any, AsyncGenerator, Optional, Union

from magic_llm.model.ModelChatStream import ChatCompletionModel

from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes import (
    ModelAgentFlowTypesModel,
    LlmNodeModel,
    TextNodeModel,
    UserInputNodeModel,
    ParserNodeModel,
    FetchNodeModel,
    ClientNodeModel,
    SendMessageNodeModel,
    LoopNodeModel,
)
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
    NodeLoop,
    sort_nodes,
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
    extra = {'debug': debug, 'node_id': node['id'], 'node_type': node['type']}
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
        ModelAgentFlowTypesModel.LOOP:         (NodeLoop, LoopNodeModel),
        ModelAgentFlowTypesModel.VOID:         (NodeEND, None),
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


async def execute_graph_loop(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None,
) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Execute an agent flow graph that contains a Loop node, handling dynamic iteration.
    """
    nodes = graph.nodes
    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app='magic-research'
    )
    # Identify the single Loop node
    loop_id = next(nid for nid, node in nodes.items() if isinstance(node, NodeLoop))
    # Partition edges based on Loop handles and spec ordering
    all_edges = list(graph.edges)
    idx_list = next(i for i, e in enumerate(all_edges)
                    if e.target == loop_id and e.targetHandle == NodeLoop.INPUT_HANDLE_LIST)
    idx_loop_in = next(i for i, e in enumerate(all_edges)
                       if e.target == loop_id and e.targetHandle == NodeLoop.INPUT_HANDLE_LOOP)
    idx_end = next(i for i, e in enumerate(all_edges)
                   if e.source == loop_id and e.sourceHandle == NodeLoop.OUTPUT_HANDLE_END)
    pre_loop = all_edges[: idx_list + 1]
    iteration_edges = all_edges[idx_list + 1 : idx_loop_in]
    loop_in_edges = all_edges[idx_loop_in : idx_loop_in + 1]
    end_edges = all_edges[idx_end : idx_end + 1]
    post_loop = all_edges[idx_end + 1 :]

    # Helper to process a single edge
    async def _process_edge(edge: EdgeNodeModel):
        source = nodes.get(edge.source)
        target = nodes.get(edge.target)
        if not source or not target:
            return
        if not source.outputs:
            async for msg in source(chat_log):
                if msg['type'] == 'end':
                    source.outputs[edge.sourceHandle] = msg['content']
                elif msg['type'] == 'content':
                    yield msg['content']
        target.add_parent(source.outputs, edge.sourceHandle, edge.targetHandle)

    # Execute edges up to loop start
    for edge in pre_loop:
        async for out in _process_edge(edge):
            yield out

    loop_node = nodes[loop_id]
    # Prepare items to iterate
    raw = loop_node.inputs.get(NodeLoop.INPUT_HANDLE_LIST)
    if isinstance(raw, str):
        import json as _json

        items = _json.loads(raw)
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError(f"Loop node '{loop_id}' expects a list, got {type(items)}")

    # Identify body nodes for state reset
    body_node_ids = set()
    for e in iteration_edges:
        if e.source != loop_id:
            body_node_ids.add(e.source)
        if e.target != loop_id:
            body_node_ids.add(e.target)
    body_nodes = [nodes[n] for n in body_node_ids]

    # Iterate and process loop body
    for item in items:
        # reset state for each iteration
        loop_node._response = None
        loop_node.outputs.clear()
        for bn in body_nodes:
            bn._response = None
            bn.outputs.clear()
            bn.inputs.clear()
        # inject current item to body edges
        loop_node.outputs[NodeLoop.OUTPUT_HANDLE_ITEM] = loop_node.prep(item)
        for edge in iteration_edges:
            async for out in _process_edge(edge):
                yield out
        # collect iteration result into loop inputs
        for edge in loop_in_edges:
            src = nodes.get(edge.source)
            loop_node.add_parent(src.outputs, edge.sourceHandle, edge.targetHandle)

    # After looping, handle aggregation end
    for edge in end_edges:
        loop_node.outputs.clear()
        loop_node.outputs[edge.sourceHandle] = loop_node.prep(
            loop_node.inputs.get(NodeLoop.INPUT_HANDLE_LOOP, [])
        )
        async for out in _process_edge(edge):
            yield out

    # Continue with remaining edges
    for edge in post_loop:
        async for out in _process_edge(edge):
            yield out


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
    # Detect a Loop node for iterative dynamic execution
    loop_nodes = [nid for nid, node in graph.nodes.items() if isinstance(node, NodeLoop)]
    if loop_nodes:
        async for msg in execute_graph_loop(graph, id_chat=id_chat, id_thread=id_thread, id_user=id_user):
            yield msg
        return
    # Standard execution for acyclic graphs
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
