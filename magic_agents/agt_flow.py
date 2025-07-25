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
    InnerNodeModel,
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
    NodeInner,
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
        ModelAgentFlowTypesModel.CHAT: (NodeChat, None),
        ModelAgentFlowTypesModel.LLM: (NodeLLM, LlmNodeModel),
        ModelAgentFlowTypesModel.END: (NodeEND, None),
        ModelAgentFlowTypesModel.TEXT: (NodeText, TextNodeModel),
        ModelAgentFlowTypesModel.USER_INPUT: (NodeUserInput, UserInputNodeModel),
        ModelAgentFlowTypesModel.PARSER: (NodeParser, ParserNodeModel),
        ModelAgentFlowTypesModel.FETCH: (NodeFetch, FetchNodeModel),
        ModelAgentFlowTypesModel.CLIENT: (NodeClientLLM, ClientNodeModel),
        ModelAgentFlowTypesModel.SEND_MESSAGE: (NodeSendMessage, SendMessageNodeModel),
        ModelAgentFlowTypesModel.LOOP: (NodeLoop, LoopNodeModel),
        ModelAgentFlowTypesModel.INNER: (NodeInner, InnerNodeModel),
        ModelAgentFlowTypesModel.VOID: (NodeEND, None),
    }
    if node_type not in node_map:
        raise ValueError(f"Unsupported node type: {node_type}")
    constructor, model_cls = node_map[node_type]
    if node_type == ModelAgentFlowTypesModel.CHAT:
        return constructor(load_chat=load_chat, **extra, **node_data)
    elif node_type == ModelAgentFlowTypesModel.INNER:
        return constructor(load_chat=load_chat, **extra, data=InnerNodeModel(**extra, **node_data))
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

    loop_id = next(nid for nid, node in nodes.items() if isinstance(node, NodeLoop))
    loop_node = nodes[loop_id]

    async def _process_edge(edge: EdgeNodeModel):
        src = nodes.get(edge.source)
        tgt = nodes.get(edge.target)
        if not src or not tgt:
            return
        
        # Special handling for end edges - always process them to transfer data
        is_end_edge = edge.source == loop_id and edge.sourceHandle == NodeLoop.OUTPUT_HANDLE_END
        
        # For end edges, just transfer existing data without re-executing
        if is_end_edge:
            pass  # Just transfer data below
        else:
            # For loop node sources, always execute if no outputs
            # For other nodes, execute if no outputs OR if response was cleared (indicating it should re-execute)
            should_execute = (edge.source == loop_id and not src.outputs) or \
                            (edge.source != loop_id and (not src.outputs or src._response is None))
            
            if should_execute:
                async for msg in src(chat_log):
                    if msg["type"] == "content":
                        yield msg["content"]
                    elif msg["type"] == "end":
                        src.outputs[edge.sourceHandle] = msg["content"]
                    else:
                        src.outputs[msg["type"]] = msg["content"]
        
        tgt.add_parent(src.outputs, edge.sourceHandle, edge.targetHandle)
        
        # For end edges, execute the target node after adding the input
        if is_end_edge and tgt._response is None:
            async for msg in tgt(chat_log):
                if msg["type"] == "content":
                    yield msg["content"]
                elif msg["type"] == "end":
                    # Find appropriate handle for this output
                    if hasattr(tgt, 'OUTPUT_HANDLE_GENERATED_END'):
                        handle = tgt.OUTPUT_HANDLE_GENERATED_END
                    else:
                        handle = "handle_generated_end"
                    tgt.outputs[handle] = msg["content"]
                else:
                    tgt.outputs[msg["type"]] = msg["content"]

    all_edges = list(graph.edges)
    item_edges = [e for e in all_edges
                  if e.source == loop_id and e.sourceHandle == NodeLoop.OUTPUT_HANDLE_ITEM]
    loop_back_edges = [e for e in all_edges
                       if e.target == loop_id and e.targetHandle == NodeLoop.INPUT_HANDLE_LOOP]
    end_edges = [e for e in all_edges
                 if e.source == loop_id and e.sourceHandle == NodeLoop.OUTPUT_HANDLE_END]
    static_edges = [e for e in all_edges
                    if e not in item_edges + loop_back_edges + end_edges]

    for edge in static_edges:
        async for out in _process_edge(edge):
            yield out

    raw = loop_node.inputs.get(NodeLoop.INPUT_HANDLE_LIST)
    if isinstance(raw, str):
        items = __import__('json').loads(raw)
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError(f"Loop node '{loop_id}' expects a list, got {type(items)}")

    loop_agg = []
    for item in items:
        loop_node._response = None
        loop_node.outputs.clear()
        for node in nodes.values():
            if getattr(node, 'iterate', False):
                node._response = None
                node.outputs.clear()
                # Clear generated content for NodeLLM instances
                if hasattr(node, 'generated'):
                    node.generated = ''

        loop_node.outputs[NodeLoop.OUTPUT_HANDLE_ITEM] = loop_node.prep(str(item))

        for edge in item_edges + loop_back_edges:
            async for out in _process_edge(edge):
                yield out
        fb = loop_node.inputs.get(NodeLoop.INPUT_HANDLE_LOOP)
        loop_agg.append(fb)

    loop_node._response = None
    loop_node.outputs.clear()
    loop_node.outputs[NodeLoop.OUTPUT_HANDLE_END] = loop_node.prep(loop_agg)

    # Clear response for nodes that will receive loop end results so they can re-execute
    for edge in end_edges:
        target_node = nodes.get(edge.target)
        if target_node:
            target_node._response = None
            target_node.outputs.clear()

    for edge in end_edges:
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

    # Track which nodes have been executed
    executed_nodes = set()

    # Helper to check if all dependencies of a node are satisfied
    def are_dependencies_satisfied(node_id: str) -> bool:
        for edge in graph.edges:
            if edge.target == node_id and edge.source not in executed_nodes:
                return False
        return True

    async def process_edge(edge: EdgeNodeModel):
        source_node = nodes.get(edge.source)
        target_node = nodes.get(edge.target)
        if not source_node:
            logger.error(f"Source node {edge.source} not found.")
            return
        if not target_node:
            logger.error(f"Target node {edge.target} not found.")
            return

        # Execute source node only if not already executed and dependencies are satisfied
        if edge.source not in executed_nodes:
            if are_dependencies_satisfied(edge.source):
                if not source_node.outputs:
                    async for item in source_node(chat_log):
                        if item["type"] == "content":
                            yield item["content"]
                        elif item["type"] == "end":
                            source_node.outputs[edge.sourceHandle] = item["content"]
                        else:
                            source_node.outputs[item["type"]] = item["content"]
                executed_nodes.add(edge.source)

        # Pass output to target only if source has been executed
        if edge.source in executed_nodes:
            source_handle = edge.sourceHandle
            target_handle = edge.targetHandle
            target_node.add_parent(source_node.outputs, source_handle, target_handle)

    # Process edges in topological order
    remaining_edges = list(graph.edges)
    while remaining_edges:
        made_progress = False
        for i, edge in enumerate(remaining_edges):
            # Check if this edge can be processed
            if are_dependencies_satisfied(edge.target) or edge.source in executed_nodes:
                async for result in process_edge(edge):
                    yield result
                remaining_edges.pop(i)
                made_progress = True
                break

        if not made_progress:
            # Try to find a node with all dependencies satisfied
            for edge in remaining_edges:
                if edge.source not in executed_nodes and are_dependencies_satisfied(edge.source):
                    async for result in process_edge(edge):
                        yield result
                    made_progress = True
                    break

            if not made_progress:
                # No progress could be made - likely circular dependency
                raise ValueError("Circular dependency detected or missing node in graph")


def build(agt_data, message: str, images: list[str] = None, load_chat=None) -> AgentFlowModel:
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
            if node['type'] == ModelAgentFlowTypesModel.USER_INPUT:
                node['data']['images'] = images
        elif node['type'] == ModelAgentFlowTypesModel.END:
            agt_data['edges'].append({
                "id": uuid.uuid4().hex,
                "source": node['id'],
                "target": void_id
            })
    nodes: Dict[str, Any] = {
        node['id']: create_node(node, load_chat, agt_data.get('debug', False)) for node in agt_data['nodes']
    }
    
    # Build inner graphs for NodeInner nodes
    for node_id, node_instance in nodes.items():
        if isinstance(node_instance, NodeInner):
            # Build the inner graph from the magic_flow dict
            inner_graph = build(
                node_instance.magic_flow,
                message="",  # Will be overridden by the input at runtime
                images=None,
                load_chat=load_chat
            )
            # Set the built graph on the NodeInner instance
            node_instance.inner_graph = inner_graph
    
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
