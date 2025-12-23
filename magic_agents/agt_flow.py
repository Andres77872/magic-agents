"""
Magic Agents Flow Execution Module

This module provides the core execution engine for agent flow graphs.
Uses a reactive event-based execution model for automatic parallel execution.
"""

import logging
import uuid
from typing import Callable, Dict, Any, AsyncGenerator, Optional, Union
from datetime import datetime

from magic_llm.model.ModelChatStream import ChatCompletionModel

from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.debug_feedback import GraphDebugFeedback
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
    BaseNodeModel,
    InnerNodeModel,
    ConditionalNodeModel,
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
    NodeConditional,
    sort_nodes,
)
from magic_agents.execution import (
    execute_graph_reactive,
    execute_graph_loop_reactive,
)
from magic_agents.util.const import HANDLE_VOID
from magic_agents.util.graph_validator import ConditionalEdgeValidator

logger = logging.getLogger(__name__)


def create_node(node: dict, load_chat: Callable, debug: bool = False) -> Any:
    """
    Factory method to create node instances.
    
    JSON is the source of truth - all node configuration comes from JSON.
    Handle names can be customized via data.handles in JSON.

    Args:
        node (dict): Node data from JSON.
        load_chat (Callable): Load chat function.
        debug (bool): Debug mode. Defaults to False.

    Returns:
        Any: Node instance.
    """
    extra = {'debug': debug, 'node_id': node['id'], 'node_type': node['type']}
    node_type = node['type']
    node_data = node.get('data', {})
    
    # Extract handles from JSON data - this allows JSON to override default handle names
    handles = node_data.pop('handles', None)
    if handles:
        extra['handles'] = handles
    
    # Debug - log the raw node definition before instantiation
    logger.debug("Creating node %s of type %s with data %s", node['id'], node_type, node_data)
    
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
        ModelAgentFlowTypesModel.CONDITIONAL: (NodeConditional, ConditionalNodeModel),
        ModelAgentFlowTypesModel.INNER: (NodeInner, InnerNodeModel),
        ModelAgentFlowTypesModel.VOID: (NodeEND, None),
    }
    
    if node_type not in node_map:
        error_msg = f"Unsupported node type: {node_type}"
        logger.error("create_node: %s (node_id=%s)", error_msg, node['id'])
        # Return a stub node that yields an error when executed
        stub = NodeEND(**extra)
        stub._error_info = {
            "error_type": "UnsupportedNodeType",
            "error_message": error_msg,
            "node_id": node['id'],
            "attempted_type": node_type,
            "available_types": list(node_map.keys())
        }
        return stub
    
    constructor, model_cls = node_map[node_type]
    
    if node_type == ModelAgentFlowTypesModel.CHAT:
        return constructor(load_chat=load_chat, **extra, **node_data)
    elif node_type == ModelAgentFlowTypesModel.CONDITIONAL:
        # Validate conditional config using Pydantic model
        try:
            validated = ConditionalNodeModel(**node_data)
            # Pass validated data to constructor
            return constructor(**extra, **validated.model_dump(exclude_none=True))
        except Exception as e:
            logger.error("Invalid conditional node config: %s", e)
            # Return a stub node that reports the validation error
            stub = NodeEND(**extra)
            stub._error_info = {
                "error_type": "ConditionalValidationError",
                "error_message": str(e),
                "node_id": node['id'],
                "node_data": node_data
            }
            return stub
    elif node_type == ModelAgentFlowTypesModel.LOOP:
        # Loop node uses handles for routing configuration
        return constructor(**extra, **node_data)
    elif node_type == ModelAgentFlowTypesModel.INNER:
        return constructor(load_chat=load_chat, **extra, data=InnerNodeModel(**extra, **node_data))
    elif model_cls:
        return constructor(**extra, data=model_cls(**extra, **node_data))
    else:
        return constructor(**extra)


async def execute_graph(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None
) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Execute the agent flow graph asynchronously using reactive event-based model.
    
    This function uses the new reactive executor which enables automatic
    parallel execution of independent nodes based on graph topology.

    Args:
        graph (AgentFlowModel): Agent flow graph.
        id_chat (Optional[Union[int, str]]): Chat ID. Defaults to None.
        id_thread (Optional[Union[int, str]]): Thread ID. Defaults to None.
        id_user (Optional[Union[int, str]]): User ID. Defaults to None.

    Yields:
        AsyncGenerator[ChatCompletionModel, None]: ChatCompletionModel results.
    """
    async for result in execute_graph_reactive(
        graph=graph,
        id_chat=id_chat,
        id_thread=id_thread,
        id_user=id_user
    ):
        yield result


async def execute_graph_loop(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None,
) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Execute an agent flow graph that contains a Loop node using reactive model.
    
    This is now a thin wrapper around the reactive loop executor.
    """
    async for result in execute_graph_loop_reactive(
        graph=graph,
        id_chat=id_chat,
        id_thread=id_thread,
        id_user=id_user
    ):
        yield result


def validate_graph(nodes: list[dict], edges: list[dict]) -> dict:
    """
    Validate the agent flow graph structure.
    
    Args:
        nodes (list[dict]): List of nodes in the graph.
        edges (list[dict]): List of edges in the graph.
    
    Returns:
        dict: Validation result with 'valid' (bool) and 'errors' (list) keys.
    """
    errors = []
    
    # Validation 1: Only ONE NodeUserInput (start node) is allowed
    user_input_nodes = [node for node in nodes if node['type'] == ModelAgentFlowTypesModel.USER_INPUT]
    if len(user_input_nodes) == 0:
        errors.append({
            "error_type": "GraphValidationError",
            "error_message": "Graph must contain exactly one USER_INPUT node (start node). Found: 0",
            "context": {"user_input_nodes_count": 0}
        })
    elif len(user_input_nodes) > 1:
        node_ids = [node['id'] for node in user_input_nodes]
        errors.append({
            "error_type": "GraphValidationError",
            "error_message": f"Graph must contain exactly one USER_INPUT node (start node). Found {len(user_input_nodes)} nodes.",
            "context": {
                "user_input_nodes_count": len(user_input_nodes),
                "node_ids": node_ids
            }
        })
    else:
        logger.info("Validation passed: Found single USER_INPUT node (id=%s)", user_input_nodes[0]['id'])
    
    # Validation 2: Check for duplicate edges (same source, target, and handles)
    edge_signatures = set()
    duplicate_edges = []
    
    for edge in edges:
        edge_signature = (
            edge.get('source'),
            edge.get('target'),
            edge.get('sourceHandle'),
            edge.get('targetHandle')
        )
        if edge_signature in edge_signatures:
            duplicate_edges.append({
                'edge_id': edge.get('id'),
                'source': edge.get('source'),
                'target': edge.get('target'),
                'sourceHandle': edge.get('sourceHandle'),
                'targetHandle': edge.get('targetHandle')
            })
        else:
            edge_signatures.add(edge_signature)
    
    if duplicate_edges:
        error_msg = "Found duplicate edges with same source, target, and handles"
        errors.append({
            "error_type": "GraphValidationError",
            "error_message": error_msg,
            "context": {
                "duplicate_edges": duplicate_edges,
                "duplicate_count": len(duplicate_edges)
            }
        })
    else:
        logger.info("Validation passed: No duplicate edges found (total edges: %d)", len(edges))
    
    # Note: Multiple END nodes are allowed (no validation needed)
    end_nodes = [node for node in nodes if node['type'] == ModelAgentFlowTypesModel.END]
    logger.info("Graph contains %d END node(s) (multiple END nodes are allowed)", len(end_nodes))
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def build(agt_data, message: str, images: list[str] = None, load_chat=None) -> AgentFlowModel:
    """
    Prepare and build the agent flow graph from input data and message.

    Args:
        agt_data: Agent data. Can be either:
            - Flat structure: {'nodes': [...], 'edges': [...], ...}
            - Nested structure: {'content': {'nodes': [...], 'edges': [...]}, ...}
        message (str): Message.
        images (list[str]): Images. Defaults to None.
        load_chat: Load chat function. Defaults to None.

    Returns:
        AgentFlowModel: Agent flow graph. If validation fails, the graph will contain error information.
    """
    # Normalize data structure - handle nested 'content' wrapper
    if 'content' in agt_data and isinstance(agt_data['content'], dict):
        # Nested structure: extract nodes/edges from content
        content = agt_data['content']
        graph_data = {
            'type': agt_data.get('type', 'graph'),
            'debug': agt_data.get('debug', False),
            'debug_config': agt_data.get('debug_config'),
            'nodes': content.get('nodes', []),
            'edges': content.get('edges', []),
        }
        # Copy any additional top-level properties
        for key in agt_data:
            if key not in ('content', 'type', 'debug', 'debug_config'):
                graph_data[key] = agt_data[key]
        agt_data = graph_data
    
    # Validate the graph structure before building
    validation_result = validate_graph(agt_data['nodes'], agt_data['edges'])
    
    # Store validation errors for later (will be set on model after creation)
    validation_errors = None
    if not validation_result['valid']:
        validation_errors = validation_result['errors']
        logger.error("Graph validation failed with %d error(s)", len(validation_errors))
        for err in validation_errors:
            logger.error("  - %s: %s", err['error_type'], err['error_message'])
    
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
                "target": void_id,
                "sourceHandle": "handle_end_output"  # Match NodeEND.DEFAULT_OUTPUT_HANDLE
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
    
    # Convert edges to EdgeNodeModel for validation
    edge_models = [EdgeNodeModel(**e) for e in agt_data['edges']]
    
    # Run conditional-specific validation (after nodes are created)
    conditional_errors = ConditionalEdgeValidator.validate(nodes, edge_models)
    if conditional_errors:
        if validation_errors is None:
            validation_errors = []
        validation_errors.extend(conditional_errors)
        
        # Log warnings separately from errors
        for err in conditional_errors:
            if err.get('severity') == 'warning':
                logger.warning("Conditional validation warning: %s", err['error_message'])
            else:
                logger.error("Conditional validation error: %s", err['error_message'])
    
    agt_data['nodes'] = nodes
    agt = AgentFlowModel(**agt_data)
    
    # Set validation errors on model (private attribute)
    if validation_errors:
        agt._validation_errors = validation_errors
    
    return agt


async def run_agent(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None
) -> AsyncGenerator[ChatCompletionModel, None]:
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
        id_user=id_user
    ):
        yield result
