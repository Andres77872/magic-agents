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
    # Debug ‑ log the raw node definition before instantiation
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
        ModelAgentFlowTypesModel.CONDITIONAL: (NodeConditional, None),
        ModelAgentFlowTypesModel.INNER: (NodeInner, InnerNodeModel),
        ModelAgentFlowTypesModel.VOID: (NodeEND, None),
    }
    if node_type not in node_map:
        error_msg = f"Unsupported node type: {node_type}"
        logger.error("create_node: %s (node_id=%s)", error_msg, node['id'])
        # Return a stub node that yields an error when executed
        # NodeEND is already imported at module level
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
        # Pass condition and other params directly
        return constructor(**extra, **node_data)
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
    # Check for validation errors and yield them as debug messages
    if hasattr(graph, '_validation_errors') and graph._validation_errors:
        for error in graph._validation_errors:
            yield {
                "type": "debug",
                "content": {
                    **error,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        # Still continue execution - let nodes handle their own errors
    
    nodes = graph.nodes
    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app='magic-research'
    )
    logger.info("Starting execute_graph_loop: nodes=%d edges=%d ids(chat=%s, thread=%s, user=%s)",
                len(nodes), len(graph.edges), id_chat, id_thread, id_user)
    
    # Initialize debug feedback if debug mode is enabled
    debug_feedback: Optional[GraphDebugFeedback] = None
    if graph.debug:
        debug_feedback = GraphDebugFeedback(
            execution_id=uuid.uuid4().hex,
            graph_type=graph.type,
            start_time=datetime.utcnow().isoformat()
        )
        logger.info("Debug mode enabled for loop graph execution: %s", debug_feedback.execution_id)

    loop_id = next(nid for nid, node in nodes.items() if isinstance(node, NodeLoop))
    loop_node = nodes[loop_id]

    async def _process_edge(edge: EdgeNodeModel):
        src = nodes.get(edge.source)
        tgt = nodes.get(edge.target)
        if not src or not tgt:
            return
        # Trace loop edge routing
        logger.debug("Loop process_edge edge=%s: %s[%s] -> %s[%s]",
                     edge.id, edge.source, edge.sourceHandle, edge.target, edge.targetHandle)
        
        # Track edge processing in debug mode
        if debug_feedback:
            debug_feedback.add_edge_info(
                source=edge.source,
                target=edge.target,
                source_handle=edge.sourceHandle,
                target_handle=edge.targetHandle
            )
        
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
                logger.debug("Loop executing node %s (%s)", edge.source, src.__class__.__name__)
                async for msg in src(chat_log):
                    if msg["type"] == "content":
                        # Yield content messages with type wrapper
                        yield {
                            "type": "content",
                            "content": msg["content"]["content"]  # Extract actual ChatCompletionModel
                        }
                    elif msg["type"] == "end":
                        src.outputs[edge.sourceHandle] = msg["content"]
                    else:
                        src.outputs[msg["type"]] = msg["content"]
                
                # Yield debug info immediately after node execution if debug mode is enabled
                if debug_feedback and hasattr(src, 'get_debug_info'):
                    node_debug_info = src.get_debug_info()
                    if node_debug_info and node_debug_info.was_executed:
                        yield {
                            "type": "debug",
                            "content": node_debug_info.model_dump()
                        }
        
        logger.debug("Loop passing output %s -> %s (%s -> %s)", edge.source, edge.target, edge.sourceHandle, edge.targetHandle)
        tgt.add_parent(src.outputs, edge.sourceHandle, edge.targetHandle)
        
        # For end edges, execute the target node after adding the input
        if is_end_edge and tgt._response is None:
            async for msg in tgt(chat_log):
                if msg["type"] == "content":
                    # Yield content messages with type wrapper
                    yield {
                        "type": "content",
                        "content": msg["content"]["content"]  # Extract actual ChatCompletionModel
                    }
                elif msg["type"] == "end":
                    # Find appropriate handle for this output
                    if hasattr(tgt, 'OUTPUT_HANDLE_GENERATED_END'):
                        handle = tgt.OUTPUT_HANDLE_GENERATED_END
                    else:
                        handle = "handle_generated_end"
                    tgt.outputs[handle] = msg["content"]
                else:
                    tgt.outputs[msg["type"]] = msg["content"]
            
            # Yield debug info immediately after node execution if debug mode is enabled
            if debug_feedback and hasattr(tgt, 'get_debug_info'):
                node_debug_info = tgt.get_debug_info()
                if node_debug_info and node_debug_info.was_executed:
                    yield {
                        "type": "debug",
                        "content": node_debug_info.model_dump()
                    }

    all_edges = list(graph.edges)
    item_edges = [e for e in all_edges
                  if e.source == loop_id and e.sourceHandle == NodeLoop.OUTPUT_HANDLE_ITEM]
    loop_back_edges = [e for e in all_edges
                       if e.target == loop_id and e.targetHandle == NodeLoop.INPUT_HANDLE_LOOP]
    end_edges = [e for e in all_edges
                 if e.source == loop_id and e.sourceHandle == NodeLoop.OUTPUT_HANDLE_END]
    static_edges = [e for e in all_edges
                    if e not in item_edges + loop_back_edges + end_edges]

    logger.debug("Loop edges: static=%d item=%d loop_back=%d end=%d",
                 len(static_edges), len(item_edges), len(loop_back_edges), len(end_edges))

    for edge in static_edges:
        async for out in _process_edge(edge):
            yield out

    logger.info("Starting execute_graph_loop: loop_id=%s total_edges=%d", loop_id, len(all_edges))
    raw = loop_node.inputs.get(NodeLoop.INPUT_HANDLE_LIST)
    if isinstance(raw, str):
        items = __import__('json').loads(raw)
    else:
        items = raw
    if not isinstance(items, list):
        error_msg = f"Loop node '{loop_id}' expects a list, got {type(items)}"
        logger.error("execute_graph_loop: %s", error_msg)
        yield {
            "type": "debug",
            "content": {
                "node_id": loop_id,
                "node_type": "LOOP",
                "error_type": "ValidationError",
                "error_message": error_msg,
                "context": {
                    "received_type": type(items).__name__,
                    "value_preview": str(items)[:200]
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        return

    logger.info("Loop items to iterate: %d", len(items))
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
    
    # Finalize and yield summary debug information if debug mode is enabled
    if debug_feedback:
        # Collect all node info for summary
        for node_id, node in nodes.items():
            if hasattr(node, 'get_debug_info'):
                node_debug_info = node.get_debug_info()
                # Only include nodes that were executed or bypassed (part of the execution)
                if node_debug_info and (node_debug_info.was_executed or node_debug_info.was_bypassed):
                    debug_feedback.add_node_info(node_debug_info)
        
        # Finalize debug feedback
        debug_feedback.finalize()
        
        # Yield final summary debug feedback
        yield {
            "type": "debug_summary",
            "content": debug_feedback.model_dump()
        }
        
        logger.info(
            "Debug summary: %d nodes (%d executed, %d bypassed, %d failed)",
            debug_feedback.total_nodes,
            debug_feedback.executed_nodes,
            debug_feedback.bypassed_nodes,
            debug_feedback.failed_nodes
        )
    
    logger.info("Finished execute_graph_loop: loop_id=%s", loop_id)


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
    # Check for validation errors and yield them as debug messages
    if hasattr(graph, '_validation_errors') and graph._validation_errors:
        for error in graph._validation_errors:
            yield {
                "type": "debug",
                "content": {
                    **error,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        # Still continue execution - let nodes handle their own errors
    
    # Detect a Loop node for iterative dynamic execution
    loop_nodes = [nid for nid, node in graph.nodes.items() if isinstance(node, NodeLoop)]
    if loop_nodes:
        logger.info("Detected loop nodes in graph: %s. Delegating to execute_graph_loop.", loop_nodes)
        async for msg in execute_graph_loop(graph, id_chat=id_chat, id_thread=id_thread, id_user=id_user):
            yield msg
        return

    # Standard execution for acyclic graphs
    nodes = graph.nodes
    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app='magic-research'
    )
    
    # Initialize debug feedback if debug mode is enabled
    debug_feedback: Optional[GraphDebugFeedback] = None
    if graph.debug:
        debug_feedback = GraphDebugFeedback(
            execution_id=uuid.uuid4().hex,
            graph_type=graph.type,
            start_time=datetime.utcnow().isoformat()
        )
        logger.info("Debug mode enabled for graph execution: %s", debug_feedback.execution_id)

    # Track status of nodes: 'executed' | 'bypassed'
    executed_nodes: set[str] = set()
    node_state: Dict[str, str] = {}

    # Store ids of bypassed edges produced from conditional branching
    bypass_edges: set[str] = set()

    # Convenience predicate helpers
    def is_edge_bypassed(edge: EdgeNodeModel) -> bool:
        return edge.id in bypass_edges

    def mark_edge_bypass(edge: EdgeNodeModel):
        bypass_edges.add(edge.id)

    def propagate_bypass(node_id: str):
        """Recursively mark node and outgoing edges as bypassed if all parents bypassed."""
        if node_state.get(node_id) == "bypassed":
            return
        incoming = [e for e in graph.edges if e.target == node_id]
        if incoming and all(is_edge_bypassed(e) for e in incoming):
            node_state[node_id] = "bypassed"
            # Mark node as bypassed in debug mode
            if debug_feedback and node_id in nodes:
                nodes[node_id].mark_bypassed()
            for e in graph.edges:
                if e.source == node_id:
                    mark_edge_bypass(e)
                    propagate_bypass(e.target)

    # Helper to check if all non-bypassed dependencies of a node are satisfied
    def are_dependencies_satisfied(node_id: str) -> bool:
        for edge in graph.edges:
            if edge.target == node_id and not is_edge_bypassed(edge):
                # Edge still relevant, ensure source executed
                if edge.source not in node_state:
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
        logger.debug("Processing edge=%s: %s[%s] -> %s[%s]",
                     edge.id, edge.source, edge.sourceHandle, edge.target, edge.targetHandle)
        
        # Track edge processing in debug mode
        if debug_feedback:
            debug_feedback.add_edge_info(
                source=edge.source,
                target=edge.target,
                source_handle=edge.sourceHandle,
                target_handle=edge.targetHandle
            )

        # Execute source node only if not already executed and dependencies are satisfied
        # Skip if this edge/path is bypassed
        if is_edge_bypassed(edge):
            return

        if node_state.get(edge.source) not in ("executed", "bypassed"):
            if are_dependencies_satisfied(edge.source):
                if not source_node.outputs:
                    logger.debug("Executing node %s (%s)", edge.source, source_node.__class__.__name__)
                    async for item in source_node(chat_log):
                        if item["type"] == "content":
                            # Yield content messages with type wrapper
                            yield {
                                "type": "content",
                                "content": item["content"]["content"]  # Extract actual ChatCompletionModel
                            }
                        elif item["type"] == "end":
                            source_node.outputs[edge.sourceHandle] = item["content"]
                        else:
                            source_node.outputs[item["type"]] = item["content"]
                
                # Yield debug info immediately after node execution if debug mode is enabled
                if debug_feedback and hasattr(source_node, 'get_debug_info'):
                    node_debug_info = source_node.get_debug_info()
                    if node_debug_info and node_debug_info.was_executed:
                        yield {
                            "type": "debug",
                            "content": node_debug_info.model_dump()
                        }
                
                node_state[edge.source] = "executed"
                executed_nodes.add(edge.source)
                logger.debug("Node %s executed", edge.source)

                # If the node is a Conditional, decide bypass paths
                if isinstance(source_node, NodeConditional):
                    produced = set(source_node.outputs.keys()) - {"end"}
                    selected_handle = next(iter(produced), None)
                    logger.debug("Conditional %s produced handle=%s; bypassing non-selected paths", edge.source, selected_handle)
                    for e in graph.edges:
                        if e.source == edge.source and e.sourceHandle != selected_handle:
                            mark_edge_bypass(e)
                            propagate_bypass(e.target)

        # Pass output to target only if source has been executed
        if node_state.get(edge.source) in ("executed", "bypassed"):
            source_handle = edge.sourceHandle
            target_handle = edge.targetHandle
            logger.debug("Passing output %s -> %s (%s -> %s)", edge.source, edge.target, source_handle, target_handle)
            target_node.add_parent(source_node.outputs, source_handle, target_handle)

    # Process edges in topological order
    remaining_edges = list(graph.edges)
    while remaining_edges:
        made_progress = False
        for i, edge in enumerate(remaining_edges):
            # Remove bypassed edges immediately
            if is_edge_bypassed(edge):
                remaining_edges.pop(i)
                made_progress = True
                break

            # Check if this edge can be processed
            if are_dependencies_satisfied(edge.target) or node_state.get(edge.source) in ("executed", "bypassed"):
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
                logger.error("No progress in graph execution; possible circular dependency or missing node")
                remaining_node_ids = list(set(e.source for e in remaining_edges) | set(e.target for e in remaining_edges))
                yield {
                    "type": "debug",
                    "content": {
                        "error_type": "GraphExecutionError",
                        "error_message": "Circular dependency detected or missing node in graph. No progress could be made.",
                        "context": {
                            "remaining_edges_count": len(remaining_edges),
                            "remaining_node_ids": remaining_node_ids,
                            "executed_nodes": list(executed_nodes),
                            "node_states": node_state
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                return

    # Finalize and yield summary debug information if debug mode is enabled
    if debug_feedback:
        # Collect all node info for summary
        for node_id, node in nodes.items():
            if hasattr(node, 'get_debug_info'):
                node_debug_info = node.get_debug_info()
                # Only include nodes that were executed or bypassed (part of the execution)
                if node_debug_info and (node_debug_info.was_executed or node_debug_info.was_bypassed):
                    debug_feedback.add_node_info(node_debug_info)
        
        # Finalize debug feedback
        debug_feedback.finalize()
        
        # Yield final summary debug feedback
        yield {
            "type": "debug_summary",
            "content": debug_feedback.model_dump()
        }
        
        logger.info(
            "Debug summary: %d nodes (%d executed, %d bypassed, %d failed)",
            debug_feedback.total_nodes,
            debug_feedback.executed_nodes,
            debug_feedback.bypassed_nodes,
            debug_feedback.failed_nodes
        )
    
    logger.info("Finished execute_graph")


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
    # Note: Edges with same source/target but different handles are NOT duplicates
    # as they represent different connections through different ports
    edge_signatures = set()
    duplicate_edges = []
    
    for edge in edges:
        # Create a unique signature for the edge including handles
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
    agt_data: Agent data.
    message (str): Message.
    load_chat: Load chat function. Defaults to None.

    Returns:
    AgentFlowModel: Agent flow graph. If validation fails, the graph will contain error information.
    """
    # Validate the graph structure before building
    validation_result = validate_graph(agt_data['nodes'], agt_data['edges'])
    
    # Store validation errors in agt_data for later retrieval
    if not validation_result['valid']:
        agt_data['_validation_errors'] = validation_result['errors']
        logger.error("Graph validation failed with %d error(s)", len(validation_result['errors']))
        for err in validation_result['errors']:
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
