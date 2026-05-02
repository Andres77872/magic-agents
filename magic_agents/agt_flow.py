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
from magic_agents.models.factory.Nodes import (
    ModelAgentFlowTypesModel,
    LlmNodeModel,
    TextNodeModel,
    ConstantNodeModel,
    UserInputNodeModel,
    ParserNodeModel,
    FetchNodeModel,
    ClientNodeModel,
    SendMessageNodeModel,
    LoopNodeModel,
    BaseNodeModel,
    InnerNodeModel,
    ConditionalNodeModel,
    PythonExecNodeModel,
    McpNodeModel,
    ChatNodeModel,
    HookNodeModel,
)
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system import (
    NodeChat,
    NodeLLM,
    NodeEND,
    NodeText,
    NodeConstant,
    NodeUserInput,
    NodeFetch,
    NodeClientLLM,
    NodeSendMessage,
    NodeParser,
    NodeLoop,
    NodeInner,
    NodeConditional,
    NodePythonExec,
    NodeMcp,
    NodeHook,
    sort_nodes,
)
from magic_agents.execution import (
    execute_graph_reactive,
    execute_graph_loop_reactive,
)
from magic_agents.util.const import HANDLE_VOID
from magic_agents.util.env_resolver import resolve_env_placeholders
from magic_agents.hooks.runtime_config import RuntimeConfig
from magic_agents.hooks.flow_hooks import FlowHooks
from magic_agents.util.graph_validator import (
    ConditionalEdgeValidator,
    validate_edge_handles,
    run_all_validations,
)
from magic_agents.models.factory.AgentFlowModel import (
    AgentFlowModel,
    ContractConfig,
    GraphContractReport,
)

logger = logging.getLogger(__name__)

# Node types that can provide tools to LLM nodes
_TOOL_CAPABLE_TYPES = {ModelAgentFlowTypesModel.FETCH, ModelAgentFlowTypesModel.PYTHON_EXEC, ModelAgentFlowTypesModel.MCP}


# ─── Phase 0 Execution Tree Persistence Callback ────────────────────────────
#
# CallbackEmitter: a module-level registry for api.magic_llm to register
# a persistence callback that receives structured debug events during graph
# execution (GRAPH_START, NODE_START, LLM_GENERATION, ITERATION_START/END,
# SUBGRAPH_START/END, GRAPH_END).
#
# This is strictly additive — existing debug event infrastructure is unchanged.
# The callback is invoked from execute_graph_reactive / execute_graph_loop_reactive
# after each debug event is yielded.
#

class CallbackEmitter:
    """Module-level registry for execution tree persistence callbacks."""
    
    _callbacks: list = []
    
    @classmethod
    def register(cls, callback) -> None:
        """Register a callback that receives (event_type, payload, chat_log)."""
        if callback not in cls._callbacks:
            cls._callbacks.append(callback)
    
    @classmethod
    def unregister(cls, callback) -> None:
        """Unregister a previously registered callback."""
        if callback in cls._callbacks:
            cls._callbacks.remove(callback)
    
    @classmethod
    def emit(cls, event: dict, chat_log: Optional['ModelAgentRunLog'] = None) -> None:
        """Emit an event to all registered callbacks. Non-blocking, exceptions logged."""
        for cb in cls._callbacks:
            try:
                cb(event, chat_log)
            except Exception as exc:
                logger.warning("CallbackEmitter: callback %s raised %s", cb, exc)


# ─── Task Subagents Integration ─────────────────────────────────────────────
#
# ARCHITECTURE CHANGE (Option 1 stricter boundary):
# - magic-llm owns ALL subagent architecture (definitions, loader, registry,
#   binder, bundle, decorator, config, runtime safeguards)
# - magic-agents is PURE usage layer — no local subagents package
# - Subagent loading happens in NodeLLM where MagicLLM client is available
#
# Feature flag for application-level control:
# - ENABLE_TASK_SUBAGENTS (application-level, not repo-level)
# - Repo-level defaults are in magic_llm/agent/config.py
#
# Migration guide:
# - Import from magic_llm.agent: SubagentManifest, ManifestLoader, etc.
# - Use @subagent decorator from magic_llm.agent.decorator
# - Pass code_registry dict to client.load_subagents()
# - NO reset_depths import — handled by magic-llm internally
#

# Feature flag (application-level control)
ENABLE_TASK_SUBAGENTS: bool = False

def enable_task_subagents() -> None:
    """Enable task subagents feature at application level."""
    global ENABLE_TASK_SUBAGENTS
    ENABLE_TASK_SUBAGENTS = True
    logger.debug("Task subagents feature enabled")

def disable_task_subagents() -> None:
    """Disable task subagents feature at application level."""
    global ENABLE_TASK_SUBAGENTS
    ENABLE_TASK_SUBAGENTS = False
    logger.debug("Task subagents feature disabled")

def is_task_subagents_enabled() -> bool:
    """Check if task subagents feature is enabled at application level."""
    return ENABLE_TASK_SUBAGENTS

# Application-level code registry for @subagent decorator population
# NO global mutable state in magic-llm — this dict is passed explicitly
_code_registry: dict = {}

def get_code_registry() -> dict:
    """Get the application-level code registry for decorator population.
    
    This dict is populated by @subagent decorators and passed to
    MagicLLM.load_subagents() at execution time.
    
    NOTE: This is application-level state, NOT repo-level.
    magic-llm uses instance-scoped registry (no global state).
    """
    return _code_registry


def _assign_tool_handles(nodes: list[dict], edges: list[dict]) -> None:
    """Auto-generate unique targetHandle values for tool->LLM edges.

    For each edge where the source node is tool-capable (fetch with tool_mode=true,
    python_exec, mcp) and the target node is an LLM node, assigns a deterministic unique
    targetHandle: handle-tool-definition-0, handle-tool-definition-1, etc.

    Also sets sourceHandle to the source node's resolved output handle so that
    propagate_outputs() can correctly route tool outputs to LLM inputs.

    The counter is per-LLM-node, so each LLM node gets its own sequence.
    targetHandle is only auto-assigned when not already explicit.
    sourceHandle is ALWAYS overwritten for tool-capable edges (backend is authoritative).
    """
    node_map = {n['id']: n for n in nodes}
    tool_counters: dict[str, int] = {}  # LLM node ID -> counter

    for edge in edges:
        target_node = node_map.get(edge.get('target'))
        source_node = node_map.get(edge.get('source'))
        if not (target_node and target_node.get('type') == ModelAgentFlowTypesModel.LLM):
            continue
        if not source_node:
            continue

        source_type = source_node.get('type')
        if source_type not in _TOOL_CAPABLE_TYPES:
            continue

        source_data = source_node.get('data', {})

        # For fetch nodes, only assign tool handles when tool_mode is true
        if source_type == ModelAgentFlowTypesModel.FETCH:
            if not source_data.get('tool_mode', False):
                continue

        # For python_exec nodes in node mode (data.code is set), skip tool
        # handle assignment entirely. Node-mode python_exec edges route from
        # handle-python_exec-result to downstream graph nodes, NOT to LLM tools.
        if source_type == ModelAgentFlowTypesModel.PYTHON_EXEC:
            if source_data.get('code'):
                # Node-mode python_exec: preserve original handles, don't assign tool handles
                continue

        # Resolve the source node's output handle and backfill sourceHandle
        # (always done for tool-capable edges, preserving existing values)
        handles = source_data.get('handles', {})
        if source_type == ModelAgentFlowTypesModel.FETCH:
            # Fetch: output → response → default
            resolved_handle = handles.get('output', handles.get('response', 'handle_fetch_output'))
        elif source_type == ModelAgentFlowTypesModel.MCP:
            # MCP: output → default (handle-tool-definition)
            resolved_handle = handles.get('output', 'handle-tool-definition')
        else:
            # python_exec: output → default
            resolved_handle = handles.get('output', 'handle-tool-definition')
        edge['sourceHandle'] = resolved_handle

        # Only auto-assign targetHandle when not already explicit
        if not edge.get('targetHandle'):
            llm_id = edge['target']
            idx = tool_counters.get(llm_id, 0)
            edge['targetHandle'] = f'handle-tool-definition-{idx}'
            tool_counters[llm_id] = idx + 1


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
        ModelAgentFlowTypesModel.CHAT: (NodeChat, ChatNodeModel),
        ModelAgentFlowTypesModel.LLM: (NodeLLM, LlmNodeModel),
        ModelAgentFlowTypesModel.END: (NodeEND, None),
        ModelAgentFlowTypesModel.TEXT: (NodeText, TextNodeModel),
        ModelAgentFlowTypesModel.CONSTANT: (NodeConstant, ConstantNodeModel),
        ModelAgentFlowTypesModel.USER_INPUT: (NodeUserInput, UserInputNodeModel),
        ModelAgentFlowTypesModel.PARSER: (NodeParser, ParserNodeModel),
        ModelAgentFlowTypesModel.FETCH: (NodeFetch, FetchNodeModel),
        ModelAgentFlowTypesModel.CLIENT: (NodeClientLLM, ClientNodeModel),
        ModelAgentFlowTypesModel.SEND_MESSAGE: (NodeSendMessage, SendMessageNodeModel),
        ModelAgentFlowTypesModel.LOOP: (NodeLoop, LoopNodeModel),
        ModelAgentFlowTypesModel.CONDITIONAL: (NodeConditional, ConditionalNodeModel),
        ModelAgentFlowTypesModel.INNER: (NodeInner, InnerNodeModel),
        ModelAgentFlowTypesModel.VOID: (NodeEND, None),
        ModelAgentFlowTypesModel.PYTHON_EXEC: (NodePythonExec, PythonExecNodeModel),
        ModelAgentFlowTypesModel.MCP: (NodeMcp, McpNodeModel),
        ModelAgentFlowTypesModel.HOOK: (NodeHook, HookNodeModel),
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
    
    if node_type == ModelAgentFlowTypesModel.CONDITIONAL:
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
        # Validate inner config using Pydantic model
        try:
            validated = InnerNodeModel(**node_data)
            return constructor(load_chat=load_chat, **extra, data=validated)
        except Exception as e:
            logger.error("Invalid inner node config: %s", e)
            stub = NodeEND(**extra)
            stub._error_info = {
                "error_type": "InnerNodeValidationError",
                "error_message": str(e),
                "node_id": node['id'],
                "node_data": node_data
            }
            return stub
    elif model_cls:
        # Validate node config using Pydantic model (strict validation)
        try:
            validated = model_cls(**node_data)
            return constructor(**extra, data=validated)
        except Exception as e:
            logger.error("Invalid node config for %s: %s", node_type, e)
            stub = NodeEND(**extra)
            stub._error_info = {
                "error_type": "NodeValidationError",
                "error_message": str(e),
                "node_id": node['id'],
                "node_type": node_type,
                "node_data": node_data
            }
            return stub
    else:
        return constructor(**extra)


async def execute_graph(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None,
    extras: Optional[dict[str, Any]] = None,
    flow_state: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,   # Phase 0: execution tree identity
    parent_run_id: Optional[str] = None,  # Phase 0: parent run identity
    hooks: Optional[RuntimeConfig] = None,  # Phase 8.3: hook runtime config
    debug_callback=None,  # Phase 1: optional async callback for debug events
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
        extras (Optional[dict[str, Any]]): Client-provided contextual data. Defaults to None.
        flow_state (Optional[dict[str, Any]]): Per-flow volatile state (runtime-only). Defaults to None.
        run_id (Optional[str]): Phase 0 execution tree run identity. Defaults to None.
        parent_run_id (Optional[str]): Phase 0 parent run identity. Defaults to None.
        hooks (Optional[RuntimeConfig]): Phase 8.3 hook runtime config.

    Yields:
        AsyncGenerator[ChatCompletionModel, None]: ChatCompletionModel results.
    """
    # Phase 8.3: Create hook registry from runtime config
    # Priority order:
    #   1. RuntimeConfig hooks (global + graph) — full priority
    #   2. AgentFlowModel.hooks (graph-level) — fallback
    #   3. No hooks — backward compatible
    _registry = None
    if hooks is not None and not hooks.is_empty():
        _registry = hooks.create_registry()
        # Also register graph-level hooks from AgentFlowModel if present
        if graph.hooks is not None:
            _registry.register_graph(graph.hooks)
    elif graph.hooks is not None:
        from magic_agents.hooks.hook_registry import HookRegistry
        _registry = HookRegistry()
        _registry.register_graph(graph.hooks)

    async for result in execute_graph_reactive(
        graph=graph,
        id_chat=id_chat,
        id_thread=id_thread,
        id_user=id_user,
        extras=extras,
        flow_state=flow_state,
        run_id=run_id,
        parent_run_id=parent_run_id,
        hooks=_registry,
        debug_callback=debug_callback,
    ):
        yield result


async def execute_graph_loop(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None,
    extras: Optional[dict[str, Any]] = None,
    flow_state: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,         # Phase 0
    parent_run_id: Optional[str] = None,   # Phase 0
    hooks: Optional[RuntimeConfig] = None,  # Phase 8.3: hook runtime config
    debug_callback=None,                   # Phase 1: optional async callback for debug events
) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Execute an agent flow graph that contains a Loop node using reactive model.
    
    This is now a thin wrapper around the reactive loop executor.

    Args:
        graph (AgentFlowModel): Agent flow graph.
        id_chat (Optional[Union[int, str]]): Chat ID. Defaults to None.
        id_thread (Optional[Union[int, str]]): Thread ID. Defaults to None.
        id_user (Optional[Union[int, str]]): User ID. Defaults to None.
        extras (Optional[dict[str, Any]]): Client-provided contextual data. Defaults to None.
        flow_state (Optional[dict[str, Any]]): Per-flow volatile state. Defaults to None.
        run_id (Optional[str]): Phase 0 run identity. Defaults to None.
        parent_run_id (Optional[str]): Phase 0 parent run identity. Defaults to None.
        hooks (Optional[RuntimeConfig]): Phase 8.3 hook runtime config.

    Yields:
        AsyncGenerator[ChatCompletionModel, None]: ChatCompletionModel results.
    """
    # Phase 8.3: Create hook registry from runtime config
    _registry = None
    if hooks is not None and not hooks.is_empty():
        _registry = hooks.create_registry()
    elif graph.hooks is not None:
        from magic_agents.hooks.hook_registry import HookRegistry
        _registry = HookRegistry()
        _registry.register_graph(graph.hooks)

    async for result in execute_graph_loop_reactive(
        graph=graph,
        id_chat=id_chat,
        id_thread=id_thread,
        id_user=id_user,
        extras=extras,
        flow_state=flow_state,
        run_id=run_id,
        parent_run_id=parent_run_id,
        hooks=_registry,
        debug_callback=debug_callback,
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
    
    # Validation 3: Edge connectivity — source and target nodes must exist
    node_ids = {node['id'] for node in nodes}
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        edge_id = edge.get('id', 'unknown')
        
        if source not in node_ids:
            errors.append({
                "error_type": "InvalidEdgeSource",
                "error_message": f"Edge '{edge_id}' references non-existent source node: '{source}'",
                "context": {"edge_id": edge_id, "source": source, "target": target}
            })
        
        if target not in node_ids:
            errors.append({
                "error_type": "InvalidEdgeTarget",
                "error_message": f"Edge '{edge_id}' references non-existent target node: '{target}'",
                "context": {"edge_id": edge_id, "source": source, "target": target}
            })
        
        if source == target:
            errors.append({
                "error_type": "SelfLoopEdge",
                "error_message": f"Edge '{edge_id}' creates a self-loop on node: '{source}'",
                "context": {"edge_id": edge_id, "node_id": source}
            })
    
    # Note: Multiple END nodes are allowed (no validation needed)
    end_nodes = [node for node in nodes if node['type'] == ModelAgentFlowTypesModel.END]
    logger.info("Graph contains %d END node(s) (multiple END nodes are allowed)", len(end_nodes))
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def build(agt_data, message: str, images: list[str] = None, load_chat=None, extras: Optional[dict[str, Any]] = None, history_messages: Optional[list[dict[str, Any]]] = None) -> AgentFlowModel:
    """
    Prepare and build the agent flow graph from input data and message.
    
    CRITICAL FIX: Ensures all edges have unique edge.id for fan-in tracking.
    Edge IDs are assigned before tracker construction.

    Args:
        agt_data: Agent data. Can be either:
            - Flat structure: {'nodes': [...], 'edges': [...], ...}
            - Nested structure: {'content': {'nodes': [...], 'edges': [...]}, ...}
        message (str): Message.
        images (list[str]): Images. Defaults to None.
        load_chat: Load chat function. Defaults to None. (DEPRECATED - backend-authoritative)
        extras (Optional[dict[str, Any]]): Client-provided contextual data that flows 
            through UserInput node to downstream nodes. Defaults to None.
        history_messages (Optional[list[dict[str, Any]]]): Backend-authoritative persisted + runtime
            history messages. Injected into CHAT node data as Slot 1 base. Defaults to None.

    Returns:
        AgentFlowModel: Agent flow graph. If validation fails, the graph will contain error information.
    """
    # NOTE: Subagent loading moved to NodeLLM.process() where MagicLLM client
    # is available. magic-llm's load_subagents() is async and requires a client.
    # Feature flag checked at execution time via is_task_subagents_enabled().
    
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
    
    agt_data = resolve_env_placeholders(agt_data)

    # Validate the graph structure before building
    validation_result = validate_graph(agt_data['nodes'], agt_data['edges'])
    
    # Store validation errors for later (will be set on model after creation)
    validation_errors = None
    if not validation_result['valid']:
        validation_errors = validation_result['errors']
        logger.error("Graph validation failed with %d error(s)", len(validation_errors))
        for err in validation_errors:
            logger.error("  - %s: %s", err['error_type'], err['error_message'])
    
    # Filter out edges with invalid source/target or self-loops before sort_nodes to prevent crashes
    node_ids = {node['id'] for node in agt_data['nodes']}
    valid_edges = [
        e for e in agt_data['edges']
        if e.get('source') in node_ids
        and e.get('target') in node_ids
        and e.get('source') != e.get('target')  # Reject self-loops
    ]
    if len(valid_edges) != len(agt_data['edges']):
        dropped = len(agt_data['edges']) - len(valid_edges)
        logger.warning("Dropped %d edge(s) with invalid references (missing source/target or self-loop)", dropped)
        agt_data['edges'] = valid_edges
    
    # Auto-generate unique targetHandle values for tool->LLM edges
    _assign_tool_handles(agt_data['nodes'], agt_data['edges'])
    
    # CRITICAL: Ensure ALL edges have unique edge.id for fan-in tracking
    # This is the P0 fix - edge.id is the primary key for NodeInputTracker
    for edge in agt_data['edges']:
        if 'id' not in edge or not edge['id']:
            edge['id'] = uuid.uuid4().hex
    
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
                # Pass extras to UserInput node if provided
                if extras is not None:
                    node['data']['extras'] = extras
            elif node['type'] == ModelAgentFlowTypesModel.CHAT:
                # BACKEND-AUTHORITATIVE: Pass history_messages to Chat node
                # Per spec.md: Backend prepares persisted + runtime history (Slot 1)
                # NodeChat reads this from data and uses as base_messages
                if history_messages is not None:
                    node['data']['history_messages'] = history_messages
        elif node['type'] == ModelAgentFlowTypesModel.END:
            # END edges also get unique ID
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
            # Skip if magic_flow is missing or empty
            if not node_instance.magic_flow:
                logger.warning(
                    "NodeInner '%s' has no magic_flow — inner graph will not be built. "
                    "Execution will yield a ConfigurationError.",
                    node_id,
                )
                continue
            
            # Validate magic_flow has required keys before building
            if not isinstance(node_instance.magic_flow, dict):
                logger.warning(
                    "NodeInner '%s' magic_flow is not a dict (type=%s) — inner graph will not be built.",
                    node_id,
                    type(node_instance.magic_flow).__name__,
                )
                continue
            
            required_keys = {'nodes', 'edges'}
            missing_keys = required_keys - set(node_instance.magic_flow.keys())
            if missing_keys:
                logger.warning(
                    "NodeInner '%s' magic_flow is malformed — missing required keys: %s. "
                    "Execution will yield a ConfigurationError.",
                    node_id,
                    sorted(missing_keys),
                )
                continue
            
            # Build the inner graph from the magic_flow dict
            inner_graph = build(
                node_instance.magic_flow,
                message="",  # Will be overridden by the input at runtime
                images=None,
                load_chat=load_chat,
                extras=None  # Child flow starts with isolated extras (will be set at runtime)
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
    
    # Run edge handle validation (clean-break: reject legacy handles)
    handle_errors = validate_edge_handles(nodes, edge_models)
    if handle_errors:
        if validation_errors is None:
            validation_errors = []
        validation_errors.extend(handle_errors)
        
        # Log handle validation errors
        for err in handle_errors:
            if err.get('severity') == 'warning':
                logger.warning("Handle validation warning: %s", err['error_message'])
            else:
                logger.error("Handle validation error: %s", err['error_message'])
    
    agt_data['nodes'] = nodes
    agt = AgentFlowModel(**agt_data)
    
    # Set validation errors on model (private attribute)
    if validation_errors:
        agt._validation_errors = validation_errors
    
    # NEW (Phase 3): Run full contract validation chain and attach report
    # Use contract_config.mode to drive validation behavior (shadow + warn default)
    mode = agt.contract_config.mode
    
    # Skip validation entirely if mode is "off" (rollback path)
    if mode == "off":
        contract_report = GraphContractReport(
            mode=mode,
            edge_count=len(agt.edges),
            node_count=len(agt.nodes),
            diagnostics=[],  # No diagnostics when off
        )
        agt._contract_report = contract_report
        logger.debug("Contract validation disabled (mode=off)")
    else:
        # Run full validation chain
        contract_diagnostics = run_all_validations(agt, mode=mode)
        
        # Create and attach GraphContractReport
        contract_report = GraphContractReport(
            mode=mode,
            edge_count=len(agt.edges),
            node_count=len(agt.nodes),
            diagnostics=contract_diagnostics,
        )
        agt._contract_report = contract_report
        
        # In warn mode: surface diagnostics as warnings
        if mode == "warn":
            for diag in contract_diagnostics:
                severity = diag.get('severity', 'warning')
                if severity == 'error':
                    # In warn mode, errors become warnings (shadow + warn combined)
                    logger.warning("Contract validation: %s", diag.get('error_message', str(diag)))
                elif severity == 'warning':
                    logger.warning("Contract validation: %s", diag.get('error_message', str(diag)))
                elif severity == 'info':
                    logger.debug("Contract info: %s", diag.get('error_message', str(diag)))
        
        # In shadow mode: diagnostics computed but not surfaced
        # (attached to report only, no logging)
    
    return agt


async def run_agent(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None,
    extras: Optional[dict[str, Any]] = None,
    hooks: Optional[RuntimeConfig] = None,
    debug_callback=None,  # Phase 1: optional async callback for debug events
) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Run the agent flow and yield ChatCompletionModel results as they are generated.

    Args:
        graph (AgentFlowModel): Agent flow graph.
        id_chat (Optional[Union[int, str]]): Chat ID. Defaults to None.
        id_thread (Optional[Union[int, str]]): Thread ID. Defaults to None.
        id_user (Optional[Union[int, str]]): User ID. Defaults to None.
        extras (Optional[dict[str, Any]]): Client-provided contextual data. Defaults to None.
        hooks (Optional[RuntimeConfig]): Optional hook runtime config for global hooks.
        debug_callback: Optional async callback for debug events (Phase 1).

    Yields:
        AsyncGenerator[ChatCompletionModel, None]: ChatCompletionModel results.
    """
    async for result in execute_graph(
        graph=graph,
        id_chat=id_chat,
        id_thread=id_thread,
        id_user=id_user,
        extras=extras,
        hooks=hooks,
        debug_callback=debug_callback,
    ):
        yield result
