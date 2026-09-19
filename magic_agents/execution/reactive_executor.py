"""
ReactiveExecutor - Event-based graph execution engine.

This module implements the main execution logic for the reactive
parallel execution model. Nodes execute automatically when their
inputs are ready, enabling natural parallelism based on graph topology.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, UTC

# Loop execution constants
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_ITERATION_TIMEOUT_MS = 30000
DEFAULT_TOTAL_TIMEOUT_MS = 300000

from magic_llm.model.ModelChatStream import ChatCompletionModel

from magic_agents.execution.event_dispatcher import GraphEventDispatcher, NodeState
from magic_agents.execution.conditional_routing import ConditionalRouting
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalSignalTypes
from magic_agents.util.const import SYSTEM_EVENT_STREAMING, SYSTEM_EVENT_DEBUG, SYSTEM_EVENT_DEBUG_SUMMARY, SYSTEM_EVENT_TYPES
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.hooks.runtime_config import RuntimeConfig
from magic_agents.debug.registry import ObserverRegistry
from magic_agents.debug.observer import DebugObserver

logger = logging.getLogger(__name__)


def find_iteration_subgraph(
    loop_id: str,
    nodes: Dict[str, Any],
    edges: List[Any]
) -> Set[str]:
    """
    Find all nodes in the iteration subgraph.
    
    Uses BFS starting from nodes connected to handle_item,
    following edges until reaching handle_loop or the loop node.
    
    Args:
        loop_id: The ID of the loop node
        nodes: Dictionary of all nodes in the graph
        edges: List of all edges in the graph
        
    Returns:
        Set of node IDs that are part of the iteration subgraph
    """
    loop_node = nodes[loop_id]
    item_handle = loop_node.OUTPUT_HANDLE_ITEM
    loop_handle = loop_node.INPUT_HANDLE_LOOP
    end_handle = loop_node.OUTPUT_HANDLE_END
    
    # Find starting nodes (receive handle_item)
    start_nodes = set()
    for edge in edges:
        if edge.source == loop_id and edge.sourceHandle == item_handle:
            start_nodes.add(edge.target)
    
    # BFS to find all reachable nodes
    iteration_nodes = set()
    visited = set()
    queue = list(start_nodes)
    
    while queue:
        node_id = queue.pop(0)
        
        if node_id in visited:
            continue
        visited.add(node_id)
        
        # Don't include the loop node itself
        if node_id == loop_id:
            continue
            
        iteration_nodes.add(node_id)
        
        # Find downstream nodes
        for edge in edges:
            if edge.source == node_id:
                # Stop traversal at loop feedback edge
                if edge.target == loop_id and edge.targetHandle == loop_handle:
                    continue
                # Don't traverse to end-graph nodes via handle_end
                if edge.source == loop_id and edge.sourceHandle == end_handle:
                    continue
                if edge.target not in visited:
                    queue.append(edge.target)
    
    return iteration_nodes


def topological_sort_iteration(
    iteration_nodes: Set[str],
    item_edges: List[Any],
    loop_back_edges: List[Any],
    all_edges: List[Any] = None,
) -> List[str]:
    """
    Sort iteration nodes in execution order using Kahn's algorithm.
    
    Args:
        iteration_nodes: Set of node IDs in the iteration subgraph
        item_edges: Edges from loop handle_item to downstream nodes
        loop_back_edges: Edges that feed back to loop handle_loop
        all_edges: All graph edges (optional) — used to discover internal
            edges between iteration nodes (e.g. conditional branches).
        
    Returns:
        List of node IDs in topological execution order
    """
    # Build adjacency graph for iteration subgraph only
    in_degree = {n: 0 for n in iteration_nodes}
    adjacency = {n: [] for n in iteration_nodes}
    
    relevant_edges = item_edges + loop_back_edges
    for edge in relevant_edges:
        if edge.source in iteration_nodes and edge.target in iteration_nodes:
            adjacency[edge.source].append(edge.target)
            in_degree[edge.target] += 1
    
    # Also include edges between iteration nodes from the full edge list
    # (e.g. conditional → branch edges that aren't item_edges or loop_back_edges)
    if all_edges is not None:
        for edge in all_edges:
            if (edge.source in iteration_nodes and edge.target in iteration_nodes
                    and edge not in relevant_edges):
                adjacency[edge.source].append(edge.target)
                in_degree[edge.target] += 1
    
    # Also consider edges where source is NOT in iteration_nodes but target is
    # (these are the entry points from handle_item)
    for edge in item_edges:
        if edge.source not in iteration_nodes and edge.target in iteration_nodes:
            # Target has no in-degree from other iteration nodes
            pass  # in_degree is already 0 from initialization
    
    # Kahn's algorithm
    queue = [n for n in iteration_nodes if in_degree[n] == 0]
    result = []
    
    while queue:
        node = queue.pop(0)
        result.append(node)
        
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If result doesn't contain all nodes, there's a cycle (shouldn't happen)
    if len(result) != len(iteration_nodes):
        logger.warning(
            "Topological sort incomplete: got %d of %d nodes",
            len(result), len(iteration_nodes)
        )
        # Add missing nodes at the end
        for n in iteration_nodes:
            if n not in result:
                result.append(n)
    
    return result


def prepare_item_output(item: Any, index: int) -> Dict[str, Any]:
    """
    Prepare item for output preserving type information.
    
    Args:
        item: The item to prepare (can be any type)
        index: The iteration index
        
    Returns:
        Dictionary with node info, content, index, and type metadata
    """
    return {
        "node": "NodeLoop",
        "content": item,  # Original type preserved
        "index": index,
        "type": type(item).__name__
    }


def emit_loop_progress(
    loop_id: str,
    current_index: int,
    total_items: int,
    item: Any,
    elapsed_ms: float
) -> Dict:
    """
    Emit progress event for loop iteration.
    
    Args:
        loop_id: The loop node ID
        current_index: Current iteration index (0-based)
        total_items: Total number of items to iterate
        item: Current item being processed
        elapsed_ms: Time elapsed since loop started in milliseconds
        
    Returns:
        Progress event dictionary
    """
    def estimate_remaining(current: int, total: int, elapsed: float) -> float:
        """Estimate remaining time based on current progress."""
        if current == 0:
            return 0
        avg_per_item = elapsed / (current + 1)
        remaining_items = total - current - 1
        return round(avg_per_item * remaining_items, 2)
    
    return {
        "type": "loop_progress",
        "content": {
            "loop_id": loop_id,
            "current": current_index,
            "total": total_items,
            "progress": round((current_index + 1) / total_items * 100, 1) if total_items > 0 else 0,
            "item_preview": str(item)[:100],
            "elapsed_ms": round(elapsed_ms, 2),
            "estimated_remaining_ms": estimate_remaining(current_index, total_items, elapsed_ms)
        }
    }


def reset_iteration_nodes(nodes: Dict[str, Any], iteration_nodes: Set[str]) -> None:
    """
    Reset all nodes in the iteration subgraph for a new iteration.
    
    Args:
        nodes: Dictionary of all nodes
        iteration_nodes: Set of node IDs to reset
    """
    for nid in iteration_nodes:
        node = nodes.get(nid)
        if node:
            node._response = None
            node.outputs.clear()
            # Clear input from previous iteration loop feedback
            if hasattr(node, 'inputs'):
                # Preserve non-loop inputs but allow them to be overwritten
                pass
            if hasattr(node, 'generated'):
                node.generated = ''


def _wire_hooks_to_registry(
    registry: HookRegistry,
    runtime_config: Optional[RuntimeConfig],
    graph: AgentFlowModel,
    id_chat: str,
    id_thread: str,
    id_user: str,
) -> None:
    """Wire persistence and debug hooks into existing HookRegistry.

    P1-3 + P1-4: Registers GraphPersistenceHook and DebugSSEHook as global-tier
    hooks on the existing HookRegistry. No new hook manager is created.

    Graph override takes precedence over RuntimeConfig for persistence:
      - graph.persistence_enabled=False → skip persistence entirely
      - graph.persistence_enabled=True → use graph-level value
      - graph.persistence_enabled=None (default) → fall back to RuntimeConfig

    Args:
        registry: Existing HookRegistry instance (must have execution_id/run_id set).
        runtime_config: Optional RuntimeConfig with persistence/SSE configuration.
        graph: AgentFlowModel with optional persistence_enabled override.
        id_chat: Chat/session identity.
        id_thread: Thread identity.
        id_user: User identity.
    """
    if runtime_config is None:
        return

    # === GraphPersistenceHook (P1-3) ===
    # Graph override wins if set; otherwise fall back to RuntimeConfig
    graph_persistence = getattr(graph, 'persistence_enabled', None)
    if graph_persistence is False:
        # Graph explicitly disabled persistence — skip entirely
        pass
    elif graph_persistence is True or runtime_config.is_persistence_enabled():
        hook = runtime_config.build_persistence_hook(
            id_chat=id_chat,
            id_thread=id_thread,
            id_user=id_user,
        )
        if hook is not None:
            registry.register_global(hook)

    # === DebugSSEHook (P1-4) ===
    if runtime_config.is_debug_sse_enabled():
        hook = runtime_config.build_debug_sse_hook(id_chat=id_chat)
        if hook is not None:
            registry.register_global(hook)


async def execute_graph_reactive(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None,
    extras: Optional[dict[str, Any]] = None,
    flow_state: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,         # Phase 0: execution tree identity
    parent_run_id: Optional[str] = None,   # Phase 0: parent run identity
    hooks: Optional[HookRegistry] = None,  # Phase 4: hook registry for graph/node hooks
    runtime_config: Optional[RuntimeConfig] = None,  # Phase 4: runtime config for persistence/SSE auto-wiring
    debug_callback = None,  # Phase 1: optional async callback for debug events
) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Execute graph using reactive event-based model.
    
    Nodes execute automatically when all their inputs are ready.
    Parallel execution happens naturally based on the graph topology.
    
    Args:
        graph: The agent flow graph to execute
        id_chat: Optional chat ID
        id_thread: Optional thread ID
        id_user: Optional user ID
        extras: Optional client-provided contextual data (for consistency with entry points)
        flow_state: Optional per-flow volatile state (runtime-only, never persisted)
        run_id: Optional Phase 0 run identity for execution tree persistence
        parent_run_id: Optional Phase 0 parent run identity
        hooks: Optional HookRegistry for graph/node lifecycle hooks (Phase 4)
        runtime_config: Optional RuntimeConfig for persistence/SSE auto-wiring (Phase 4)
        
    Yields:
        Streaming content and final outputs from nodes
    """
    # Check for validation errors — fail fast on blocking errors before starting execution
    if hasattr(graph, '_validation_errors') and graph._validation_errors:
        # Only block on structural graph errors that make execution impossible.
        # Conditional routing errors (MissingConditionalEdge, etc.) are handled
        # at runtime via bypass propagation and should NOT block execution.
        blocking_types = {'GraphValidationError', 'HookValidationError'}
        blocking_errors = [
            e for e in graph._validation_errors
            if e.get('error_type') in blocking_types
            or e.get('type') in blocking_types
        ]
        for error in graph._validation_errors:
            yield {
                "type": SYSTEM_EVENT_DEBUG,
                "content": {
                    **error,
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
        if blocking_errors:
            logger.error("Aborting execution: %d blocking validation error(s)", len(blocking_errors))
            return

    # Phase 4: Wire persistence and debug hooks into existing HookRegistry.
    # Must happen BEFORE the loop detection (which may early-return to loop executor)
    # so that loop graphs also receive auto-wired hooks on the shared registry.
    if hooks is not None and runtime_config is not None:
        _wire_hooks_to_registry(
            registry=hooks,
            runtime_config=runtime_config,
            graph=graph,
            id_chat=str(id_chat) if id_chat is not None else '',
            id_thread=str(id_thread) if id_thread is not None else '',
            id_user=str(id_user) if id_user is not None else '',
        )

    # Detect loop nodes - delegate to loop handler
    from magic_agents.node_system import NodeLoop
    loop_nodes = [nid for nid, node in graph.nodes.items() if isinstance(node, NodeLoop)]
    if loop_nodes:
        logger.info("Detected loop nodes: %s. Delegating to loop executor.", loop_nodes)
        async for msg in execute_graph_loop_reactive(
            graph, id_chat=id_chat, id_thread=id_thread, id_user=id_user,
            extras=extras, flow_state=flow_state,
            run_id=run_id, parent_run_id=parent_run_id,
            hooks=hooks, debug_callback=debug_callback,
        ):
            yield msg
        return
    
    nodes = graph.nodes
    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app=getattr(graph, 'app_id', None) or getattr(graph, 'id_app', None),
        flow_state=flow_state or {},  # Initialize per-flow volatile state (isolated per flow)
        run_id=run_id,                  # Phase 0: execution tree identity
        parent_run_id=parent_run_id,    # Phase 0: parent run identity
    )
    
    # Inject extras into UserInput nodes if provided (for run_agent(graph, extras=...) path)
    # This handles the case where a graph was built without extras but run_agent passes extras
    from magic_agents.node_system import NodeUserInput
    if extras is not None:
        for node_id, node in nodes.items():
            if isinstance(node, NodeUserInput):
                # Update UserInput node's extras if it wasn't set during build
                if node._extras is None:
                    node._extras = extras
                    logger.debug("Injected extras into UserInput node '%s'", node_id)
    
    logger.info(
        "Starting reactive execution: nodes=%d edges=%d",
        len(nodes), len(graph.edges)
    )
    
    # Phase 0: emit GRAPH_START event for persistence callback
    _graph_start_event = {
        "type": SYSTEM_EVENT_DEBUG,
        "content": {
            "event_type": "GRAPH_START",
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "graph_type": graph.type,
            "node_count": len(nodes),
            "edge_count": len(graph.edges),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    }
    yield _graph_start_event
    from magic_agents.agt_flow import CallbackEmitter
    CallbackEmitter.emit(_graph_start_event, chat_log)
    
    # Generate execution ID for traceability (used by hooks and debug feedback)
    _execution_id = uuid.uuid4().hex

    # Phase 4: Set execution identity on registry so Node.__call__ and
    # HookRelay can access real execution_id/run_id for HookContext construction.
    if hooks is not None:
        hooks.execution_id = _execution_id
        hooks.run_id = run_id or ''

    # === HOOK: on_graph_start (AFTER validation, BEFORE task creation, Phase 4) ===
    _graph_hook_context = None
    _graph_has_errors = False  # Track whether any node errored (for on_graph_error)
    if hooks is not None and not hooks.is_empty():
        from magic_agents.hooks.context_factory import HookContextFactory
        _graph_hook_context = HookContextFactory.build_graph_context(
            execution_id=_execution_id,
            run_id=run_id or '',
            metadata={
                "graph_id": getattr(graph, 'id', '') or '',
                "graph_type": graph.type,
                "node_count": len(nodes),
                "edge_count": len(graph.edges),
            },
        )
        await hooks.invoke("on_graph_start", _graph_hook_context)

    # Initialize observer registry (replaces inline GraphDebugFeedback)
    _debug_enabled_global = os.environ.get('DEBUG_ENABLED', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
    _resolved_debug_config = getattr(graph, 'resolved_debug_config', None)
    observer_registry = ObserverRegistry.create(
        debug_enabled_global=_debug_enabled_global,
        graph_debug=graph.debug,
        graph_debug_config=_resolved_debug_config,
        execution_id=_execution_id,
        graph_type=graph.type,
        total_nodes=len(nodes),
        total_edges=len(graph.edges),
        callback=debug_callback,
    )
    
    # Capture execution start time for duration measurement
    _exec_start_time = datetime.now(UTC)
    
    # OBSERVER: on_graph_start (executor-owned hook)
    if observer_registry.is_active:
        await observer_registry.graph_observer.on_graph_start(
            graph_type=graph.type,
            execution_id=_execution_id,
            node_count=len(nodes),
            edge_count=len(graph.edges),
        )
    
    # Create event dispatcher with graph-level timeout
    dispatcher = GraphEventDispatcher(
        nodes, graph.edges,
        timeout=graph.timeout,
        execution_id=_execution_id,
        run_id=run_id or '',
    )
    
    # Output queue for collecting results from parallel tasks
    output_queue: asyncio.Queue = asyncio.Queue()

    def _bypassed_node_ids() -> set[str]:
        """Snapshot nodes already marked bypassed by the dispatcher."""
        return {
            node_id
            for node_id in nodes
            if dispatcher.get_state(node_id) == NodeState.BYPASSED
        }

    async def _notify_conditional_bypasses(previously_bypassed: set[str]) -> None:
        """Emit observer/hook notifications for newly bypassed static nodes.

        Conditional propagation marks dispatcher state before waiting node tasks
        resume. Those tasks intentionally return early when they see BYPASSED,
        so the selecting node owns the single lifecycle notification.
        """
        newly_bypassed = _bypassed_node_ids() - previously_bypassed
        for bypassed_id in sorted(newly_bypassed):
            bypassed_node = nodes[bypassed_id]
            node_type = getattr(bypassed_node, 'node_type', 'unknown') or 'unknown'
            node_class = type(bypassed_node).__name__

            if observer_registry.is_active:
                bypass_observer = observer_registry.observer_for(
                    bypassed_id, bypassed_node
                )
                await bypass_observer.on_node_bypass(
                    node_id=bypassed_id,
                    node_type=node_type,
                    node_class=node_class,
                    reason="condition",
                )

            if hooks is not None and not hooks.is_empty():
                from magic_agents.hooks.context_factory import HookContextFactory

                bypass_context = HookContextFactory.build_bypass_context(
                    execution_id=_execution_id,
                    run_id=run_id or '',
                    node_id=bypassed_id,
                    node_type=node_type,
                    node_class=node_class,
                    reason="condition",
                    metadata={"bypass_path": "conditional", "phase": "static"},
                )
                await hooks.invoke(
                    "on_node_bypass", bypass_context, reason="condition"
                )
    
    async def execute_single_node(node_id: str):
        """Execute a single node when ready."""
        nonlocal _graph_has_errors
        node = nodes[node_id]
        tracker = dispatcher.get_tracker(node_id)
        
        if not tracker:
            logger.error("No tracker for node %s", node_id)
            return
        
        try:
            # Wait for all inputs
            should_execute = await tracker.wait_ready(timeout=dispatcher.timeout)
            
            if not should_execute:
                # GUARD: If error cascade already bypassed this node via
                # _propagate_error_bypass, the dispatcher state
                # is already BYPASSED and the hook was already fired with
                # reason="upstream_error". Skip redundant not_ready hook.
                if dispatcher.get_state(node_id) == NodeState.BYPASSED:
                    return
                
                # Node is bypassed
                dispatcher.set_state(node_id, NodeState.BYPASSED)
                node.mark_bypassed()
                logger.debug("Node %s bypassed", node_id)
                
                # Notify observer (executor-owned hook)
                if observer_registry.is_active:
                    _bypass_observer = observer_registry.observer_for(node_id, node)
                    await _bypass_observer.on_node_bypass(
                        node_id=node_id,
                        node_type=getattr(node, 'node_type', 'unknown') or 'unknown',
                        node_class=type(node).__name__,
                        reason="inputs_not_ready",
                    )
                
                # HOOK: on_node_bypass (Phase 4) — reason="not_ready"
                if hooks is not None and not hooks.is_empty():
                    from magic_agents.hooks.context_factory import HookContextFactory
                    _bypass_ctx = HookContextFactory.build_bypass_context(
                        execution_id=_execution_id,
                        run_id=run_id or '',
                        node_id=node_id,
                        node_type=getattr(node, 'node_type', 'unknown') or 'unknown',
                        node_class=type(node).__name__,
                        reason="not_ready",
                        metadata={"bypass_path": "single_node", "phase": "static"},
                    )
                    await hooks.invoke("on_node_bypass", _bypass_ctx, reason="not_ready")
                return
            
            # Execute the node
            dispatcher.set_state(node_id, NodeState.EXECUTING)
            logger.debug("Executing node %s (%s)", node_id, node.__class__.__name__)
            
            conditional_selected_handle: Optional[str] = None
            bypass_all_signaled = False

            # Resolve observer for this node (allows per-node specialization)
            _node_observer = observer_registry.observer_for(node_id, node) if observer_registry.is_active else None

            async for item in node(chat_log, hooks=hooks, observer=_node_observer):
                item_type = item.get("type", "")
                
                # Check if this is a streaming content event (for immediate output)
                # Nodes can use any handle name for streaming - check the output handle configuration
                is_streaming = False
                if hasattr(node, 'OUTPUT_HANDLE_CONTENT'):
                    is_streaming = item_type == node.OUTPUT_HANDLE_CONTENT
                elif item_type == SYSTEM_EVENT_STREAMING:
                    is_streaming = True
                
                if is_streaming:
                    # Queue streaming content for immediate output
                    await output_queue.put({
                        "type": SYSTEM_EVENT_STREAMING,
                        "content": item["content"]["content"],
                        "source_node": node_id
                    })
                elif item_type == SYSTEM_EVENT_DEBUG:
                    # Queue debug info (legacy path — Node may still yield debug events
                    # for backward compatibility; these are forwarded through the queue)
                    await output_queue.put(item)
                elif ConditionalSignalTypes.is_system_signal(item_type):
                    # Track BYPASS_ALL for post-loop handling
                    if item_type == ConditionalSignalTypes.BYPASS_ALL:
                        bypass_all_signaled = True
                    logger.debug("Node %s emitted system signal: %s", node_id, item_type)
                else:
                    # Handle-specific output (conditional routing, etc.)
                    node.outputs[item_type] = item["content"]
                    # Track conditional selection (only non-system signals)
                    if isinstance(node, ConditionalRouting) and conditional_selected_handle is None:
                        if item_type not in (SYSTEM_EVENT_DEBUG, SYSTEM_EVENT_DEBUG_SUMMARY):
                            conditional_selected_handle = item_type
            
            # Mark completed
            dispatcher.set_state(node_id, NodeState.COMPLETED)
            logger.debug("Node %s completed", node_id)
            
            # Propagate outputs to downstream nodes
            await dispatcher.propagate_outputs(node_id, node.outputs)
            
            # Handle BYPASS_ALL from any node (conditional or non-conditional)
            if bypass_all_signaled:
                previously_bypassed = _bypassed_node_ids()
                await dispatcher.handle_bypass_all_signal(node_id)
                await _notify_conditional_bypasses(previously_bypassed)
            # Handle conditional bypass propagation (skip if BYPASS_ALL was already handled)
            elif isinstance(node, ConditionalRouting):
                selected_handle = conditional_selected_handle or getattr(node, 'selected_handle', None)
                if selected_handle:
                    # Verify edge exists for selected handle
                    outgoing = [e for e in graph.edges if e.source == node_id]
                    has_matching_edge = any(e.sourceHandle == selected_handle for e in outgoing)
                    
                    if not has_matching_edge:
                        # Selected handle has no matching edge — this is a routing error.
                        # The conditional emitted output on a handle that no downstream node
                        # is listening to. All downstream nodes must be bypassed to prevent
                        # them from hanging forever waiting for data that will never arrive.
                        #
                        # NOTE: We do NOT fall back to default_handle here. The default_handle
                        # is designed for when the condition evaluates to EMPTY (handled in
                        # NodeConditional.process()), not for when it evaluates to a
                        # non-existent handle. Falling back to default_handle would leave
                        # nodes on the default path in selected_targets (not bypassed) but
                        # without data, causing an indefinite hang.
                        await output_queue.put(node.yield_debug_error(
                            error_type="GraphRoutingError",
                            error_message=f"Conditional selected handle '{selected_handle}', but no outgoing edge matches.",
                            context={
                                "selected_handle": selected_handle,
                                "outgoing_handles": [e.sourceHandle for e in outgoing],
                                "node_id": node_id,
                                "default_handle": getattr(node, 'default_handle', None),
                                "suggestion": "Ensure the condition template evaluates to a handle name that has a corresponding outgoing edge."
                            }
                        ))
                        previously_bypassed = _bypassed_node_ids()
                        await dispatcher.handle_bypass_all_signal(node_id)
                        await _notify_conditional_bypasses(previously_bypassed)
                    else:
                        previously_bypassed = _bypassed_node_ids()
                        await dispatcher.propagate_conditional_bypass(node_id, selected_handle)
                        await _notify_conditional_bypasses(previously_bypassed)
        
        except asyncio.TimeoutError:
            dispatcher.set_state(node_id, NodeState.ERROR)
            _graph_has_errors = True
            logger.error("Node %s timed out", node_id)
            await output_queue.put({
                "type": SYSTEM_EVENT_DEBUG,
                "content": {
                    "node_id": node_id,
                    "error_type": "TimeoutError",
                    "error_message": f"Node timed out waiting for inputs after {dispatcher.timeout}s",
                    "timestamp": datetime.now(UTC).isoformat()
                }
            })
            await _propagate_error_bypass(node_id)
        
        except Exception as e:
            dispatcher.set_state(node_id, NodeState.ERROR)
            _graph_has_errors = True
            logger.error("Node %s failed: %s", node_id, str(e))
            await output_queue.put({
                "type": SYSTEM_EVENT_DEBUG,
                "content": {
                    "node_id": node_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.now(UTC).isoformat()
                }
            })
            await _propagate_error_bypass(node_id)
    
    async def _propagate_error_bypass(failed_node_id: str):
        """Always bypass downstream nodes; notify lifecycle hooks when present.
        
        Routing correctness cannot depend on whether a FlowHooks registry was
        supplied. Declarative edge hooks also rely on this cascade to release
        their virtual input dependency when the source edge cannot traverse.
        
        Args:
            failed_node_id: The node that encountered an error.
        """
        bypassed_nids = await dispatcher.propagate_error_bypass(failed_node_id)
        if hooks is None or hooks.is_empty():
            return

        from magic_agents.hooks.context_factory import HookContextFactory
        for bid in bypassed_nids:
            bnode = nodes.get(bid)
            bypass_ctx = HookContextFactory.build_bypass_context(
                execution_id=_execution_id,
                run_id=run_id or '',
                node_id=bid,
                node_type=bnode.node_type if bnode else None,
                node_class=bnode.__class__.__name__ if bnode else None,
                reason="upstream_error",
                metadata={"upstream_error_node": failed_node_id},
            )
            await hooks.invoke("on_node_bypass", bypass_ctx, reason="upstream_error")

    # Create tasks for all nodes - they will wait for their inputs
    tasks: Dict[str, asyncio.Task] = {}
    for node_id in nodes.keys():
        task = asyncio.create_task(
            execute_single_node(node_id),
            name=f"node_{node_id}"
        )
        tasks[node_id] = task
    
    # Signal when all tasks complete
    all_done = asyncio.Event()
    
    async def wait_for_tasks():
        """Wait for all node tasks to complete."""
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        all_done.set()
        # Signal queue that no more items will be added
        await output_queue.put(None)
    
    # Start task waiter
    waiter = asyncio.create_task(wait_for_tasks())
    
    # Yield results as they arrive
    while True:
        try:
            item = await asyncio.wait_for(output_queue.get(), timeout=1.0)
            if item is None:
                break
            yield item
        except asyncio.TimeoutError:
            # Check if all done
            if all_done.is_set():
                # Drain remaining queue items
                while not output_queue.empty():
                    item = output_queue.get_nowait()
                    if item is not None:
                        yield item
                break
    
    # Wait for waiter task
    await waiter
    
    # === HOOK: on_graph_end / on_graph_error (AFTER all tasks complete, BEFORE return, Phase 4) ===
    # Spec requirement: on_graph_end fires for successful execution only.
    # on_graph_error fires when any node errored; on_graph_end is NOT invoked for failures.
    if _graph_hook_context is not None:
        _graph_hook_context.timestamp = datetime.now(UTC)
        _summary = dispatcher.get_execution_summary()
        _graph_hook_context.metadata["execution_summary"] = _summary
        if _graph_has_errors:
            _graph_hook_context.error_message = (
                f"Graph execution completed with {_summary['errors']} node error(s)"
            )
            _graph_hook_context.metadata["failed_nodes"] = _summary["states"].get("error", [])
            await hooks.invoke("on_graph_error", _graph_hook_context, error=RuntimeError(f"Graph execution failed: {_summary['errors']} node error(s)"))
        else:
            await hooks.invoke("on_graph_end", _graph_hook_context)
    
    # Finalize observer — emit graph_end event and summary
    _exec_end_time = datetime.now(UTC)
    _total_duration = (_exec_end_time - _exec_start_time).total_seconds() * 1000
    _summary = dispatcher.get_execution_summary()
    
    if observer_registry.is_active:
        await observer_registry.graph_observer.on_graph_end(
            graph_type=graph.type,
            execution_id=_execution_id,
            total_duration_ms=_total_duration,
            node_count=len(nodes),
            executed_count=_summary.get("completed", 0),
            bypassed_count=_summary.get("bypassed", 0),
            failed_count=_summary.get("errors", 0),
        )
        
        # Phase 0: emit GRAPH_END to persistence callback (separate from observer)
        from magic_agents.agt_flow import CallbackEmitter
        _graph_end_event = {
            "type": SYSTEM_EVENT_DEBUG,
            "content": {
                "event_type": "GRAPH_END",
                "run_id": run_id,
                "execution_id": _execution_id,
                "total_duration_ms": _total_duration,
                "status": "completed" if not _graph_has_errors else "errors",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        }
        CallbackEmitter.emit(_graph_end_event, chat_log)
    
    logger.info(
        "Execution complete: %d completed, %d bypassed, %d errors",
        _summary.get("completed", 0),
        _summary.get("bypassed", 0),
        _summary.get("errors", 0),
    )
    
    logger.info("Finished reactive execution")


async def execute_graph_loop_reactive(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None,
    extras: Optional[dict[str, Any]] = None,
    flow_state: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,         # Phase 0: execution tree identity
    parent_run_id: Optional[str] = None,   # Phase 0: parent run identity
    hooks: Optional[HookRegistry] = None,  # Phase 4: hook registry for graph/node hooks
    runtime_config: Optional[RuntimeConfig] = None,  # Direct-loop persistence/SSE auto-wiring
    debug_callback=None,                   # Phase 1: optional async callback for debug events
) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Execute an agent flow graph containing a Loop node using reactive model.
    
    This handles the special iteration semantics of loop nodes while still
    enabling parallel execution within each iteration.
    
    Args:
        graph: The agent flow graph to execute
        id_chat: Optional chat ID
        id_thread: Optional thread ID
        id_user: Optional user ID
        extras: Optional client-provided contextual data
        flow_state: Optional per-flow volatile state (runtime-only)
        hooks: Optional HookRegistry for graph/node lifecycle hooks (Phase 4)
        
    Yields:
        Streaming content and final outputs from nodes
    """
    import json
    from magic_agents.node_system import NodeLoop
    
    # Check for validation errors — fail fast on blocking errors before starting execution
    if hasattr(graph, '_validation_errors') and graph._validation_errors:
        # Only block on structural graph errors that make execution impossible.
        # Conditional routing errors (MissingConditionalEdge, etc.) are handled
        # at runtime via bypass propagation and should NOT block execution.
        blocking_types = {'GraphValidationError', 'HookValidationError'}
        blocking_errors = [
            e for e in graph._validation_errors
            if e.get('error_type') in blocking_types
            or e.get('type') in blocking_types
        ]
        for error in graph._validation_errors:
            yield {
                "type": SYSTEM_EVENT_DEBUG,
                "content": {
                    **error,
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
        if blocking_errors:
            logger.error("Aborting loop execution: %d blocking validation error(s)", len(blocking_errors))
            return
    
    nodes = graph.nodes
    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app=getattr(graph, 'app_id', None) or getattr(graph, 'id_app', None),
        flow_state=flow_state or {},  # Initialize per-flow volatile state (isolated per flow)
        run_id=run_id,                  # Phase 0: execution tree identity
        parent_run_id=parent_run_id,    # Phase 0: parent run identity
    )
    
    # Generate execution ID for hooks traceability
    _execution_id = uuid.uuid4().hex

    # ``execute_graph_reactive`` wires before delegating and intentionally does
    # not pass runtime_config here.  The public direct-loop wrapper does pass it,
    # so this branch provides identical auto-wiring without duplicate hooks.
    if hooks is not None and runtime_config is not None:
        _wire_hooks_to_registry(
            registry=hooks,
            runtime_config=runtime_config,
            graph=graph,
            id_chat=str(id_chat) if id_chat is not None else '',
            id_thread=str(id_thread) if id_thread is not None else '',
            id_user=str(id_user) if id_user is not None else '',
        )

    if hooks is not None:
        hooks.execution_id = _execution_id
        hooks.run_id = run_id or ''

    logger.info(
        "Starting reactive loop execution: nodes=%d edges=%d",
        len(nodes), len(graph.edges)
    )
    
    # Phase 0: emit GRAPH_START event for persistence callback
    _graph_start_event = {
        "type": SYSTEM_EVENT_DEBUG,
        "content": {
            "event_type": "GRAPH_START",
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "graph_type": graph.type,
            "node_count": len(nodes),
            "edge_count": len(graph.edges),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    }
    yield _graph_start_event
    from magic_agents.agt_flow import CallbackEmitter
    CallbackEmitter.emit(_graph_start_event, chat_log)
    
    # === HOOK: on_graph_start (Phase 4) ===
    _graph_hook_context = None
    if hooks is not None and not hooks.is_empty():
        from magic_agents.hooks.context_factory import HookContextFactory
        _graph_hook_context = HookContextFactory.build_graph_context(
            execution_id=_execution_id,
            run_id=run_id or '',
            metadata={
                "graph_id": getattr(graph, 'id', '') or '',
                "graph_type": graph.type,
                "node_count": len(nodes),
                "edge_count": len(graph.edges),
            },
        )
        await hooks.invoke("on_graph_start", _graph_hook_context)

    # Initialize observer registry (replaces inline GraphDebugFeedback for loop executor)
    _debug_enabled_global = os.environ.get('DEBUG_ENABLED', 'true').strip().lower() in {'1', 'true', 'yes', 'on'}
    _resolved_debug_config = getattr(graph, 'resolved_debug_config', None)
    observer_registry = ObserverRegistry.create(
        debug_enabled_global=_debug_enabled_global,
        graph_debug=graph.debug,
        graph_debug_config=_resolved_debug_config,
        execution_id=_execution_id,
        graph_type=graph.type,
        total_nodes=len(nodes),
        total_edges=len(graph.edges),
        callback=debug_callback,
    )
    _exec_start_time = datetime.now(UTC)
    
    if observer_registry.is_active:
        await observer_registry.graph_observer.on_graph_start(
            graph_type=graph.type,
            execution_id=_execution_id,
            node_count=len(nodes),
            edge_count=len(graph.edges),
        )
    
    # Find the loop node
    loop_id = next(nid for nid, node in nodes.items() if isinstance(node, NodeLoop))
    loop_node = nodes[loop_id]
    
    # Classify edges by their role in the loop
    all_edges = list(graph.edges)
    item_edges = [e for e in all_edges if e.source == loop_id and e.sourceHandle == loop_node.OUTPUT_HANDLE_ITEM]
    loop_back_edges = [e for e in all_edges if e.target == loop_id and e.targetHandle == loop_node.INPUT_HANDLE_LOOP]
    end_edges = [e for e in all_edges if e.source == loop_id and e.sourceHandle == loop_node.OUTPUT_HANDLE_END]
    static_edges = [e for e in all_edges if e not in item_edges + loop_back_edges + end_edges]
    iteration_subgraph = find_iteration_subgraph(loop_id, nodes, all_edges)
    
    logger.debug(
        "Loop edges: static=%d item=%d loop_back=%d end=%d",
        len(static_edges), len(item_edges), len(loop_back_edges), len(end_edges)
    )
    logger.debug("Iteration subgraph nodes: %s", iteration_subgraph)
    
    # Create dispatcher for the full graph with graph-level timeout
    dispatcher = GraphEventDispatcher(
        nodes, graph.edges,
        timeout=graph.timeout,
        execution_id=_execution_id,
        run_id=run_id or '',
    )

    failed_node_ids: Set[str] = set()

    def mark_completed(node_id: str) -> None:
        """Record a unique graph node as successfully executed."""
        if node_id not in failed_node_ids:
            dispatcher.set_state(node_id, NodeState.COMPLETED)

    def mark_bypassed(node_id: str) -> None:
        """Record a bypass unless the node executed or failed in another iteration."""
        if (
            node_id not in failed_node_ids
            and dispatcher.get_state(node_id) != NodeState.COMPLETED
        ):
            dispatcher.set_state(node_id, NodeState.BYPASSED)

    def mark_failed(node_id: str) -> None:
        failed_node_ids.add(node_id)
        dispatcher.set_state(node_id, NodeState.ERROR)

    # ``NodeLoop`` is an executor-owned orchestration node: invoking its
    # ``process`` method would iterate independently of the graph phases below.
    # Mirror ``Node.__call__`` lifecycle delivery around that orchestration so
    # hooks, persistence, and observers still see the loop as one graph node.
    _loop_hook_context = None
    _loop_observer = (
        observer_registry.observer_for(loop_id, loop_node)
        if observer_registry.is_active
        else None
    )
    _loop_started_at: Optional[datetime] = None
    _loop_lifecycle_finished = False

    async def start_loop_lifecycle() -> None:
        nonlocal _loop_hook_context, _loop_started_at
        if _loop_started_at is not None:
            return

        _loop_started_at = datetime.now(UTC)
        safe_inputs = loop_node._safe_copy_dict(loop_node.inputs)

        if hooks is not None and not hooks.is_empty():
            from magic_agents.hooks.context_factory import HookContextFactory
            _loop_hook_context = HookContextFactory.build_node_context(
                execution_id=_execution_id,
                run_id=run_id or '',
                node_id=loop_id,
                node_type=getattr(loop_node, 'node_type', 'unknown') or 'unknown',
                node_class=type(loop_node).__name__,
                inputs=safe_inputs,
                start_time=_loop_started_at,
                parent_run_id=parent_run_id,
            )
            await hooks.invoke("on_node_start", _loop_hook_context)

        if _loop_observer is not None:
            await _loop_observer.on_node_start(
                node_id=loop_id,
                node_type=getattr(loop_node, 'node_type', 'unknown') or 'unknown',
                node_class=type(loop_node).__name__,
                inputs=safe_inputs,
            )

    async def end_loop_lifecycle() -> None:
        nonlocal _loop_lifecycle_finished
        if _loop_lifecycle_finished:
            return
        await start_loop_lifecycle()

        ended_at = datetime.now(UTC)
        duration_ms = (ended_at - _loop_started_at).total_seconds() * 1000
        safe_outputs = loop_node._safe_copy_dict(loop_node.outputs)

        if _loop_observer is not None:
            await _loop_observer.on_node_end(
                node_id=loop_id,
                node_type=getattr(loop_node, 'node_type', 'unknown') or 'unknown',
                node_class=type(loop_node).__name__,
                outputs=safe_outputs,
                internal_state=loop_node._capture_internal_state(),
                duration_ms=duration_ms,
                start_time=_loop_started_at.isoformat(),
            )

        if _loop_hook_context is not None:
            _loop_hook_context.timestamp = ended_at
            _loop_hook_context.end_time = ended_at
            _loop_hook_context.duration_ms = duration_ms
            _loop_hook_context.outputs = safe_outputs
            await hooks.invoke("on_node_end", _loop_hook_context)

        _loop_lifecycle_finished = True

    async def error_loop_lifecycle(error: Exception) -> None:
        nonlocal _loop_lifecycle_finished
        if _loop_lifecycle_finished:
            return
        await start_loop_lifecycle()

        ended_at = datetime.now(UTC)
        duration_ms = (ended_at - _loop_started_at).total_seconds() * 1000
        safe_inputs = loop_node._safe_copy_dict(loop_node.inputs)
        safe_outputs = loop_node._safe_copy_dict(loop_node.outputs)

        if _loop_observer is not None:
            await _loop_observer.on_node_error(
                node_id=loop_id,
                node_type=getattr(loop_node, 'node_type', 'unknown') or 'unknown',
                node_class=type(loop_node).__name__,
                error=str(error),
                error_type=type(error).__name__,
                inputs=safe_inputs,
                outputs=safe_outputs,
                duration_ms=duration_ms,
                start_time=_loop_started_at.isoformat(),
            )

        if _loop_hook_context is not None:
            _loop_hook_context.timestamp = ended_at
            _loop_hook_context.end_time = ended_at
            _loop_hook_context.duration_ms = duration_ms
            _loop_hook_context.error = error
            _loop_hook_context.error_type = type(error).__name__
            _loop_hook_context.error_message = str(error)
            await hooks.invoke("on_node_error", _loop_hook_context, error=error)

        _loop_lifecycle_finished = True

    async def execute_traversed_edge_hooks(
        source_node_id: str,
        outputs: Dict[str, Any],
        candidate_edges: List,
        ignored_edge_ids: Optional[Set[str]] = None,
    ):
        """Execute NodeHook once for each active edge traversal.

        Loop graphs are scheduled phase-by-phase instead of through tracker
        tasks.  Edge hooks therefore run inline immediately after their source
        output becomes available.  Resetting the hook node supports repeated
        traversals of the same edge across loop iterations.
        """

        ignored_edge_ids = ignored_edge_ids or set()
        for edge in candidate_edges:
            if (
                edge.id in ignored_edge_ids
                or edge.source != source_node_id
                or edge.sourceHandle not in outputs
            ):
                continue

            output = outputs[edge.sourceHandle]
            content = (
                output["content"]
                if isinstance(output, dict) and "content" in output
                else output
            )
            hook_node_id = await dispatcher.dispatch_edge_hook(
                edge,
                source_node_id,
                content,
                notify_tracker=False,
            )
            if hook_node_id is None:
                continue

            hook_node = nodes[hook_node_id]
            hook_node._response = None
            hook_node.outputs.clear()
            async for hook_output in execute_node_inline(
                hook_node_id,
                edges_to_process=[],
            ):
                yield hook_output

    # Helper to execute a single node inline
    async def execute_node_inline(
        node_id: str,
        edges_to_process: List = None,
        ignored_edge_ids: Optional[Set[str]] = None,
    ):
        """Execute a node and propagate outputs.
        
        Args:
            node_id: ID of the node to execute
            edges_to_process: List of edges to check for inputs. If None, uses all_edges.
        """
        node = nodes[node_id]
        
        # Use all edges for input resolution to handle cross-phase dependencies
        # (e.g., client nodes executed in static phase feeding LLM nodes executed post-loop)
        edges_for_inputs = edges_to_process if edges_to_process is not None else all_edges
        
        # First apply inputs from edges
        ignored_edge_ids = ignored_edge_ids or set()
        for edge in edges_for_inputs:
            if edge.id in ignored_edge_ids:
                continue
            if edge.target == node_id:
                source_node = nodes.get(edge.source)
                if source_node and source_node.outputs:
                    node.add_parent(source_node.outputs, edge.sourceHandle, edge.targetHandle)
        
        # Execute if not already done
        if node._response is None:
            logger.debug("Executing loop node %s", node_id)
            _node_obs = observer_registry.observer_for(node_id, node) if observer_registry.is_active else None
            if node_id not in failed_node_ids:
                dispatcher.set_state(node_id, NodeState.EXECUTING)
            try:
                async for item in node(chat_log, hooks=hooks, observer=_node_obs):
                    item_type = item.get("type", "")

                    # Check if this is streaming content
                    is_streaming = False
                    if hasattr(node, 'OUTPUT_HANDLE_CONTENT'):
                        is_streaming = item_type == node.OUTPUT_HANDLE_CONTENT
                    elif item_type == SYSTEM_EVENT_STREAMING:
                        is_streaming = True

                    if is_streaming:
                        yield {
                            "type": SYSTEM_EVENT_STREAMING,
                            "content": item["content"]["content"]
                        }
                    elif item_type == SYSTEM_EVENT_DEBUG:
                        yield item
                    elif ConditionalSignalTypes.is_system_signal(item_type):
                        logger.debug("Loop node %s emitted system signal: %s", node_id, item_type)
                    else:
                        # All other outputs stored using their handle name
                        node.outputs[item_type] = item["content"]
            except Exception as exc:
                mark_failed(node_id)
                logger.error("Loop node %s failed: %s", node_id, exc)
                yield {
                    "type": SYSTEM_EVENT_DEBUG,
                    "content": {
                        "node_id": node_id,
                        "node_type": getattr(node, "node_type", "unknown") or "unknown",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                }
            else:
                mark_completed(node_id)

                async for hook_output in execute_traversed_edge_hooks(
                    node_id,
                    node.outputs,
                    edges_for_inputs,
                    ignored_edge_ids,
                ):
                    yield hook_output

    # Track bypassed nodes during static phase
    bypassed_nodes: Set[str] = set()
    bypassed_edge_ids: Set[str] = set()
    
    # Pending observer bypass notifications for static phase
    # Collected during sync propagate_bypass_static, flushed asynchronously
    _pending_static_bypasses: List[Tuple[str, str, str, str, str]] = []
    
    # Feedback edges do not participate in deciding whether the loop received
    # its initial list. All other phase boundaries are included so convergence
    # with an active static/item/end path is preserved.
    static_bypass_edges = [e for e in all_edges if e not in loop_back_edges]

    def _record_phase_bypass(
        node_id: str,
        phase_bypassed_nodes: Set[str],
        pending: List[Tuple[str, str, str, str, str]],
    ) -> bool:
        """Record one phase-local node bypass and its lifecycle notification."""
        if node_id in phase_bypassed_nodes:
            return False
        phase_bypassed_nodes.add(node_id)
        mark_bypassed(node_id)
        node = nodes.get(node_id)
        if node and hasattr(node, 'mark_bypassed'):
            node.mark_bypassed()
            pending.append((
                node_id,
                getattr(node, 'node_type', 'unknown') or 'unknown',
                type(node).__name__,
                _execution_id,
                run_id or '',
            ))
        return True

    def _propagate_edge_bypass(
        edge,
        *,
        phase_edges: List,
        phase_bypassed_edges: Set[str],
        phase_bypassed_nodes: Set[str],
        pending: List[Tuple[str, str, str, str, str]],
        allowed_nodes: Optional[Set[str]] = None,
    ) -> None:
        """Account for one inactive edge and bypass only fully inactive targets."""
        if edge.id in phase_bypassed_edges:
            return
        phase_bypassed_edges.add(edge.id)

        if edge.hooks and edge.hooks.enabled and edge.hooks.hook_node_id:
            hook_node_id = edge.hooks.hook_node_id
            if hook_node_id in nodes:
                _record_phase_bypass(
                    hook_node_id,
                    phase_bypassed_nodes,
                    pending,
                )

        target_id = edge.target
        if target_id not in nodes:
            return
        if allowed_nodes is not None and target_id not in allowed_nodes:
            return

        incoming = [candidate for candidate in phase_edges if candidate.target == target_id]
        if incoming and any(candidate.id not in phase_bypassed_edges for candidate in incoming):
            logger.debug(
                "Preserving converged node %s: at least one incoming edge remains active",
                target_id,
            )
            return

        if not _record_phase_bypass(target_id, phase_bypassed_nodes, pending):
            return

        logger.debug("Bypassing node %s after all incoming edges were bypassed", target_id)
        for outgoing in phase_edges:
            if outgoing.source == target_id:
                _propagate_edge_bypass(
                    outgoing,
                    phase_edges=phase_edges,
                    phase_bypassed_edges=phase_bypassed_edges,
                    phase_bypassed_nodes=phase_bypassed_nodes,
                    pending=pending,
                    allowed_nodes=allowed_nodes,
                )

    def propagate_bypass_static(node_id: str) -> None:
        """Force a phase root to bypass, then account for downstream edges."""
        if not _record_phase_bypass(node_id, bypassed_nodes, _pending_static_bypasses):
            return
        for edge in static_bypass_edges:
            if edge.source == node_id:
                _propagate_edge_bypass(
                    edge,
                    phase_edges=static_bypass_edges,
                    phase_bypassed_edges=bypassed_edge_ids,
                    phase_bypassed_nodes=bypassed_nodes,
                    pending=_pending_static_bypasses,
                )
    
    def handle_conditional_bypass_static(cond_node_id: str, selected_handle: str):
        """Handle conditional bypass in static phase."""
        outgoing = [e for e in static_bypass_edges if e.source == cond_node_id]
        for edge in outgoing:
            if edge.sourceHandle != selected_handle:
                _propagate_edge_bypass(
                    edge,
                    phase_edges=static_bypass_edges,
                    phase_bypassed_edges=bypassed_edge_ids,
                    phase_bypassed_nodes=bypassed_nodes,
                    pending=_pending_static_bypasses,
                )
                logger.debug(
                    "Static conditional bypass: %s.%s -> %s (not selected)",
                    cond_node_id, edge.sourceHandle, edge.target
                )

    async def flush_static_bypass_notifications(
        observer_reason: str,
        phase: str,
        hook_reason: str = "condition",
    ) -> None:
        """Deliver and clear bypass notifications accumulated by sync traversal."""
        if not _pending_static_bypasses:
            return

        if hooks is not None and not hooks.is_empty():
            from magic_agents.hooks.context_factory import HookContextFactory
            for _nid, _ntype, _nclass, _eid, _rid in _pending_static_bypasses:
                _bypass_ctx = HookContextFactory.build_bypass_context(
                    execution_id=_eid,
                    run_id=_rid,
                    node_id=_nid,
                    node_type=_ntype,
                    node_class=_nclass,
                    reason=hook_reason,
                    metadata={"phase": phase},
                )
                await hooks.invoke(
                    "on_node_bypass",
                    _bypass_ctx,
                    reason=hook_reason,
                )

        if observer_registry.is_active:
            _bypass_observer = observer_registry.graph_observer
            for _nid, _ntype, _nclass, _eid, _rid in _pending_static_bypasses:
                await _bypass_observer.on_node_bypass(
                    node_id=_nid,
                    node_type=_ntype,
                    node_class=_nclass,
                    reason=observer_reason,
                )

        _pending_static_bypasses.clear()
    
    # Build topological order for static nodes
    def topological_sort_static() -> List[str]:
        """Sort static nodes in execution order."""
        # First, find ALL nodes reachable from loop's handle_end (post-loop nodes)
        # These should NOT be executed in static phase - they need loop output
        post_loop_nodes = set()
        queue = [e.target for e in end_edges]
        while queue:
            node_id = queue.pop(0)
            if node_id in post_loop_nodes or node_id == loop_id:
                continue
            post_loop_nodes.add(node_id)
            # Find all downstream nodes via static edges
            for edge in static_edges:
                if edge.source == node_id and edge.target not in post_loop_nodes:
                    queue.append(edge.target)
        
        logger.debug("Post-loop nodes excluded from static phase: %s", post_loop_nodes)
        
        # Collect all nodes involved in static edges EXCEPT post-loop nodes
        static_nodes = set()
        for edge in static_edges:
            static_nodes.add(edge.source)
            static_nodes.add(edge.target)
        # Remove loop node from static processing
        static_nodes.discard(loop_id)
        # Remove post-loop nodes - they will be executed after the loop
        static_nodes -= post_loop_nodes
        # Branches reached from handle_item belong exclusively to iteration.
        # Internal conditional edges are otherwise classified as static edges,
        # which used to execute both branches once before the loop started.
        static_nodes -= iteration_subgraph
        
        # Build in-degree map
        in_degree = {n: 0 for n in static_nodes}
        adjacency = {n: [] for n in static_nodes}
        
        for edge in static_edges:
            if edge.source in static_nodes and edge.target in static_nodes:
                adjacency[edge.source].append(edge.target)
                in_degree[edge.target] += 1
        
        # Kahn's algorithm
        result = []
        queue = [n for n in static_nodes if in_degree.get(n, 0) == 0]
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            for neighbor in adjacency.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Add any remaining (handles cycles)
        for n in static_nodes:
            if n not in result:
                result.append(n)
        
        return result
    
    # Process static phase in topological order with conditional support
    static_order = topological_sort_static()
    logger.debug("Static execution order: %s", static_order)
    
    for node_id in static_order:
        # Skip if already bypassed
        if node_id in bypassed_nodes:
            logger.debug("Skipping bypassed node %s", node_id)
            continue

        # Apply inputs from completed static nodes
        for edge in static_edges:
            if edge.id in bypassed_edge_ids:
                continue
            if edge.target == node_id:
                source_node = nodes.get(edge.source)
                target_node = nodes.get(node_id)
                if source_node and target_node and source_node.outputs:
                    if edge.sourceHandle in source_node.outputs:
                        target_node.add_parent(source_node.outputs, edge.sourceHandle, edge.targetHandle)

        # Skip conditional nodes that have no inputs — their inputs come from
        # the loop's iteration output (handle_item) which isn't available yet.
        # They will be executed during the iteration phase instead.
        # Use hasattr for pre-execution detection (selected_handle not set yet).
        node_obj = nodes.get(node_id)
        if hasattr(node_obj, 'condition_template'):
            cond_node = node_obj
            has_inputs = any(
                cond_node.inputs.get(h) is not None
                for h in cond_node.inputs.keys()
            )
            if not has_inputs:
                logger.debug(
                    "Skipping conditional %s in static phase — no inputs yet "
                    "(depends on loop iteration output)",
                    node_id,
                )
                continue

        # Execute the node
        async for out in execute_node_inline(
            node_id,
            static_edges,
            ignored_edge_ids=bypassed_edge_ids,
        ):
            yield out

        if dispatcher.get_state(node_id) == NodeState.ERROR:
            for edge in static_bypass_edges:
                if edge.source == node_id:
                    _propagate_edge_bypass(
                        edge,
                        phase_edges=static_bypass_edges,
                        phase_bypassed_edges=bypassed_edge_ids,
                        phase_bypassed_nodes=bypassed_nodes,
                        pending=_pending_static_bypasses,
                    )
            continue
        
        # Handle conditional bypass propagation
        node = nodes.get(node_id)
        if isinstance(node, ConditionalRouting):
            selected_handle = getattr(node, 'selected_handle', None)
            if selected_handle:
                handle_conditional_bypass_static(node_id, selected_handle)
                logger.debug("Conditional %s selected handle: %s", node_id, selected_handle)
    
    # Deliver static-phase bypass notifications before loop input resolution.
    await flush_static_bypass_notifications("static_conditional_bypass", "static")
    
    # Transfer final outputs to loop node
    for edge in static_edges:
        if edge.id in bypassed_edge_ids:
            continue
        if edge.target == loop_id:
            source_node = nodes.get(edge.source)
            if source_node and source_node.outputs and edge.sourceHandle in source_node.outputs:
                loop_node.add_parent(source_node.outputs, edge.sourceHandle, edge.targetHandle)

    # Get the list to iterate
    raw = loop_node.inputs.get(loop_node.INPUT_HANDLE_LIST)
    if isinstance(raw, str):
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            items = raw
    else:
        items = raw
    
    # Track if loop is bypassed
    loop_bypassed = False
    
    # Handle case where loop input was bypassed
    if raw is None:
        list_input_edges = [
            edge for edge in static_edges
            if edge.target == loop_id
            and edge.targetHandle == loop_node.INPUT_HANDLE_LIST
        ]
        list_path_bypassed = bool(list_input_edges) and all(
            edge.id in bypassed_edge_ids for edge in list_input_edges
        )

        if loop_id in bypassed_nodes or list_path_bypassed:
            logger.info("Loop input source was bypassed - skipping loop execution")
            loop_bypassed = True
            propagate_bypass_static(loop_id)
            # These bypasses are discovered after the initial static-phase
            # flush, so they need their own delivery before graph finalization.
            await flush_static_bypass_notifications("loop_input_bypassed", "loop_bypass")
        else:
            # No input and not bypassed - this is an error
            error_msg = f"Loop node '{loop_id}' did not receive input on handle '{loop_node.INPUT_HANDLE_LIST}'"
            logger.error(error_msg)
            loop_error = ValueError(error_msg)
            await start_loop_lifecycle()
            yield {
                "type": SYSTEM_EVENT_DEBUG,
                "content": {
                    "node_id": loop_id,
                    "node_type": "LOOP",
                    "error_type": "InputError",
                    "error_message": error_msg,
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
            mark_failed(loop_id)
            await error_loop_lifecycle(loop_error)
            loop_bypassed = True
            for edge in item_edges + end_edges:
                _propagate_edge_bypass(
                    edge,
                    phase_edges=static_bypass_edges,
                    phase_bypassed_edges=bypassed_edge_ids,
                    phase_bypassed_nodes=bypassed_nodes,
                    pending=_pending_static_bypasses,
                )
            await flush_static_bypass_notifications("loop_input_error", "loop_error")
    elif not isinstance(items, list):
        error_msg = f"Loop node '{loop_id}' expects a list, got {type(items)}"
        logger.error(error_msg)
        loop_error = ValueError(error_msg)
        await start_loop_lifecycle()
        yield {
            "type": SYSTEM_EVENT_DEBUG,
            "content": {
                "node_id": loop_id,
                "node_type": "LOOP",
                "error_type": "ValidationError",
                "error_message": error_msg,
                "timestamp": datetime.now(UTC).isoformat()
            }
        }
        mark_failed(loop_id)
        await error_loop_lifecycle(loop_error)
        loop_bypassed = True
        for edge in item_edges + end_edges:
            _propagate_edge_bypass(
                edge,
                phase_edges=static_bypass_edges,
                phase_bypassed_edges=bypassed_edge_ids,
                phase_bypassed_nodes=bypassed_nodes,
                pending=_pending_static_bypasses,
            )
        await flush_static_bypass_notifications("loop_input_error", "loop_error")
    
    # Only execute loop iterations if not bypassed
    if not loop_bypassed:
        logger.info("Loop iterating over %d items", len(items))
        await start_loop_lifecycle()
        dispatcher.set_state(loop_id, NodeState.EXECUTING)
        
        # Get topological order for iteration execution
        execution_order = topological_sort_iteration(iteration_subgraph, item_edges, loop_back_edges, all_edges)
        logger.debug("Iteration execution order: %s", execution_order)
        
        # Find the feedback-producing node (the one that feeds back to handle_loop)
        feedback_node_id = None
        for edge in loop_back_edges:
            if edge.target == loop_id and edge.targetHandle == loop_node.INPUT_HANDLE_LOOP:
                feedback_node_id = edge.source
                break
        
        logger.debug("Feedback node: %s", feedback_node_id)
        
        # Get loop configuration (with defaults)
        max_iterations = getattr(loop_node, 'max_iterations', DEFAULT_MAX_ITERATIONS)
        
        loop_agg = []
        start_time = time.time()
        total_items = len(items)
        item_edges_traversed = False
        
        for idx, item in enumerate(items):
            # Check iteration limit
            if idx >= max_iterations:
                logger.warning("Loop reached max iterations limit: %d", max_iterations)
                yield {
                    "type": SYSTEM_EVENT_DEBUG,
                    "content": {
                        "node_id": loop_id,
                        "node_type": "LOOP",
                        "error_type": "MaxIterationsExceeded",
                        "error_message": f"Loop exceeded max iterations ({max_iterations})",
                        "iterations_completed": idx,
                        "timestamp": datetime.now(UTC).isoformat()
                    }
                }
                break
            
            # Emit progress event
            elapsed_ms = (time.time() - start_time) * 1000
            yield emit_loop_progress(loop_id, idx, total_items, item, elapsed_ms)
            
            # Phase 0: emit ITERATION_START debug event for execution tree persistence
            iteration_start = time.time()
            yield {
                "type": SYSTEM_EVENT_DEBUG,
                "content": {
                    "event_type": "ITERATION_START",
                    "loop_node_id": loop_id,
                    "iteration": idx,
                    "total_items": total_items,
                    "current_item_preview": str(item)[:100] if item is not None else None,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            }
            
            # Reset loop state for this iteration
            loop_node._response = None
            loop_node.outputs.clear()
            # Clear the feedback input from previous iteration
            if loop_node.INPUT_HANDLE_LOOP in loop_node.inputs:
                del loop_node.inputs[loop_node.INPUT_HANDLE_LOOP]
            
            # Reset ALL nodes in the iteration subgraph (not just immediate downstream)
            reset_iteration_nodes(nodes, iteration_subgraph)
            
            # Track bypassed nodes WITHIN this iteration (reset each iteration).
            # When a conditional selects one branch, all other branches and their
            # downstream nodes must be skipped.
            iteration_bypassed: Set[str] = {
                node_id for node_id in bypassed_nodes if node_id in iteration_subgraph
            }
            iteration_bypassed_edges: Set[str] = set(bypassed_edge_ids)
            
            # Pending observer bypass notifications for this iteration.
            # Collected during sync propagate_bypass_iteration, flushed
            # asynchronously at the end of the iteration's node execution phase.
            _pending_iteration_bypasses: List[Tuple[str, str, str, str, str]] = []
            
            def bypass_non_selected_conditional_branches(cond_node_id: str, selected_handle: str):
                """After a conditional executes, bypass all non-selected branches."""
                for edge in all_edges:
                    if edge.source == cond_node_id and edge.sourceHandle != selected_handle:
                        logger.debug(
                            "Iteration %d: conditional %s selected '%s', bypassing '%s' -> %s",
                            idx, cond_node_id, selected_handle, edge.sourceHandle, edge.target
                        )
                        _propagate_edge_bypass(
                            edge,
                            phase_edges=all_edges,
                            phase_bypassed_edges=iteration_bypassed_edges,
                            phase_bypassed_nodes=iteration_bypassed,
                            pending=_pending_iteration_bypasses,
                            allowed_nodes=iteration_subgraph,
                        )
            
            # Set current item as loop output - PRESERVING TYPE (Issue #4 fix)
            loop_node.outputs[loop_node.OUTPUT_HANDLE_ITEM] = prepare_item_output(item, idx)
            item_edges_traversed = True
            
            # Process item edges - transfer loop item to first downstream nodes
            for edge in item_edges:
                if edge.id in iteration_bypassed_edges:
                    continue
                target_node = nodes.get(edge.target)
                if target_node:
                    target_node.add_parent(loop_node.outputs, edge.sourceHandle, edge.targetHandle)

            async for hook_output in execute_traversed_edge_hooks(
                loop_id,
                loop_node.outputs,
                item_edges,
                iteration_bypassed_edges,
            ):
                yield hook_output
            
            # Execute iteration subgraph in TOPOLOGICAL ORDER (Issue #2 fix)
            # This ensures each node completes before its dependents start
            for node_id in execution_order:
                node = nodes.get(node_id)
                if not node:
                    continue
                
                # Skip nodes bypassed by conditional branch selection in this iteration
                if node_id in iteration_bypassed or node_id in bypassed_nodes:
                    logger.debug("Skipping bypassed iteration node %s", node_id)
                    continue
                
                # Apply inputs from any edges where source has completed
                # Use all_edges to capture conditional branch edges too
                for edge in all_edges:
                    if edge.id in iteration_bypassed_edges:
                        continue
                    if edge.target == node_id:
                        source_node = nodes.get(edge.source)
                        # Source could be loop node or another iteration node
                        # Don't apply inputs from bypassed sources
                        if edge.source in iteration_bypassed:
                            continue
                        if edge.source == loop_id:
                            node.add_parent(loop_node.outputs, edge.sourceHandle, edge.targetHandle)
                        elif source_node and source_node.outputs:
                            node.add_parent(source_node.outputs, edge.sourceHandle, edge.targetHandle)
                
                # Execute the node and WAIT for completion
                async for out in execute_node_inline(
                    node_id,
                    all_edges,
                    ignored_edge_ids=iteration_bypassed_edges,
                ):
                    yield out

                if dispatcher.get_state(node_id) == NodeState.ERROR:
                    for edge in all_edges:
                        if edge.source == node_id:
                            _propagate_edge_bypass(
                                edge,
                                phase_edges=all_edges,
                                phase_bypassed_edges=iteration_bypassed_edges,
                                phase_bypassed_nodes=iteration_bypassed,
                                pending=_pending_iteration_bypasses,
                                allowed_nodes=iteration_subgraph,
                            )
                    continue
                
                # After execution, handle conditional bypass propagation
                if isinstance(node, ConditionalRouting):
                    selected_handle = getattr(node, 'selected_handle', None)
                    if selected_handle:
                        bypass_non_selected_conditional_branches(node_id, selected_handle)
                        logger.debug(
                            "Iteration %d: conditional %s selected '%s', bypassed: %s",
                            idx, node_id, selected_handle, iteration_bypassed
                        )
                
                # After execution, propagate outputs to downstream nodes in subgraph
                # Use all_edges to capture conditional branch edges too.
                # Also propagate to loop node via loop-back edges (loop is NOT in iteration_subgraph
                # but needs to receive feedback).
                for edge in all_edges:
                    if edge.id in iteration_bypassed_edges:
                        continue
                    if edge.source == node_id:
                        # Only propagate to nodes in the iteration subgraph or the loop node itself
                        if edge.target not in iteration_subgraph and edge.target != loop_id:
                            continue
                        target = nodes.get(edge.target)
                        if target and node.outputs and edge.target not in iteration_bypassed:
                            target.add_parent(node.outputs, edge.sourceHandle, edge.targetHandle)
            
            # HOOK: on_node_bypass for iteration phase conditional bypasses (Phase 4) — reason="condition"
            if hooks is not None and not hooks.is_empty() and _pending_iteration_bypasses:
                from magic_agents.hooks.context_factory import HookContextFactory
                for _nid, _ntype, _nclass, _eid, _rid in _pending_iteration_bypasses:
                    _bypass_ctx = HookContextFactory.build_bypass_context(
                        execution_id=_eid,
                        run_id=_rid,
                        node_id=_nid,
                        node_type=_ntype,
                        node_class=_nclass,
                        reason="condition",
                        metadata={"phase": "iteration"},
                    )
                    await hooks.invoke("on_node_bypass", _bypass_ctx, reason="condition")
            
            # Flush pending iteration bypass observer notifications
            if observer_registry.is_active and _pending_iteration_bypasses:
                _bypass_observer = observer_registry.graph_observer
                for _nid, _ntype, _nclass, _eid, _rid in _pending_iteration_bypasses:
                    await _bypass_observer.on_node_bypass(
                        node_id=_nid,
                        node_type=_ntype,
                        node_class=_nclass,
                        reason="iteration_conditional_bypass",
                    )
                _pending_iteration_bypasses.clear()
            
            # NOW collect the feedback AFTER all processing is complete (Issue #1 fix)
            # The feedback-producing node should have written to loop_node.inputs
            fb = loop_node.inputs.get(loop_node.INPUT_HANDLE_LOOP)
            
            # Extract actual content if wrapped in standard output format
            if isinstance(fb, dict) and 'content' in fb:
                fb = fb['content']
            
            logger.debug("Iteration %d feedback: %s", idx, str(fb)[:100] if fb else "None")
            loop_agg.append(fb)
            
            # Phase 0: emit ITERATION_END debug event for execution tree persistence
            iteration_duration_ms = (time.time() - iteration_start) * 1000
            yield {
                "type": SYSTEM_EVENT_DEBUG,
                "content": {
                    "event_type": "ITERATION_END",
                    "loop_node_id": loop_id,
                    "iteration": idx,
                    "duration_ms": round(iteration_duration_ms, 2),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            }

        if not item_edges_traversed:
            for edge in item_edges:
                _propagate_edge_bypass(
                    edge,
                    phase_edges=all_edges,
                    phase_bypassed_edges=bypassed_edge_ids,
                    phase_bypassed_nodes=bypassed_nodes,
                    pending=_pending_static_bypasses,
                    allowed_nodes=iteration_subgraph,
                )
            await flush_static_bypass_notifications(
                "loop_no_iterations",
                "loop_no_iterations",
                hook_reason="not_ready",
            )
        
        # Finish loop - set aggregated result
        loop_node._response = None
        loop_node.outputs.clear()
        loop_node.outputs[loop_node.OUTPUT_HANDLE_END] = loop_node.prep(loop_agg)
        mark_completed(loop_id)
        await end_loop_lifecycle()

        async for hook_output in execute_traversed_edge_hooks(
            loop_id,
            loop_node.outputs,
            end_edges,
            bypassed_edge_ids,
        ):
            yield hook_output
        
        # Clear response for end nodes so they execute
        for edge in end_edges:
            target_node = nodes.get(edge.target)
            if target_node:
                target_node._response = None
                target_node.outputs.clear()

    # Process end edges and ALL downstream nodes using topological order
    # Find all nodes reachable from the loop's handle_end output
    def find_post_loop_nodes() -> List[str]:
        """Find all nodes downstream of loop's handle_end in topological order."""
        # Start with direct targets of end_edges
        post_loop_nodes = set()
        queue = [e.target for e in end_edges]
        
        while queue:
            node_id = queue.pop(0)
            if node_id in post_loop_nodes:
                continue
            if node_id in iteration_subgraph:
                continue  # Don't include nodes already in iteration subgraph
            if node_id == loop_id:
                continue  # Don't include the loop node itself

            post_loop_nodes.add(node_id)
            
            # Find downstream nodes
            for edge in all_edges:
                if edge.source == node_id and edge.target not in post_loop_nodes:
                    queue.append(edge.target)
        
        # Topological sort of post-loop nodes
        in_degree = {n: 0 for n in post_loop_nodes}
        adjacency = {n: [] for n in post_loop_nodes}
        
        # Get all static nodes that were executed (not bypassed)
        executed_static_nodes = set(static_order) - bypassed_nodes
        
        for edge in all_edges:
            # Only consider edges within post_loop_nodes OR from static sources
            if edge.target in post_loop_nodes:
                if edge.source in post_loop_nodes:
                    adjacency[edge.source].append(edge.target)
                    in_degree[edge.target] += 1
                elif edge.source == loop_id:
                    # Edge from loop node - these are entry points (in_degree stays 0)
                    pass
                elif edge.source in executed_static_nodes:
                    # Edge from executed static node - these are also ready
                    pass
                else:
                    # Edge from another source that might not be in post_loop
                    # Check if source has been executed
                    source_node = nodes.get(edge.source)
                    if source_node and source_node._response is None:
                        # Source not yet executed, count it
                        in_degree[edge.target] += 1
        
        # Kahn's algorithm
        result = []
        queue = [n for n in post_loop_nodes if in_degree.get(n, 0) == 0]
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for neighbor in adjacency.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Add any remaining nodes (handles cycles or complex dependencies)
        for n in post_loop_nodes:
            if n not in result:
                result.append(n)
        
        return result
    
    # Get the execution order for post-loop nodes
    post_loop_order = find_post_loop_nodes()
    post_loop_node_ids = set(post_loop_order)
    logger.debug("Post-loop execution order: %s", post_loop_order)
    
    # First, transfer loop outputs to direct targets (only if loop wasn't bypassed)
    if not loop_bypassed:
        for edge in end_edges:
            if edge.id in bypassed_edge_ids:
                continue
            target_node = nodes.get(edge.target)
            if target_node:
                target_node.add_parent(loop_node.outputs, edge.sourceHandle, edge.targetHandle)
    
    # Execute all post-loop nodes in topological order, respecting bypasses
    for node_id in post_loop_order:
        # Skip bypassed nodes
        if node_id in bypassed_nodes:
            logger.debug("Skipping bypassed post-loop node %s", node_id)
            continue
        
        node = nodes.get(node_id)
        if not node:
            continue
        
        # Reset node for execution
        node._response = None
        node.outputs.clear()
        
        # Execute the node - execute_node_inline will apply inputs from all_edges
        async for out in execute_node_inline(
            node_id,
            ignored_edge_ids=bypassed_edge_ids,
        ):
            yield out

        if dispatcher.get_state(node_id) == NodeState.ERROR:
            for edge in all_edges:
                if edge.source == node_id:
                    _propagate_edge_bypass(
                        edge,
                        phase_edges=all_edges,
                        phase_bypassed_edges=bypassed_edge_ids,
                        phase_bypassed_nodes=bypassed_nodes,
                        pending=_pending_static_bypasses,
                        allowed_nodes=post_loop_node_ids,
                    )
            continue

        if isinstance(node, ConditionalRouting):
            selected_handle = getattr(node, 'selected_handle', None)
            if selected_handle:
                for edge in all_edges:
                    if edge.source == node_id and edge.sourceHandle != selected_handle:
                        _propagate_edge_bypass(
                            edge,
                            phase_edges=all_edges,
                            phase_bypassed_edges=bypassed_edge_ids,
                            phase_bypassed_nodes=bypassed_nodes,
                            pending=_pending_static_bypasses,
                            allowed_nodes=post_loop_node_ids,
                        )
        
        # Propagate outputs to downstream nodes within post_loop
        for edge in all_edges:
            if edge.id in bypassed_edge_ids:
                continue
            if edge.source == node_id:
                target = nodes.get(edge.target)
                if target and node.outputs:
                    target.add_parent(node.outputs, edge.sourceHandle, edge.targetHandle)

    await flush_static_bypass_notifications("post_loop_conditional_bypass", "post_loop")
    
    # === HOOK: on_graph_end / on_graph_error (Phase 4) ===
    # Spec requirement: on_graph_end fires for successful execution only.
    # on_graph_error fires when any node errored; on_graph_end is NOT invoked for failures.
    _summary = dispatcher.get_execution_summary()
    if _graph_hook_context is not None:
        _graph_hook_context.timestamp = datetime.now(UTC)
        _graph_hook_context.metadata["execution_summary"] = _summary
        if _summary["errors"] > 0:
            _graph_hook_context.error_message = (
                f"Graph execution completed with {_summary['errors']} node error(s)"
            )
            _graph_hook_context.metadata["failed_nodes"] = _summary["states"].get("error", [])
            await hooks.invoke("on_graph_error", _graph_hook_context, error=RuntimeError(f"Loop execution failed: {_summary['errors']} node error(s)"))
        else:
            await hooks.invoke("on_graph_end", _graph_hook_context)

    # Finalize observer — emit graph_end event and summary
    _exec_end_time = datetime.now(UTC)
    _total_duration = (_exec_end_time - _exec_start_time).total_seconds() * 1000
    
    if observer_registry.is_active:
        await observer_registry.graph_observer.on_graph_end(
            graph_type=graph.type,
            execution_id=_execution_id,
            total_duration_ms=_total_duration,
            node_count=len(nodes),
            executed_count=_summary.get("completed", 0),
            bypassed_count=_summary.get("bypassed", 0),
            failed_count=_summary.get("errors", 0),
        )
    
    logger.info("Finished reactive loop execution")
