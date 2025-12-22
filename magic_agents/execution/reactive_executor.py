"""
ReactiveExecutor - Event-based graph execution engine.

This module implements the main execution logic for the reactive
parallel execution model. Nodes execute automatically when their
inputs are ready, enabling natural parallelism based on graph topology.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union
from datetime import datetime, UTC

# Loop execution constants
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_ITERATION_TIMEOUT_MS = 30000
DEFAULT_TOTAL_TIMEOUT_MS = 300000

from magic_llm.model.ModelChatStream import ChatCompletionModel

from magic_agents.execution.event_dispatcher import GraphEventDispatcher, NodeState
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.models.debug_feedback import GraphDebugFeedback
from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalSignalTypes
from magic_agents.util.const import SYSTEM_EVENT_STREAMING, SYSTEM_EVENT_DEBUG, SYSTEM_EVENT_DEBUG_SUMMARY, SYSTEM_EVENT_TYPES

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
    loop_back_edges: List[Any]
) -> List[str]:
    """
    Sort iteration nodes in execution order using Kahn's algorithm.
    
    Args:
        iteration_nodes: Set of node IDs in the iteration subgraph
        item_edges: Edges from loop handle_item to downstream nodes
        loop_back_edges: Edges that feed back to loop handle_loop
        
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


async def execute_graph_reactive(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None,
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
        
    Yields:
        Streaming content and final outputs from nodes
    """
    # Check for validation errors
    if hasattr(graph, '_validation_errors') and graph._validation_errors:
        for error in graph._validation_errors:
            yield {
                "type": SYSTEM_EVENT_DEBUG,
                "content": {
                    **error,
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
    
    # Detect loop nodes - delegate to loop handler
    from magic_agents.node_system import NodeLoop
    loop_nodes = [nid for nid, node in graph.nodes.items() if isinstance(node, NodeLoop)]
    if loop_nodes:
        logger.info("Detected loop nodes: %s. Delegating to loop executor.", loop_nodes)
        async for msg in execute_graph_loop_reactive(
            graph, id_chat=id_chat, id_thread=id_thread, id_user=id_user
        ):
            yield msg
        return
    
    nodes = graph.nodes
    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app=getattr(graph, 'app_id', None) or getattr(graph, 'id_app', None)
    )
    
    logger.info(
        "Starting reactive execution: nodes=%d edges=%d",
        len(nodes), len(graph.edges)
    )
    
    # Initialize debug feedback if enabled
    debug_feedback: Optional[GraphDebugFeedback] = None
    debug_config = None
    if graph.debug:
        # Get resolved debug config (from JSON debug_config or default)
        debug_config = graph.resolved_debug_config
        
        # Only proceed if config is enabled (allows debug=true but config.enabled=false)
        if debug_config and debug_config.enabled:
            debug_feedback = GraphDebugFeedback(
                execution_id=uuid.uuid4().hex,
                graph_type=graph.type,
                start_time=datetime.now(UTC).isoformat()
            )
            logger.info(
                "Debug mode enabled: %s (config: redact=%s, max_payload=%d)",
                debug_feedback.execution_id,
                debug_config.redact_sensitive,
                debug_config.max_payload_length
            )
    
    # Create event dispatcher
    dispatcher = GraphEventDispatcher(nodes, graph.edges)
    
    # Output queue for collecting results from parallel tasks
    output_queue: asyncio.Queue = asyncio.Queue()
    
    # Import NodeConditional for type checking
    from magic_agents.node_system import NodeConditional
    
    async def execute_single_node(node_id: str):
        """Execute a single node when ready."""
        node = nodes[node_id]
        tracker = dispatcher.get_tracker(node_id)
        
        if not tracker:
            logger.error("No tracker for node %s", node_id)
            return
        
        try:
            # Wait for all inputs
            should_execute = await tracker.wait_ready(timeout=dispatcher.timeout)
            
            if not should_execute:
                # Node is bypassed
                dispatcher.set_state(node_id, NodeState.BYPASSED)
                node.mark_bypassed()
                logger.debug("Node %s bypassed", node_id)
                
                # Track in debug
                if debug_feedback:
                    debug_feedback.add_edge_info(
                        source=node_id,
                        target="(bypassed)",
                        source_handle="",
                        target_handle=""
                    )
                return
            
            # Execute the node
            dispatcher.set_state(node_id, NodeState.EXECUTING)
            logger.debug("Executing node %s (%s)", node_id, node.__class__.__name__)
            
            conditional_selected_handle: Optional[str] = None
            bypass_all_signaled = False
            
            async for item in node(chat_log):
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
                    # Queue debug info
                    await output_queue.put(item)
                elif ConditionalSignalTypes.is_system_signal(item_type):
                    # Other system signals - log and continue
                    logger.debug("Node %s emitted system signal: %s", node_id, item_type)
                else:
                    # Handle-specific output (conditional routing, etc.)
                    node.outputs[item_type] = item["content"]
                    # Track conditional selection (only non-system signals)
                    if isinstance(node, NodeConditional) and conditional_selected_handle is None:
                        if item_type not in (SYSTEM_EVENT_DEBUG, SYSTEM_EVENT_DEBUG_SUMMARY):
                            conditional_selected_handle = item_type
            
            # Mark completed
            dispatcher.set_state(node_id, NodeState.COMPLETED)
            logger.debug("Node %s completed", node_id)
            
            # Yield debug info
            if debug_feedback and hasattr(node, 'get_debug_info'):
                node_debug_info = node.get_debug_info()
                if node_debug_info and node_debug_info.was_executed:
                    await output_queue.put({
                        "type": SYSTEM_EVENT_DEBUG,
                        "content": node_debug_info.model_dump()
                    })
            
            # Propagate outputs to downstream nodes
            await dispatcher.propagate_outputs(node_id, node.outputs)
            
            # Handle conditional bypass propagation (skip if BYPASS_ALL was already handled)
            if isinstance(node, NodeConditional) and not bypass_all_signaled:
                selected_handle = conditional_selected_handle or getattr(node, 'selected_handle', None)
                if selected_handle:
                    # Verify edge exists for selected handle
                    outgoing = [e for e in graph.edges if e.source == node_id]
                    has_matching_edge = any(e.sourceHandle == selected_handle for e in outgoing)
                    
                    if not has_matching_edge:
                        # No matching edge - check for default_handle fallback
                        default_handle = getattr(node, 'default_handle', None)
                        if default_handle and any(e.sourceHandle == default_handle for e in outgoing):
                            logger.warning(
                                "Using default_handle '%s' after routing error for node %s",
                                default_handle, node_id
                            )
                            selected_handle = default_handle
                            await dispatcher.propagate_conditional_bypass(node_id, selected_handle)
                        else:
                            # No matching edge and no valid default - yield error and bypass all
                            await output_queue.put(node.yield_debug_error(
                                error_type="GraphRoutingError",
                                error_message=f"Conditional selected handle '{selected_handle}', but no outgoing edge matches.",
                                context={
                                    "selected_handle": selected_handle,
                                    "outgoing_handles": [e.sourceHandle for e in outgoing],
                                    "node_id": node_id,
                                    "suggestion": "Add output_handles to conditional data for build-time validation"
                                }
                            ))
                            await dispatcher.handle_bypass_all_signal(node_id)
                    else:
                        await dispatcher.propagate_conditional_bypass(node_id, selected_handle)
        
        except asyncio.TimeoutError:
            dispatcher.set_state(node_id, NodeState.ERROR)
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
        
        except Exception as e:
            dispatcher.set_state(node_id, NodeState.ERROR)
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
    
    # Finalize debug feedback
    if debug_feedback:
        for node_id, node in nodes.items():
            if hasattr(node, 'get_debug_info'):
                node_debug_info = node.get_debug_info()
                if node_debug_info and (node_debug_info.was_executed or node_debug_info.was_bypassed):
                    debug_feedback.add_node_info(node_debug_info)
        
        debug_feedback.finalize()
        yield {
            "type": "debug_summary",
            "content": debug_feedback.model_dump()
        }
        
        summary = dispatcher.get_execution_summary()
        logger.info(
            "Execution complete: %d completed, %d bypassed, %d errors",
            summary["completed"],
            summary["bypassed"],
            summary["errors"]
        )
    
    logger.info("Finished reactive execution")


async def execute_graph_loop_reactive(
    graph: AgentFlowModel,
    id_chat: Optional[Union[int, str]] = None,
    id_thread: Optional[Union[int, str]] = None,
    id_user: Optional[Union[int, str]] = None,
) -> AsyncGenerator[ChatCompletionModel, None]:
    """
    Execute an agent flow graph containing a Loop node using reactive model.
    
    This handles the special iteration semantics of loop nodes while still
    enabling parallel execution within each iteration.
    """
    import json
    from magic_agents.node_system import NodeLoop
    
    # Check for validation errors
    if hasattr(graph, '_validation_errors') and graph._validation_errors:
        for error in graph._validation_errors:
            yield {
                "type": SYSTEM_EVENT_DEBUG,
                "content": {
                    **error,
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
    
    nodes = graph.nodes
    chat_log = ModelAgentRunLog(
        id_chat=id_chat, id_thread=id_thread, id_user=id_user,
        id_app=getattr(graph, 'app_id', None) or getattr(graph, 'id_app', None)
    )
    
    logger.info(
        "Starting reactive loop execution: nodes=%d edges=%d",
        len(nodes), len(graph.edges)
    )
    
    # Initialize debug feedback
    debug_feedback: Optional[GraphDebugFeedback] = None
    if graph.debug:
        debug_feedback = GraphDebugFeedback(
            execution_id=uuid.uuid4().hex,
            graph_type=graph.type,
            start_time=datetime.now(UTC).isoformat()
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
    
    logger.debug(
        "Loop edges: static=%d item=%d loop_back=%d end=%d",
        len(static_edges), len(item_edges), len(loop_back_edges), len(end_edges)
    )
    
    # Create dispatcher for the full graph
    dispatcher = GraphEventDispatcher(nodes, graph.edges)
    
    # Helper to execute a single node inline
    async def execute_node_inline(node_id: str, edges_to_process: List = None):
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
        for edge in edges_for_inputs:
            if edge.target == node_id:
                source_node = nodes.get(edge.source)
                if source_node and source_node.outputs:
                    node.add_parent(source_node.outputs, edge.sourceHandle, edge.targetHandle)
        
        # Execute if not already done
        if node._response is None:
            logger.debug("Executing loop node %s", node_id)
            async for item in node(chat_log):
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
                else:
                    # All other outputs stored using their handle name
                    node.outputs[item_type] = item["content"]
            
            if debug_feedback and hasattr(node, 'get_debug_info'):
                node_debug_info = node.get_debug_info()
                if node_debug_info and node_debug_info.was_executed:
                    yield {
                        "type": SYSTEM_EVENT_DEBUG,
                        "content": node_debug_info.model_dump()
                    }
    
    # Import NodeConditional for type checking in static phase
    from magic_agents.node_system import NodeConditional
    
    # Track bypassed nodes during static phase
    bypassed_nodes: Set[str] = set()
    
    def propagate_bypass_static(node_id: str):
        """Recursively mark nodes as bypassed in static phase."""
        if node_id in bypassed_nodes:
            return
        bypassed_nodes.add(node_id)
        node = nodes.get(node_id)
        if node and hasattr(node, 'mark_bypassed'):
            node.mark_bypassed()
        logger.debug("Static phase: bypassing node %s", node_id)
        # Propagate to all downstream nodes in static edges
        for edge in static_edges:
            if edge.source == node_id:
                propagate_bypass_static(edge.target)
    
    def handle_conditional_bypass_static(cond_node_id: str, selected_handle: str):
        """Handle conditional bypass in static phase."""
        outgoing = [e for e in static_edges if e.source == cond_node_id]
        for edge in outgoing:
            if edge.sourceHandle != selected_handle:
                # This path is not selected - bypass it
                propagate_bypass_static(edge.target)
                logger.debug(
                    "Static conditional bypass: %s.%s -> %s (not selected)",
                    cond_node_id, edge.sourceHandle, edge.target
                )
    
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
            if edge.target == node_id:
                source_node = nodes.get(edge.source)
                target_node = nodes.get(node_id)
                if source_node and target_node and source_node.outputs:
                    if edge.sourceHandle in source_node.outputs:
                        target_node.add_parent(source_node.outputs, edge.sourceHandle, edge.targetHandle)
        
        # Execute the node
        async for out in execute_node_inline(node_id, static_edges):
            yield out
        
        # Handle conditional bypass propagation
        node = nodes.get(node_id)
        if isinstance(node, NodeConditional):
            selected_handle = getattr(node, 'selected_handle', None)
            if selected_handle:
                handle_conditional_bypass_static(node_id, selected_handle)
                logger.debug("Conditional %s selected handle: %s", node_id, selected_handle)
    
    # Transfer final outputs to loop node
    for edge in static_edges:
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
    
    # Find complete iteration subgraph using BFS (needed for bypass marking)
    iteration_subgraph = find_iteration_subgraph(loop_id, nodes, all_edges)
    logger.debug("Iteration subgraph nodes: %s", iteration_subgraph)
    
    # Handle case where loop input was bypassed
    if raw is None:
        # Check if the source of loop input was bypassed
        loop_input_source = None
        for edge in static_edges:
            if edge.target == loop_id and edge.targetHandle == loop_node.INPUT_HANDLE_LIST:
                loop_input_source = edge.source
                break
        
        if loop_input_source and loop_input_source in bypassed_nodes:
            logger.info("Loop input source was bypassed - skipping loop execution")
            # The loop path was bypassed by a conditional - skip to end
            loop_bypassed = True
            bypassed_nodes.add(loop_id)
            loop_node.mark_bypassed()
            # Mark all iteration subgraph nodes as bypassed
            for nid in iteration_subgraph:
                bypassed_nodes.add(nid)
                if nid in nodes and hasattr(nodes[nid], 'mark_bypassed'):
                    nodes[nid].mark_bypassed()
            # Mark all post-loop nodes that depend on loop output as bypassed
            for edge in end_edges:
                propagate_bypass_static(edge.target)
        else:
            # No input and not bypassed - this is an error
            error_msg = f"Loop node '{loop_id}' did not receive input on handle '{loop_node.INPUT_HANDLE_LIST}'"
            logger.error(error_msg)
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
            return
    elif not isinstance(items, list):
        error_msg = f"Loop node '{loop_id}' expects a list, got {type(items)}"
        logger.error(error_msg)
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
        return
    
    # Only execute loop iterations if not bypassed
    if not loop_bypassed:
        logger.info("Loop iterating over %d items", len(items))
        
        # Get topological order for iteration execution
        execution_order = topological_sort_iteration(iteration_subgraph, item_edges, loop_back_edges)
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
            
            # Reset loop state for this iteration
            loop_node._response = None
            loop_node.outputs.clear()
            # Clear the feedback input from previous iteration
            if loop_node.INPUT_HANDLE_LOOP in loop_node.inputs:
                del loop_node.inputs[loop_node.INPUT_HANDLE_LOOP]
            
            # Reset ALL nodes in the iteration subgraph (not just immediate downstream)
            reset_iteration_nodes(nodes, iteration_subgraph)
            
            # Set current item as loop output - PRESERVING TYPE (Issue #4 fix)
            loop_node.outputs[loop_node.OUTPUT_HANDLE_ITEM] = prepare_item_output(item, idx)
            
            # Process item edges - transfer loop item to first downstream nodes
            for edge in item_edges:
                target_node = nodes.get(edge.target)
                if target_node:
                    target_node.add_parent(loop_node.outputs, edge.sourceHandle, edge.targetHandle)
            
            # Execute iteration subgraph in TOPOLOGICAL ORDER (Issue #2 fix)
            # This ensures each node completes before its dependents start
            for node_id in execution_order:
                node = nodes.get(node_id)
                if not node:
                    continue
                
                # Apply inputs from any edges where source has completed
                for edge in item_edges + loop_back_edges:
                    if edge.target == node_id:
                        source_node = nodes.get(edge.source)
                        # Source could be loop node or another iteration node
                        if edge.source == loop_id:
                            node.add_parent(loop_node.outputs, edge.sourceHandle, edge.targetHandle)
                        elif source_node and source_node.outputs:
                            node.add_parent(source_node.outputs, edge.sourceHandle, edge.targetHandle)
                
                # Execute the node and WAIT for completion
                async for out in execute_node_inline(node_id, item_edges + loop_back_edges):
                    yield out
                
                # After execution, propagate outputs to downstream nodes in subgraph
                for edge in item_edges + loop_back_edges:
                    if edge.source == node_id:
                        target = nodes.get(edge.target)
                        if target and node.outputs:
                            target.add_parent(node.outputs, edge.sourceHandle, edge.targetHandle)
            
            # NOW collect the feedback AFTER all processing is complete (Issue #1 fix)
            # The feedback-producing node should have written to loop_node.inputs
            fb = loop_node.inputs.get(loop_node.INPUT_HANDLE_LOOP)
            
            # Extract actual content if wrapped in standard output format
            if isinstance(fb, dict) and 'content' in fb:
                fb = fb['content']
            
            logger.debug("Iteration %d feedback: %s", idx, str(fb)[:100] if fb else "None")
            loop_agg.append(fb)
        
        # Finish loop - set aggregated result
        loop_node._response = None
        loop_node.outputs.clear()
        loop_node.outputs[loop_node.OUTPUT_HANDLE_END] = loop_node.prep(loop_agg)
        
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
    logger.debug("Post-loop execution order: %s", post_loop_order)
    
    # First, transfer loop outputs to direct targets (only if loop wasn't bypassed)
    if not loop_bypassed:
        for edge in end_edges:
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
        async for out in execute_node_inline(node_id):
            yield out
        
        # Propagate outputs to downstream nodes within post_loop
        for edge in all_edges:
            if edge.source == node_id:
                target = nodes.get(edge.target)
                if target and node.outputs:
                    target.add_parent(node.outputs, edge.sourceHandle, edge.targetHandle)
    
    # Finalize debug feedback
    if debug_feedback:
        for node_id, node in nodes.items():
            if hasattr(node, 'get_debug_info'):
                node_debug_info = node.get_debug_info()
                if node_debug_info and (node_debug_info.was_executed or node_debug_info.was_bypassed):
                    debug_feedback.add_node_info(node_debug_info)
        
        debug_feedback.finalize()
        yield {
            "type": "debug_summary",
            "content": debug_feedback.model_dump()
        }
    
    logger.info("Finished reactive loop execution")
