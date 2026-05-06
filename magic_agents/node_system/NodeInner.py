from typing import Callable, Optional, TYPE_CHECKING, Any
import logging
import uuid
from datetime import datetime, UTC

# from magic_agents.agt_flow import build, execute_graph
from magic_agents.models.factory.Nodes import InnerNodeModel
from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalSignalTypes
from magic_agents.node_system.Node import Node

if TYPE_CHECKING:
    from magic_agents.models.factory.AgentFlowModel import AgentFlowModel

logger = logging.getLogger(__name__)


class NodeInner(Node):
    """
    Node to execute a nested agent flow graph.
    Handle names are configurable via JSON data.handles.
    JSON is the source of truth for all handle names.
    
    Supports:
    - Streaming forwarding from child flow to parent executor in real-time
    - Client extras propagation from parent flow to child flow
    - Parent state exposure via static key-path mapping or default full-state
    - Flow state isolation between parent and child flows
    """
    # Default handle names - can be overridden by JSON data.handles
    DEFAULT_INPUT_HANDLE = 'handle_user_message'
    DEFAULT_OUTPUT_CONTENT = 'handle_execution_content'
    DEFAULT_OUTPUT_EXTRAS = 'handle_execution_extras'
    DEFAULT_INPUT_CLIENT_EXTRAS = 'handle_client_extras'
    # Streaming output handle - follows NodeLLM pattern
    OUTPUT_HANDLE_CONTENT = 'handle_content_stream'

    def __init__(self, data: InnerNodeModel, load_chat: Callable, handles: Optional[dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.magic_flow = data.magic_flow
        self._load_chat = load_chat
        self.parent_state_mapping = data.parent_state_mapping  # Static key-path mapping for selective parent state exposure
        self.inner_graph: 'AgentFlowModel' = None  # Will be set by build()
        # Allow JSON to override handle names
        handles = handles or {}
        self.INPUT_HANDLE = handles.get('input', handles.get('user_message', self.DEFAULT_INPUT_HANDLE))
        self.HANDLER_EXECUTION_CONTENT = handles.get('output_content', handles.get('content', self.DEFAULT_OUTPUT_CONTENT))
        self.HANDLER_EXECUTION_EXTRAS = handles.get('output_extras', handles.get('extras', self.DEFAULT_OUTPUT_EXTRAS))
        self.HANDLER_CLIENT_EXTRAS = handles.get('client_extras', self.DEFAULT_INPUT_CLIENT_EXTRAS)

    async def process(self, chat_log):
        input_message = self.inputs.get(self.INPUT_HANDLE)
        if input_message is None:
            yield self.yield_debug_error(
                error_type="InputError",
                error_message=f"NodeInner requires input '{self.INPUT_HANDLE}'",
                context={
                    "available_inputs": list(self.inputs.keys()),
                    "required_input": self.INPUT_HANDLE
                }
            )
            yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
            return
        
        if self.inner_graph is None:
            # Provide clear error message based on why inner_graph wasn't built
            if self.magic_flow is None:
                error_message = "NodeInner requires magic_flow configuration (embedded child flow JSON)."
            elif not isinstance(self.magic_flow, dict):
                error_message = f"NodeInner magic_flow must be a dict, got {type(self.magic_flow).__name__}."
            else:
                magic_flow_keys = set(self.magic_flow.keys()) if isinstance(self.magic_flow, dict) else set()
                required_keys = {'nodes', 'edges'}
                missing_keys = required_keys - magic_flow_keys
                if missing_keys:
                    error_message = f"NodeInner magic_flow is malformed — missing required keys: {sorted(missing_keys)}."
                else:
                    error_message = "NodeInner inner_graph build failed. Check inner flow configuration."
            
            yield self.yield_debug_error(
                error_type="ConfigurationError",
                error_message=error_message,
                context={
                    "has_magic_flow": self.magic_flow is not None,
                    "magic_flow_type": type(self.magic_flow).__name__ if self.magic_flow else None,
                    "magic_flow_keys": list(self.magic_flow.keys()) if isinstance(self.magic_flow, dict) else None
                }
            )
            yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
            return
        
        # Retrieve client extras from inputs (backward compatible - defaults to {} if not provided)
        client_extras = self.inputs.get(self.HANDLER_CLIENT_EXTRAS, {})
        
        # Retrieve parent state from chat_log (backward compatible - defaults to {} if not present)
        parent_state = getattr(chat_log, 'flow_state', None) or {}
        
        # Prepare child extras by merging client extras and parent state
        child_extras = self._prepare_child_extras(client_extras, parent_state)
        
        # Update input nodes in the inner graph with the current message
        from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
        for node_id, node in self.inner_graph.nodes.items():
            node_type = getattr(node, 'node_type', None)
            if node_type in [ModelAgentFlowTypesModel.USER_INPUT, ModelAgentFlowTypesModel.CHAT]:
                if node_type == ModelAgentFlowTypesModel.USER_INPUT:
                    node._text = input_message  # Update the internal text directly
                elif node_type == ModelAgentFlowTypesModel.CHAT:
                    node.message = input_message
        
        # Execute the inner graph with extras and isolated flow_state.
        # We call execute_graph_reactive directly (rather than execute_graph) so we
        # can pass our HookRegistry natively — execute_graph's `hooks` param expects
        # a RuntimeConfig, which doesn't match self._hooks (a HookRegistry).
        from magic_agents.execution.reactive_executor import execute_graph_reactive
        from magic_agents.util.const import SYSTEM_EVENT_DEBUG
        content = ''
        extras = []
        inner_had_error = False
        
        # Phase 0: generate child run_id for execution tree persistence
        child_run_id = f"run-{uuid.uuid4().hex}"
        parent_execution_id = getattr(chat_log, 'run_id', None) or self.node_id
        
        # NOTE: We intentionally do NOT mutate chat_log.run_id / parent_run_id here.
        # The child graph receives a brand-new ModelAgentRunLog built from the
        # explicit run_id=/parent_run_id= kwargs below (lines below in execute_graph_reactive),
        # so the parent's chat_log is never read by the child. Mutating it would also
        # create a forward-looking race when sibling NodeInner instances share the
        # same parent chat_log concurrently.
        
        # Phase 0: yield SUBGRAPH_START debug event
        yield {
            "type": SYSTEM_EVENT_DEBUG,
            "content": {
                "event_type": "SUBGRAPH_START",
                "parent_execution_id": parent_execution_id,
                "child_run_id": child_run_id,
                "node_id": self.node_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        }
        
        # Phase 8.3: propagate parent hooks into the inner sub-graph and merge with
        # any graph-level hooks declared on the inner_graph itself. Mirrors the merge
        # pattern in agt_flow.execute_graph (priority: parent registry first, then
        # inner_graph.hooks registered on top).
        #
        # NOTE: debug_callback is intentionally NOT propagated to the child graph.
        # Child debug events already reach the parent's callback via the manual
        # `yield evt` forwarding in the loop below — propagating debug_callback
        # would cause every child debug event to be delivered twice (once via the
        # callback, once via the forwarded yield). See P0_REGRESSION_ANALYSIS.md
        # (Option γ) for full rationale.
        from magic_agents.hooks.hook_registry import HookRegistry
        _child_hooks = self._hooks.clone() if self._hooks is not None else HookRegistry()
        if self._hooks is not None:
            _child_hooks.execution_id = self._hooks.execution_id
            _child_hooks.run_id = self._hooks.run_id
        if self.inner_graph.hooks is not None:
            _child_hooks.register_graph(self.inner_graph.hooks)
        
        async for evt in execute_graph_reactive(
                self.inner_graph,
                id_chat=chat_log.id_chat,
                id_thread=chat_log.id_thread,
                id_user=chat_log.id_user,
                extras=child_extras,
                flow_state=None,  # Child gets isolated empty state
                run_id=child_run_id,           # Phase 0
                parent_run_id=parent_execution_id,  # Phase 0
                hooks=_child_hooks,
        ):
            # Propagate debug/error events from inner graph to outer graph
            if evt.get('type') == SYSTEM_EVENT_DEBUG:
                yield evt
                evt_content = evt.get('content', {})
                if evt_content.get('error_type'):
                    inner_had_error = True
                continue

            event = evt['content']
            # Check if event is a ChatCompletionModel
            if hasattr(event, 'choices') and event.choices:
                # It's a ChatCompletionModel
                event_content = event
                if event_content.choices[0].delta.content:
                    # Forward streaming chunk to parent executor in real-time (follows NodeLLM pattern)
                    yield self.yield_static(event_content, content_type=self.OUTPUT_HANDLE_CONTENT)
                    # Still collect for final output
                    content += event_content.choices[0].delta.content
                if hasattr(event_content, 'extras') and event_content.extras:
                    extras.append(event_content.extras)
            else:
                # It's some other type of output - try to convert to string
                if self.debug:
                    logger.debug("NodeInner:%s received non-ChatCompletionModel: %s", self.node_id, type(event))
                # For now, we'll skip non-ChatCompletionModel outputs
                # In a full implementation, you might want to handle these differently
                pass

        # Phase 0: yield SUBGRAPH_END debug event
        yield {
            "type": SYSTEM_EVENT_DEBUG,
            "content": {
                "event_type": "SUBGRAPH_END",
                "parent_execution_id": parent_execution_id,
                "child_run_id": child_run_id,
                "node_id": self.node_id,
                "status": "error" if inner_had_error else "completed",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        }

        # If inner graph had errors, signal bypass so downstream nodes don't hang
        if inner_had_error:
            logger.warning(
                "NodeInner:%s inner graph had errors — signaling BYPASS_ALL",
                self.node_id
            )
            yield {"type": ConditionalSignalTypes.BYPASS_ALL, "content": None}
            # Still yield whatever content was collected (may be partial)
            if content:
                yield self.yield_static(content, content_type=self.HANDLER_EXECUTION_CONTENT)
            if extras:
                yield self.yield_static(extras, content_type=self.HANDLER_EXECUTION_EXTRAS)
            return

        yield self.yield_static(content, content_type=self.HANDLER_EXECUTION_CONTENT)
        if extras:
            yield self.yield_static(extras, content_type=self.HANDLER_EXECUTION_EXTRAS)

    def _prepare_child_extras(self, client_extras: dict, parent_state: dict) -> dict:
        """
        Prepare extras dict for child flow execution.
        
        Merges client extras and parent state according to configuration:
        - If parent_state_mapping is configured: apply static key-path mapping
        - Otherwise: expose full parent state as 'parent_state' key
        
        Args:
            client_extras: Extras received from client/parent flow
            parent_state: Parent flow_state dict
            
        Returns:
            Prepared extras dict for child flow
        """
        # Start with client extras (copy to avoid mutation)
        child_extras = dict(client_extras) if client_extras else {}
        
        if self.parent_state_mapping:
            # Apply static key-path mapping for selective parent state exposure
            for child_key, parent_path in self.parent_state_mapping.items():
                child_extras[child_key] = _get_nested_value(parent_state, parent_path)
        else:
            # Default: expose full parent state
            child_extras['parent_state'] = parent_state
        
        return child_extras

    def _capture_internal_state(self):
        """Capture Inner-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add Inner-specific variables as documented
        state['has_magic_flow'] = self.magic_flow is not None
        state['has_inner_graph'] = self.inner_graph is not None
        state['parent_state_mapping'] = self.parent_state_mapping
        state['client_extras_received'] = self.inputs.get(self.HANDLER_CLIENT_EXTRAS)
        
        # Capture magic_flow summary if available
        if isinstance(self.magic_flow, dict):
            state['magic_flow'] = {
                'nodes_count': len(self.magic_flow.get('nodes', [])),
                'edges_count': len(self.magic_flow.get('edges', [])),
                'type': self.magic_flow.get('type', 'unknown')
            }
        
        # Capture inner graph info if available
        if self.inner_graph:
            state['inner_graph'] = f"<AgentFlowModel with {len(self.inner_graph.nodes)} nodes>"
        
        return state


def _get_nested_value(data: dict, path: str) -> Any:
    """
    Get nested value from dict using dot-notation path.
    
    Traverses nested dictionaries using dot-separated keys.
    Returns None if path not found (graceful handling, no error).
    
    Args:
        data: Source dictionary to traverse
        path: Dot-notation path (e.g., 'user.profile.name')
        
    Returns:
        Value at path, or None if path not found
        
    Examples:
        >>> _get_nested_value({'user': {'name': 'Alice'}}, 'user.name')
        'Alice'
        >>> _get_nested_value({'a': {'b': {'c': 123}}}, 'a.b.c')
        123
        >>> _get_nested_value({'x': 1}, 'y.z')  # Missing path
        None
    """
    if not data or not path:
        return None
    
    keys = path.split('.')
    value = data
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None  # Path not found
    
    return value
