"""
ObserverRegistry — per-execution-scoped observer resolution.

Created by execute_graph_reactive() at execution start.
Lifecycle is fully owned by the executor — discarded on completion.
API layer does NOT create or hold this object; it passes a callback
that gets attached if debug is active.
"""

from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from magic_agents.debug.observer import DebugObserver
from magic_agents.debug.null_observer import NullObserver
from magic_agents.debug.composite_observer import CompositeObserver
from magic_agents.debug.default_observer import DefaultObserver
from magic_agents.debug.config import DebugConfig
from magic_agents.debug.emitter import EmitterRegistry, CallbackEmitter


class ObserverRegistry:
    """Per-execution-scoped observer resolution.

    Created by execute_graph_reactive() at execution start.
    Lifecycle is fully owned by the executor — discarded on completion.
    API layer does NOT create or hold this object; it passes a callback
    that gets attached if debug is active.
    """

    def __init__(
        self,
        graph_observer: DebugObserver,
        execution_id: str,
    ) -> None:
        """Initialize the registry with a graph-level observer.

        Args:
            graph_observer: The graph-level DebugObserver (could be NullObserver).
            execution_id: Unique trace identifier for this execution.
        """
        self._graph_observer = graph_observer
        self._execution_id = execution_id
        self._node_observers: Dict[str, Tuple[DebugObserver, bool]] = {}
        # bool = suppress_parent flag

    @staticmethod
    def create(
        debug_enabled_global: bool,
        graph_debug: bool,
        graph_debug_config: Optional[Dict[str, Any]],
        execution_id: str,
        graph_type: str,
        total_nodes: int,
        total_edges: int,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> "ObserverRegistry":
        """Factory: applies gate chain and returns configured registry.

        Decision chain:
        1. If not debug_enabled_global -> NullObserver (no callback attached)
        2. If not graph_debug -> NullObserver
        3. If debug_config says not enabled -> NullObserver
        4. Else -> DefaultObserver with callback attached via EmitterRegistry

        Args:
            debug_enabled_global: Global DEBUG_ENABLED env var value.
            graph_debug: Per-graph debug flag.
            graph_debug_config: Optional debug config dict from graph JSON.
            execution_id: Unique trace identifier.
            graph_type: Type of graph being executed.
            total_nodes: Total number of nodes in the graph.
            total_edges: Total number of edges in the graph.
            callback: Optional async callback from API layer for SSE events.
                     Attached to the observer's EmitterRegistry if debug is active.

        Returns:
            Configured ObserverRegistry instance.
        """
        if not debug_enabled_global:
            return ObserverRegistry(NullObserver(), execution_id)

        if not graph_debug:
            return ObserverRegistry(NullObserver(), execution_id)

        # Resolve debug config — graph_debug_config may be a DebugConfig
        # object (from AgentFlowModel.resolved_debug_config), a dict from
        # JSON, or None.  DebugConfig.from_dict() does NOT handle receiving
        # a DebugConfig object; passing one causes it to fall through to
        # default config with enabled=True, silently re-enabling debug.
        if isinstance(graph_debug_config, DebugConfig):
            config = graph_debug_config
        else:
            config = DebugConfig.from_dict(graph_debug_config or {})
        if not config.enabled:
            return ObserverRegistry(NullObserver(), execution_id)

        # Build DefaultObserver with EmitterRegistry + callback
        emitter_registry = EmitterRegistry()
        if callback is not None:
            cb_emitter = CallbackEmitter()
            cb_emitter.add_callback(callback)
            emitter_registry.register(cb_emitter)

        observer = DefaultObserver(
            execution_id=execution_id,
            graph_type=graph_type,
            config=config,
            emitter_registry=emitter_registry,
            total_nodes=total_nodes,
        )
        return ObserverRegistry(observer, execution_id)

    def observer_for(self, node_id: str, node: Any) -> DebugObserver:
        """Resolve observer for a specific node.

        Checks if the node provides a custom get_debug_observer().
        - If None: returns graph-level observer.
        - If non-None with suppress_parent=True: returns node observer only.
        - If non-None without suppress_parent: returns CompositeObserver (both fire).

        Args:
            node_id: The node's identifier.
            node: The node instance (expected to have get_debug_observer()).

        Returns:
            The resolved DebugObserver for this node.
        """
        node_observer = getattr(node, "get_debug_observer", lambda: None)()
        if node_observer is None:
            return self._graph_observer

        suppress = getattr(node_observer, "suppress_parent", False)
        if suppress:
            return node_observer

        return CompositeObserver(parent=self._graph_observer, child=node_observer)

    @property
    def graph_observer(self) -> DebugObserver:
        """Get the graph-level observer."""
        return self._graph_observer

    @property
    def is_active(self) -> bool:
        """True if the graph-level observer is NOT a NullObserver."""
        return not isinstance(self._graph_observer, NullObserver)
