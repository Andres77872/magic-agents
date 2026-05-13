"""
RuntimeConfig for application-level hook configuration.

Injected at execution time for global hook registration.
Replaces module-level CallbackEmitter pattern with clean injection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Union

from magic_agents.hooks.flow_hooks import FlowHooks
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.hooks.persistence import ExecutionPersistencePort, GraphPersistenceHook
from magic_agents.hooks.debug_sse import DebugEventSink, DebugSSEHook

logger = logging.getLogger(__name__)


class RuntimeConfig:
    """Application-level configuration for hook system.
    
    Injected at execution time for global hook registration.
    Global hooks persist across all executions within the runtime context.
    
    Usage:
        # Register global hooks (application-scoped)
        RuntimeConfig.register_global_hook(my_logging_hook)
        RuntimeConfig.register_global_hook(my_metrics_hook)
        
        # Create execution-scoped registry
        config = RuntimeConfig()
        registry = config.create_registry()
        
        # Clear for testing
        RuntimeConfig.clear_global_hooks()
    
    Design Note: Global hooks are class-level (application-scoped), but
    HookRegistry is instance-level (execution-scoped). This prevents:
    - Memory leaks from long-running applications
    - Test pollution from cross-test hook leakage
    - CallbackEmitter singleton issues
    """
    
    # Class-level storage for application-scoped global hooks
    _global_hooks: List[FlowHooks] = []
    
    @classmethod
    def register_global_hook(cls, hook: FlowHooks) -> None:
        """Register global hook that applies to ALL executions.
        
        Global hooks fire for every graph execution regardless of instance.
        Use for application-level concerns: logging, metrics, tracing.
        
        Args:
            hook: FlowHooks implementation to register globally.
        
        Example:
            class LoggingHook:
                async def on_node_start(self, context):
                    logger.info(f"Node {context.node_id} starting")
            
            RuntimeConfig.register_global_hook(LoggingHook())
        """
        if not isinstance(hook, FlowHooks):
            logger.warning(
                "register_global_hook: object does not implement FlowHooks protocol, "
                "hook may not be invoked correctly"
            )
        cls._global_hooks.append(hook)
    
    @classmethod
    def clear_global_hooks(cls) -> None:
        """Clear all registered global hooks.
        
        Used for:
        - Test isolation (clear between tests)
        - Application reset (clear when shutting down)
        - Hot configuration changes (clear + re-register)
        
        Warning: This clears ALL global hooks for the entire application.
        Use carefully in production environments.
        """
        cls._global_hooks = []
        logger.debug("Global hooks cleared")
    
    @classmethod
    def get_global_hooks(cls) -> List[FlowHooks]:
        """Get copy of all registered global hooks.
        
        Returns a copy to prevent external mutation of internal list.
        
        Returns:
            Copy of global hooks list.
        """
        return list(cls._global_hooks)
    
    @classmethod
    def get_global_hooks_count(cls) -> int:
        """Get count of registered global hooks."""
        return len(cls._global_hooks)
    
    def __init__(self, graph_hooks: Optional[List[FlowHooks]] = None):
        """Initialize RuntimeConfig with optional graph-level hooks.
        
        Args:
            graph_hooks: Graph-level hooks for this configuration instance.
                        These hooks apply only when this config is used.
        """
        self._graph_hooks: List[FlowHooks] = list(graph_hooks or [])
        
        # Persistence configuration (P1-3)
        # Default: True — no-op without sink (build_persistence_hook returns None)
        self._persistence_enabled: bool = True
        self._persistence_sink: Optional[ExecutionPersistencePort] = None
        
        # DebugSSE configuration (P1-4)
        # Default: False — explicit opt-in required
        self._debug_sse_enabled: bool = False
        self._debug_sse_sink: Optional[DebugEventSink] = None
    
    def register_graph_hook(self, hook: FlowHooks) -> None:
        """Register graph-level hook for this config instance.
        
        Graph hooks fire only for executions using this config.
        
        Args:
            hook: FlowHooks implementation to register.
        """
        if not isinstance(hook, FlowHooks):
            logger.warning(
                "register_graph_hook: object does not implement FlowHooks protocol, "
                "hook may not be invoked correctly"
            )
        self._graph_hooks.append(hook)
    
    def get_graph_hooks(self) -> List[FlowHooks]:
        """Get copy of graph-level hooks for this instance."""
        return list(self._graph_hooks)
    
    def create_registry(self) -> HookRegistry:
        """Create execution-scoped registry with hooks injected.
        
        Combines:
        - Global hooks (class-level, application-scoped)
        - Graph hooks (instance-level, config-scoped)
        
        Node-level hooks must be registered directly on the registry
        after creation (execution-specific).
        
        Returns:
            HookRegistry instance ready for execution.
        
        Example:
            config = RuntimeConfig(graph_hooks=[my_hook])
            registry = config.create_registry()
            
            # Optionally register node-level hooks
            registry.register_node('node_123', specific_hook)
            
            # Pass to executor
            await execute_graph_reactive(graph, hooks=registry)
        """
        registry = HookRegistry(global_hooks=self.get_global_hooks())
        
        # Register graph-level hooks from this config instance
        for hook in self._graph_hooks:
            registry.register_graph(hook)
        
        return registry
    
    # ──────────────────────────────────────────────
    # P1-3: Persistence Configuration
    # ──────────────────────────────────────────────

    def is_persistence_enabled(self) -> bool:
        """Check if persistence hook auto-wiring is enabled.
        
        Returns:
            True if persistence_enabled is True, False otherwise.
            Note: returns True by default; without a configured sink,
            build_persistence_hook() still returns None (no-op).
        """
        return self._persistence_enabled

    def enable_persistence(self, sink: ExecutionPersistencePort) -> None:
        """Enable GraphPersistenceHook auto-wiring with the given sink.
        
        Sets persistence_enabled=True and configures the sink.
        Without a sink, build_persistence_hook() returns None.
        
        Args:
            sink: Concrete ExecutionPersistencePort implementation.
        """
        self._persistence_enabled = True
        self._persistence_sink = sink

    def build_persistence_hook(
        self,
        id_chat: str,
        id_thread: str,
        id_user: str,
        **kwargs: Any,
    ) -> Optional[GraphPersistenceHook]:
        """Build GraphPersistenceHook if persistence is enabled AND sink configured.
        
        Logs a WARNING when persistence_enabled=True but no sink is configured.
        The executor registers the returned hook on the existing HookRegistry.
        
        Args:
            id_chat: Chat/session identity.
            id_thread: Thread identity.
            id_user: User identity.
            **kwargs: Additional keyword arguments passed to GraphPersistenceHook
                     (e.g., id_agent, assistant_message, call_source,
                      nested_depth, nested_request_id, parent_run_id).
        
        Returns:
            GraphPersistenceHook instance when enabled and sink configured,
            None otherwise.
        """
        if not self._persistence_enabled:
            return None
        if self._persistence_sink is None:
            logger.warning(
                "persistence_enabled=True but no persistence_sink configured. "
                "GraphPersistenceHook not registered."
            )
            return None
        return GraphPersistenceHook(
            sink=self._persistence_sink,
            id_chat=id_chat,
            id_thread=id_thread,
            id_user=id_user,
            **kwargs,
        )

    # ──────────────────────────────────────────────
    # P1-4: DebugSSE Configuration
    # ──────────────────────────────────────────────

    def is_debug_sse_enabled(self) -> bool:
        """Check if DebugSSEHook auto-wiring is enabled.
        
        Returns:
            True if debug_sse_enabled is True, False otherwise (default).
        """
        return self._debug_sse_enabled

    def enable_debug_sse(
        self, sink: Union[asyncio.Queue, DebugEventSink]
    ) -> None:
        """Enable DebugSSEHook auto-wiring with the given sink.
        
        Sets debug_sse_enabled=True and configures the sink.
        Default is False — explicit opt-in required.
        
        Args:
            sink: SSE event sink (asyncio.Queue or DebugEventSink protocol).
        """
        self._debug_sse_enabled = True
        self._debug_sse_sink = sink

    def build_debug_sse_hook(self, id_chat: str) -> Optional[DebugSSEHook]:
        """Build DebugSSEHook if debug_sse is enabled AND sink configured.
        
        Args:
            id_chat: Chat/session identity for SSE events.
        
        Returns:
            DebugSSEHook instance when enabled and sink configured,
            None otherwise.
        """
        if not self._debug_sse_enabled or self._debug_sse_sink is None:
            return None
        return DebugSSEHook(sink=self._debug_sse_sink, id_chat=id_chat)

    @classmethod
    def has_global_hooks(cls) -> bool:
        """Check if global hooks are registered at the class level.
        
        Returns:
            True if any global hooks are registered, False otherwise.
        """
        return len(cls._global_hooks) > 0

    def is_empty(self) -> bool:
        """Check if this config has instance-level graph hooks.
        
        Returns:
            True if no graph hooks registered on this instance, False otherwise.
        """
        return len(self._graph_hooks) == 0