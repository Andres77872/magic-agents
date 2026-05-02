"""
HookRegistry for 3-tier hook dispatch with async-safe invocation.

Execution-scoped registry that combines global, graph-level, and node-level hooks.
Dies with execution (no module-level global state).
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Dict, Any

from magic_agents.hooks.flow_hooks import FlowHooks, HookContext

logger = logging.getLogger(__name__)


class HookRegistry:
    """3-tier hook registry with async-safe dispatch.
    
    Execution-scoped: passed to executor, dies with execution.
    No module-level global state (avoids CallbackEmitter pattern issues).
    
    Execution order: Node → Graph → Global (innermost-first).
    All hooks execute regardless of failures (asyncio.gather with return_exceptions).
    
    Carries execution identity (execution_id, run_id) set by the executor
    at execution start, enabling HookContext construction in downstream
    code (Node.__call__, relay) to include real identity values.
    """
    
    def __init__(self, global_hooks: Optional[List[FlowHooks]] = None):
        """Initialize registry with optional global hooks.
        
        Args:
            global_hooks: Pre-registered global hooks from RuntimeConfig.
        """
        self._global_hooks: List[FlowHooks] = list(global_hooks or [])
        self._graph_hooks: List[FlowHooks] = []
        self._node_hooks: Dict[str, List[FlowHooks]] = {}  # node_id → hooks
        
        # Execution identity — set by executor at execution start.
        # These flow into HookContext construction (Node.__call__, HookRelay)
        # so that production hook contexts carry real identity values.
        self.execution_id: str = ''
        self.run_id: str = ''
    
    def register_global(self, hook: FlowHooks) -> None:
        """Register global execution-level hook.
        
        Global hooks fire for ALL executions regardless of graph instance.
        """
        if not isinstance(hook, FlowHooks):
            logger.warning(
                "register_global: object does not implement FlowHooks protocol, "
                "hook may not be invoked correctly"
            )
        self._global_hooks.append(hook)
    
    def register_graph(self, hook: FlowHooks) -> None:
        """Register graph-level hook for this execution.
        
        Graph hooks fire only for this specific graph execution.
        """
        if not isinstance(hook, FlowHooks):
            logger.warning(
                "register_graph: object does not implement FlowHooks protocol, "
                "hook may not be invoked correctly"
            )
        self._graph_hooks.append(hook)
    
    def register_node(self, node_id: str, hook: FlowHooks) -> None:
        """Register node-specific hook.
        
        Node hooks fire only when the specified node executes.
        
        Args:
            node_id: Target node ID for this hook.
            hook: FlowHooks implementation to register.
        """
        if not isinstance(hook, FlowHooks):
            logger.warning(
                "register_node: object does not implement FlowHooks protocol, "
                "hook may not be invoked correctly"
            )
        if node_id not in self._node_hooks:
            self._node_hooks[node_id] = []
        self._node_hooks[node_id].append(hook)
    
    async def invoke(
        self,
        hook_name: str,
        context: HookContext,
        **kwargs: Any
    ) -> None:
        """Invoke all registered hooks at appropriate tiers.
        
        Execution order: Node → Graph → Global (innermost-first).
        All hooks execute regardless of failures (asyncio.gather with return_exceptions).
        
        Args:
            hook_name: Method name to invoke (e.g., 'on_node_start').
            context: HookContext with execution payload.
            **kwargs: Additional arguments for hook method (e.g., error, reason).
        """
        node_id = context.node_id
        
        # Collect hooks to invoke in order
        hooks_to_invoke: List[Any] = []
        
        # Tier 3: Node-level hooks (if node_id provided)
        if node_id and node_id in self._node_hooks:
            for hook in self._node_hooks[node_id]:
                method = getattr(hook, hook_name, None)
                if method is not None:
                    hooks_to_invoke.append((hook, method))
        
        # Tier 2: Graph-level hooks
        for hook in self._graph_hooks:
            method = getattr(hook, hook_name, None)
            if method is not None:
                hooks_to_invoke.append((hook, method))
        
        # Tier 1: Global hooks
        for hook in self._global_hooks:
            method = getattr(hook, hook_name, None)
            if method is not None:
                hooks_to_invoke.append((hook, method))
        
        # Invoke all hooks in parallel with error isolation
        if hooks_to_invoke:
            await self._invoke_all_safe(hooks_to_invoke, context, **kwargs)
    
    async def _invoke_all_safe(
        self,
        hooks_methods: List[tuple],
        context: HookContext,
        **kwargs: Any
    ) -> None:
        """Invoke multiple hooks safely with error isolation.
        
        Uses asyncio.gather with return_exceptions=True to ensure all hooks
        execute even if some fail. Exceptions are logged with full context.
        
        Args:
            hooks_methods: List of (hook_object, method) tuples.
            context: HookContext for invocation.
            **kwargs: Additional arguments for hook method.
        """
        tasks = []
        for hook, method in hooks_methods:
            if asyncio.iscoroutinefunction(method):
                tasks.append(self._invoke_async_safe(hook, method, context, **kwargs))
            else:
                # Wrap sync hooks in asyncio.to_thread for async-safe execution
                tasks.append(asyncio.to_thread(
                    self._invoke_sync_safe,
                    hook,
                    method,
                    context,
                    **kwargs
                ))
        
        # Execute all in parallel, exceptions logged but not propagated
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _invoke_async_safe(
        self,
        hook: Any,
        method: Any,
        context: HookContext,
        **kwargs: Any
    ) -> None:
        """Invoke async hook safely with error isolation.
        
        Exceptions are caught and logged with full context.
        Execution continues after hook failure.
        
        Args:
            hook: Hook object (for logging class name).
            method: Async method to invoke.
            context: HookContext payload.
            **kwargs: Additional arguments.
        """
        try:
            await method(context, **kwargs)
        except Exception as e:
            hook_class_name = hook.__class__.__name__ if hasattr(hook, '__class__') else 'anonymous'
            logger.warning(
                "Hook %s.%s failed: %s",
                hook_class_name,
                method.__name__,
                e,
                extra={"context": context.to_dict(), "exception_type": type(e).__name__}
            )
    
    def _invoke_sync_safe(
        self,
        hook: Any,
        method: Any,
        context: HookContext,
        **kwargs: Any
    ) -> None:
        """Invoke sync hook safely (for asyncio.to_thread wrapper).
        
        Exceptions are caught and logged with full context.
        Execution continues after hook failure.
        
        Args:
            hook: Hook object (for logging class name).
            method: Sync method to invoke.
            context: HookContext payload.
            **kwargs: Additional arguments.
        """
        try:
            method(context, **kwargs)
        except Exception as e:
            hook_class_name = hook.__class__.__name__ if hasattr(hook, '__class__') else 'anonymous'
            logger.warning(
                "Hook %s.%s failed: %s",
                hook_class_name,
                method.__name__,
                e,
                extra={"context": context.to_dict(), "exception_type": type(e).__name__}
            )
    
    def is_empty(self) -> bool:
        """Check if registry has no hooks registered.
        
        Used for lazy HookContext construction optimization.
        Empty registry = no behavior change (backward compatible).
        
        Returns:
            True if no hooks at any tier, False otherwise.
        """
        return (
            len(self._global_hooks) == 0
            and len(self._graph_hooks) == 0
            and len(self._node_hooks) == 0
        )
    
    def get_global_hooks_count(self) -> int:
        """Get count of registered global hooks."""
        return len(self._global_hooks)
    
    def get_graph_hooks_count(self) -> int:
        """Get count of registered graph hooks."""
        return len(self._graph_hooks)
    
    def get_node_hooks_count(self) -> int:
        """Get total count of node-level hooks across all nodes."""
        return sum(len(hooks) for hooks in self._node_hooks.values())
    
    def get_total_hooks_count(self) -> int:
        """Get total count of all registered hooks."""
        return self.get_global_hooks_count() + self.get_graph_hooks_count() + self.get_node_hooks_count()