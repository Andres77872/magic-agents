"""
magic_agents.hooks package - Observer-first hook system.

This package provides the hook infrastructure for magic-agents:
- FlowHooks: Protocol for observer-only lifecycle hooks
- HookContext: Payload dataclass for hook invocations
- HookRegistry: 3-tier registration with async-safe dispatch
- RuntimeConfig: Application-level hook configuration
- EmitInterface: Helper for hook function outputs

Contracts:
- All hooks are observer-only (no state modification)
- Async-first (asyncio.gather for parallel invocation)
- Error-isolated (exceptions logged, execution continues)
- Backward compatible (empty registry = no behavior change)

Usage:
    from magic_agents.hooks import FlowHooks, HookContext, RuntimeConfig
    
    # Define a hook implementation
    class LoggingHook:
        async def on_node_start(self, context: HookContext):
            logger.info(f"Node {context.node_id} starting")
    
    # Register globally (application-scoped)
    RuntimeConfig.register_global_hook(LoggingHook())
    
    # Create registry for execution
    config = RuntimeConfig()
    registry = config.create_registry()
    
    # Pass to executor
    await execute_graph_reactive(graph, hooks=registry)

Note: Node-specific context extensions (NodeLLMHookContext, etc.) are
available via flow_hooks module for specialized use cases.
"""

# Core exports - no circular dependencies
from magic_agents.hooks.flow_hooks import (
    FlowHooks,
    HookContext,
)

from magic_agents.hooks.hook_registry import HookRegistry

from magic_agents.hooks.runtime_config import RuntimeConfig

from magic_agents.hooks.emit_context import EmitInterface

from magic_agents.hooks.hook_relay import HookRelay

from magic_agents.hooks.persistence import (
    AssistantMessageContext,
    ExecutionPersistencePort,
    GraphPersistenceHook,
    PersistenceHook,
)

from magic_agents.hooks.debug_sse import DebugSSEHook

from magic_agents.hooks import contracts
from magic_agents.hooks.context_factory import HookContextFactory


# Public API
__all__ = [
    # Protocol
    "FlowHooks",
    
    # Context
    "HookContext",
    
    # Registry
    "HookRegistry",
    
    # Configuration
    "RuntimeConfig",
    
    # Emit helpers
    "EmitInterface",
    
    # Relay adapter
    "HookRelay",

    # Reusable hook implementations
    "AssistantMessageContext",
    "ExecutionPersistencePort",
    "GraphPersistenceHook",
    "PersistenceHook",
    "DebugSSEHook",

    # Contracts & Factory
    "contracts",
    "HookContextFactory",
]


# Package version
__version__ = "0.1.0"
