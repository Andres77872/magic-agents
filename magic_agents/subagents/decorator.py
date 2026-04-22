"""Decorator for runtime callable binding.

@task_subagent decorator registers callable in code registry.
Manifest YAML is the source of truth for metadata.
"""
from typing import Callable, Dict


# Global code registry (decorator populates this)
_CODE_REGISTRY: Dict[str, Callable] = {}


def task_subagent(agent_id: str) -> Callable:
    """Decorator for registering task subagent callable.
    
    Usage:
        @task_subagent("research.web")
        async def research_web(query: str) -> str:
            # Agent loop execution implementation
            ...
    
    The decorator only registers the callable in the code registry.
    Manifest YAML is the source of truth for metadata (id, schema, policy).
    
    At startup, Binder joins manifest with callable to produce BoundSubagent.
    
    Args:
        agent_id: Stable registry ID (must match manifest.id)
        
    Returns:
        Decorator function that registers the callable
    """
    def decorator(func: Callable) -> Callable:
        _CODE_REGISTRY[agent_id] = func
        func._subagent_id = agent_id  # Tag for introspection
        return func
    
    return decorator


def get_code_registry() -> Dict[str, Callable]:
    """Get the global code registry.
    
    Used by Binder to resolve decorated callables.
    
    Returns:
        Dict of agent_id -> callable
    """
    return _CODE_REGISTRY


def clear_code_registry() -> None:
    """Clear the code registry (for testing).
    
    Removes all registered callables.
    """
    _CODE_REGISTRY.clear()


def register_callable(agent_id: str, callable: Callable) -> None:
    """Manually register a callable (alternative to decorator).
    
    Useful for programmatic registration without decorator syntax.
    
    Args:
        agent_id: Subagent ID
        callable: Async callable for execution
    """
    _CODE_REGISTRY[agent_id] = callable
    callable._subagent_id = agent_id