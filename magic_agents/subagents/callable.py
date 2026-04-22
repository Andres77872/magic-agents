"""TaskToolCallable: Thin schema provider — execution in magic-llm.

DEPRECATED: This file provides backward-compatible interface only.
Execution logic moved to magic_llm/agent/task_executor.py.

New code should call MagicLLM.register_task() directly:
    from magic_llm import MagicLLM
    from magic_llm.agent import TaskManifest
    
    client = MagicLLM(...)
    client.register_task(manifest, callable)
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

from .models import SubagentManifest, BoundSubagent


class TaskToolCallable:
    """Thin wrapper — schema provider only, execution delegated to magic-llm.
    
    DEPRECATED: New code should call MagicLLM.register_task() directly.
    This class exists for backward compatibility during transition.
    
    NOTE: __call__() raises NotImplementedError. Execution happens in
    magic-llm's TaskExecutor, which wraps the callable with safeguards.
    """
    
    def __init__(
        self,
        manifest: SubagentManifest,
        callable: Callable,
    ):
        """Initialize TaskToolCallable.
        
        Args:
            manifest: SubagentManifest with schema and policy
            callable: Decorated async function (passed to magic-llm)
        """
        self._manifest = manifest
        self._callable = callable
        self.__name__ = manifest.id  # For tool_functions dict
    
    @property
    def tool_schema(self) -> dict[str, Any]:
        """OpenAI-compatible tool schema."""
        return self._manifest.tool_schema
    
    @property
    def tool_callable(self) -> Callable:
        """Return the underlying callable (NOT wrapped).
        
        NOTE: The callable should be passed to MagicLLM.register_task()
        for proper wrapping with safeguards.
        """
        return self._callable
    
    async def __call__(self, **kwargs: Any) -> str:
        """DEPRECATED: Do not invoke directly.
        
        Registration should happen via MagicLLM.register_task().
        The callable is passed to magic-llm which wraps it with safeguards.
        
        Raises:
            NotImplementedError: Direct invocation is deprecated.
        """
        warnings.warn(
            "TaskToolCallable.__call__() is deprecated. "
            "Register via MagicLLM.register_task() before agent loop execution.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "Direct TaskToolCallable invocation is deprecated. "
            "Tasks must be registered via MagicLLM.register_task(). "
            "See magic_agents.subagents.__init__ for registration flow."
        )
    
    def get_manifest_and_callable(self) -> tuple[SubagentManifest, Callable]:
        """Get manifest and callable for MagicLLM.register_task().
        
        Returns:
            Tuple of (manifest, callable) for registration.
        """
        return self._manifest, self._callable