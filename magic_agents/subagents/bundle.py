"""TaskToolBundle: Container for task tools to yield to NodeLLM.

Follows MCPToolBundle pattern for _collect_tools() integration.

NOTE: This is a THIN WRAPPER. Execution safeguards (depth, timeout, semaphore)
are handled by magic-llm's TaskExecutor. Registration happens via
MagicLLM.register_task() in NodeLLM.process() before agent loop execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import SubagentManifest
from .registry import RegistryBackend, get_registry
from .binder import Binder

logger = logging.getLogger(__name__)


@dataclass
class TaskToolBundle:
    """Container for task tools to yield to NodeLLM.
    
    THIN WRAPPER architecture:
    - tool_schemas: list of OpenAI-compatible tool definitions
    - tool_callables: dict of raw callables (NOT wrapped with safeguards)
    - manifests: list of SubagentManifest for registration
    
    Registration flow:
    1. NodeLLM._collect_tools() collects this bundle
    2. NodeLLM.process() calls MagicLLM.register_task() for each manifest+callable
    3. magic-llm's TaskExecutor wraps with safeguards and executes
    
    NOTE: tool_functions dict passed to AsyncAgentLoop contains callables,
    but TaskExecutor routes via _task_registry for task-specific handling.
    """
    
    # Required: tool definitions for LLM
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    
    # Raw callables for registration (NOT wrapped)
    tool_callables: dict[str, Callable] = field(default_factory=dict)
    
    # Manifests for TaskManifest conversion
    manifests: list[SubagentManifest] = field(default_factory=list)
    
    # Metadata for debugging/observability
    registered_count: int = 0
    source: str = "subagent_registry"
    
    # Backward compatibility: alias for tool_functions
    @property
    def tool_functions(self) -> dict[str, Callable]:
        """Alias for tool_callables (backward compatibility).
        
        NOTE: These are RAW callables, not wrapped with safeguards.
        Registration via MagicLLM.register_task() applies wrapping.
        """
        return self.tool_callables
    
    @classmethod
    async def from_registry(cls, registry: RegistryBackend) -> 'TaskToolBundle':
        """Build bundle from all registered subagents.
        
        Flow:
        1. List all manifests from registry
        2. For each enabled manifest:
           - Get callable from code registry
           - Validate signature via Binder
           - Add schema, callable, and manifest to bundle
        
        NOTE: Safeguards NOT applied here. Registration in magic-llm
        handles depth tracking, timeout, semaphore, and normalization.
        
        Args:
            registry: RegistryBackend instance
            
        Returns:
            TaskToolBundle with schemas, callables, and manifests
        """
        manifests = await registry.list()
        
        schemas = []
        callables = {}
        manifest_list = []
        
        for manifest in manifests:
            if not manifest.enabled:
                logger.debug(
                    "Skipping disabled subagent '%s'",
                    manifest.id
                )
                continue
            
            callable = registry.get_callable(manifest.id)
            if callable is None:
                logger.warning(
                    "No callable registered for manifest '%s' — skipping",
                    manifest.id
                )
                continue
            
            # Validate via Binder (signature check only)
            try:
                # Use join() which returns (manifest, callable) tuple
                validated_manifest, validated_callable = Binder.join(manifest, callable)
                
                schemas.append(validated_manifest.tool_schema)
                callables[validated_manifest.id] = validated_callable
                manifest_list.append(validated_manifest)
                
                logger.debug(
                    "Added subagent '%s' to bundle (pending registration)",
                    validated_manifest.id
                )
                
            except Exception as e:
                logger.error(
                    "Failed to validate subagent '%s': %s",
                    manifest.id,
                    e
                )
                continue
        
        logger.info(
            "Built TaskToolBundle with %d subagents (registration pending)",
            len(schemas)
        )
        
        return cls(
            tool_schemas=schemas,
            tool_callables=callables,
            manifests=manifest_list,
            registered_count=len(schemas)
        )
    
    @classmethod
    async def from_global_registry(cls) -> 'TaskToolBundle':
        """Build bundle from global registry.
        
        Convenience method for NodeLLM integration.
        
        Returns:
            TaskToolBundle from get_registry()
        """
        registry = get_registry()
        return await cls.from_registry(registry)