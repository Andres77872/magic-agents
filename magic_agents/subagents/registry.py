"""Registry backend protocol and static implementation.

RegistryBackend protocol for forward-compatibility with v2 database backends.
StaticManifestRegistry implements v1 file-based loading.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

from .models import SubagentManifest
from .errors import DuplicateSubagentError

logger = logging.getLogger(__name__)


class RegistryBackend(Protocol):
    """Protocol for subagent registry backend.
    
    v1: StaticManifestRegistry (YAML files)
    v2: DatabaseRegistry or APIRegistry can implement same protocol.
    
    This ensures NodeLLM and Binder are decoupled from registry implementation.
    """
    
    async def get(self, agent_id: str) -> Optional[SubagentManifest]:
        """Resolve manifest by ID.
        
        Args:
            agent_id: Subagent ID to lookup
            
        Returns:
            SubagentManifest if found, None otherwise
        """
        ...
    
    async def list(self) -> list[SubagentManifest]:
        """List all registered manifests.
        
        Returns:
            List of all SubagentManifest instances
        """
        ...
    
    def get_callable(self, agent_id: str) -> Optional[Callable]:
        """Get runtime callable from code registry.
        
        Args:
            agent_id: Subagent ID to lookup
            
        Returns:
            Callable if registered, None otherwise
        """
        ...
    
    def register_callable(self, agent_id: str, callable: Callable) -> None:
        """Register callable from decorator.
        
        Args:
            agent_id: Subagent ID
            callable: Async callable for execution
        """
        ...


class StaticManifestRegistry:
    """Static registry loading YAML manifests from directory.
    
    Implements RegistryBackend protocol for v1.
    
    Manifests are loaded at startup, callables registered via decorator.
    Duplicate IDs raise hard error (DuplicateSubagentError).
    """
    
    def __init__(self):
        self._manifests: Dict[str, SubagentManifest] = {}
        self._callables: Dict[str, Callable] = {}  # From decorator registry
        self._initialized: bool = False
    
    def register_manifest(self, manifest: SubagentManifest) -> None:
        """Register a manifest from loader.
        
        Args:
            manifest: SubagentManifest to register
            
        Raises:
            DuplicateSubagentError: If agent_id already registered
        """
        if manifest.id in self._manifests:
            existing = self._manifests[manifest.id]
            raise DuplicateSubagentError(
                agent_id=manifest.id,
                existing_source=str(existing.source_file),
                new_source=str(manifest.source_file)
            )
        
        self._manifests[manifest.id] = manifest
        logger.debug(
            "Registered subagent manifest '%s' (version %s)",
            manifest.id,
            manifest.version
        )
    
    def register_callable(self, agent_id: str, callable: Callable) -> None:
        """Register callable from decorator.
        
        Args:
            agent_id: Subagent ID
            callable: Async callable for execution
        """
        self._callables[agent_id] = callable
        logger.debug(
            "Registered callable for subagent '%s'",
            agent_id
        )
    
    async def get(self, agent_id: str) -> Optional[SubagentManifest]:
        """Resolve manifest by ID.
        
        Args:
            agent_id: Subagent ID to lookup
            
        Returns:
            SubagentManifest if found, None otherwise
        """
        return self._manifests.get(agent_id)
    
    async def list(self) -> list[SubagentManifest]:
        """List all registered manifests.
        
        Returns:
            List of all SubagentManifest instances
        """
        return list(self._manifests.values())
    
    def get_callable(self, agent_id: str) -> Optional[Callable]:
        """Get runtime callable from code registry.
        
        Args:
            agent_id: Subagent ID to lookup
            
        Returns:
            Callable if registered, None otherwise
        """
        return self._callables.get(agent_id)
    
    def list_callable_ids(self) -> list[str]:
        """List all registered callable IDs.
        
        Returns:
            List of agent_ids with registered callables
        """
        return list(self._callables.keys())
    
    def get_registered_ids(self) -> list[str]:
        """List all registered manifest IDs.
        
        Returns:
            List of registered agent_ids
        """
        return list(self._manifests.keys())
    
    def is_initialized(self) -> bool:
        """Check if registry has been initialized.
        
        Returns:
            True if init_registry() has been called
        """
        return self._initialized
    
    def mark_initialized(self) -> None:
        """Mark registry as initialized."""
        self._initialized = True


# Global registry instance (singleton for v1)
_GLOBAL_REGISTRY: Optional[StaticManifestRegistry] = None


async def init_registry(manifest_dir: Optional[Path] = None) -> StaticManifestRegistry:
    """Initialize the global registry.
    
    Loads manifests from directory if provided.
    Creates registry instance if not exists.
    
    Args:
        manifest_dir: Directory containing *.agent.yaml files
        
    Returns:
        The initialized StaticManifestRegistry instance
    """
    global _GLOBAL_REGISTRY
    
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = StaticManifestRegistry()
    
    if manifest_dir is not None:
        from .loader import ManifestLoader
        loader = ManifestLoader(manifest_dir)
        manifests = await loader.load_all()
        
        for manifest in manifests:
            _GLOBAL_REGISTRY.register_manifest(manifest)
        
        logger.info(
            "Loaded %d subagent manifests from %s",
            len(manifests),
            manifest_dir
        )
    
    _GLOBAL_REGISTRY.mark_initialized()
    return _GLOBAL_REGISTRY


def get_registry() -> StaticManifestRegistry:
    """Get the global registry instance.
    
    Returns:
        The StaticManifestRegistry instance
        
    Raises:
        RuntimeError: If registry not initialized
    """
    global _GLOBAL_REGISTRY
    
    if _GLOBAL_REGISTRY is None:
        raise RuntimeError(
            "Registry not initialized. Call init_registry() first."
        )
    
    return _GLOBAL_REGISTRY


def reset_registry() -> None:
    """Reset the global registry (for testing).
    
    Clears all manifests and callables.
    """
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = None