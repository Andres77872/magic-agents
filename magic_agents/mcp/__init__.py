"""MCP adapter module.

Encapsulates MCP SDK and provides project-native types.
All mcp SDK imports are isolated within this module.

LAZY IMPORT STRATEGY:
This module uses __getattr__ for lazy imports to prevent MCP SDK
from being imported at module load time. This allows non-MCP graphs
to run without triggering the mcp -> pydantic_settings import chain,
which can fail if pydantic version is incompatible.

The MCP SDK is only loaded when actually needed (e.g. when NodeMcp
is instantiated). This is safe because:
1. NodeMcpProxy in node_system/__init__.py delays NodeMcp class import
2. This module delays session.py import until symbols are accessed
3. Combined, non-MCP graphs never touch the mcp SDK

Import happens on first access of any symbol from __all__.
"""

_LAZY_MODULES = {
    "MCPSessionManager": "magic_agents.mcp.session",
    "MCPToolDiscovery": "magic_agents.mcp.discovery",
    "MCPToolNamespace": "magic_agents.mcp.namespace",
    "MappedTool": "magic_agents.mcp.namespace",
    "MCPToolDispatcher": "magic_agents.mcp.dispatcher",
    "MCPToolBundle": "magic_agents.mcp.bundle",
    "MCPProtocolError": "magic_agents.mcp.errors",
    "MCPTransportError": "magic_agents.mcp.errors",
    "MCPToolError": "magic_agents.mcp.errors",
    "MCPToolNameCollisionError": "magic_agents.mcp.errors",
}

_LOADED_MODULES: dict = {}


def __getattr__(name: str):
    """Lazy import symbols on first access."""
    if name not in _LAZY_MODULES:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    
    module_path = _LAZY_MODULES[name]
    
    if module_path not in _LOADED_MODULES:
        import importlib
        _LOADED_MODULES[module_path] = importlib.import_module(module_path)
    
    module = _LOADED_MODULES[module_path]
    attr = getattr(module, name)
    
    globals()[name] = attr
    return attr


__all__ = [
    "MCPSessionManager",
    "MCPToolDiscovery",
    "MCPToolNamespace",
    "MappedTool",
    "MCPToolDispatcher",
    "MCPToolBundle",
    "MCPProtocolError",
    "MCPTransportError",
    "MCPToolError",
    "MCPToolNameCollisionError",
]