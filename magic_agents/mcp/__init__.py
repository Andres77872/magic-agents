"""MCP adapter module.

Encapsulates MCP SDK and provides project-native types.
All mcp SDK imports are isolated within this module.
"""
from magic_agents.mcp.session import MCPSessionManager
from magic_agents.mcp.discovery import MCPToolDiscovery
from magic_agents.mcp.namespace import MCPToolNamespace, MappedTool
from magic_agents.mcp.dispatcher import MCPToolDispatcher
from magic_agents.mcp.bundle import MCPToolBundle
from magic_agents.mcp.errors import (
    MCPProtocolError,
    MCPTransportError,
    MCPToolError,
    MCPToolNameCollisionError,
)

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