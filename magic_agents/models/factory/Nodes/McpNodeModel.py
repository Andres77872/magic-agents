"""MCP node configuration model.

Defines how MCP nodes are declared in agent graph JSON.
"""
import re
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator

from magic_agents.models.factory.Nodes.BaseNodeModel import BaseNodeModel


class MCPServerConfig(BaseModel):
    """Configuration for one MCP server connection.
    
    Supports stdio and Streamable HTTP transports.
    """
    model_config = ConfigDict(extra='allow')
    
    # Transport selection
    transport: Literal["stdio", "http"] = "stdio"
    
    # stdio transport config
    command: Optional[str] = None  # e.g., "uvx", "npx", "python"
    args: Optional[list[str]] = None  # e.g., ["mcp-server-filesystem", "/path"]
    env: Optional[dict[str, str]] = None  # Environment variables for subprocess
    cwd: Optional[str] = None  # Working directory for subprocess
    
    # HTTP transport config (Streamable HTTP)
    url: Optional[str] = None  # MCP endpoint URL
    headers: Optional[dict[str, str]] = None  # Auth headers (e.g., Authorization: Bearer token)
    
    # Timeout config (in seconds)
    init_timeout: float = Field(default=10.0, ge=1.0, le=120.0)
    tool_timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    discovery_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
    
    # Tool name prefix (defaults to node_id if not specified)
    prefix: Optional[str] = None
    
    # Tool filtering
    tool_allowlist: Optional[list[str]] = None  # Whitelist (None = all allowed)
    tool_denylist: Optional[list[str]] = None  # Blacklist
    
    @field_validator('url')
    @classmethod
    def validate_url_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate that HTTP URLs are valid HTTP/HTTPS endpoints."""
        if v is None:
            return v
        # Must be http:// or https:// URL
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP address
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        if not url_pattern.match(v):
            raise ValueError(f"url must be a valid HTTP or HTTPS URL: '{v}'")
        return v
    
    @model_validator(mode='after')
    def validate_transport_requirements(self):
        """Validate that required fields are present based on transport type."""
        if self.transport == 'stdio' and not self.command:
            raise ValueError("stdio transport requires 'command' field")
        if self.transport == 'http' and not self.url:
            raise ValueError("http transport requires 'url' field")
        return self


class McpNodeModel(BaseNodeModel):
    """MCP node configuration for graph JSON.
    
    Each MCP node connects to one MCP server and exposes discovered tools
    to downstream LLM nodes via MCPToolBundle.
    
    In v1, the servers list should have exactly one item (single server per node).
    """
    
    servers: list[MCPServerConfig] = Field(..., min_length=1)
    
    @model_validator(mode='before')
    @classmethod
    def validate_servers_not_empty(cls, data: Any) -> Any:
        """Reject empty servers list explicitly (even when passed as [])."""
        if isinstance(data, dict) and 'servers' in data:
            servers = data['servers']
            if servers is None or (isinstance(servers, list) and len(servers) == 0):
                raise ValueError("servers list must contain at least one server configuration")
        return data
    
    # Note: server.instructions from MCP init response are ignored (deferred)