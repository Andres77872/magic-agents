"""MCP Tool Namespace management.

Handles prefix generation, tool name mapping, and collision detection.
"""
import re
import logging
from dataclasses import dataclass
from typing import Optional

from magic_agents.mcp.errors import MCPToolNameCollisionError

logger = logging.getLogger(__name__)


@dataclass
class MappedTool:
    """A tool with local (prefixed) and remote (original) names."""
    local_name: str  # Prefixed name exposed to LLM
    remote_name: str  # Original name on MCP server
    description: str
    input_schema: dict  # MCP inputSchema (JSON Schema format)


class MCPToolNamespace:
    """Manages tool name prefixing and collision detection.
    
    All MCP tools are prefixed to prevent shadowing attacks and
    collisions when multiple MCP servers feed into one LLM node.
    
    Prefix generation:
    - Explicit prefix from config (preferred)
    - Auto-generated from node_id or server index
    
    Prefix format: `{prefix}_{original_tool_name}`
    Normalized: lowercase, alphanumeric + underscore only
    """
    
    def __init__(
        self,
        prefix: str,
        allowed: Optional[list[str]] = None,
        denied: Optional[list[str]] = None,
        server_key: Optional[str] = None
    ):
        self._raw_prefix = prefix
        self._prefix = self.normalize_prefix(prefix)
        self._allowed = allowed
        self._denied = denied
        self._server_key = server_key or ""
        
        # Track local→remote name mapping
        self._name_map: dict[str, str] = {}  # local_name → remote_name
        
    @property
    def prefix(self) -> str:
        """Normalized prefix."""
        return self._prefix
    
    @staticmethod
    def normalize_prefix(prefix: str) -> str:
        """Normalize prefix to safe format.
        
        - Lowercase
        - Alphanumeric + underscore only
        - No leading/trailing underscores
        
        Examples:
            "File-System" → "file_system"
            "mcp-server-filesystem" → "mcp_server_filesystem"
            "MyServer123" → "myserver123"
        """
        # Replace hyphens and spaces with underscores
        normalized = prefix.replace("-", "_").replace(" ", "_")
        # Remove non-alphanumeric characters (keep underscores)
        normalized = re.sub(r'[^\w]', '', normalized)
        # Lowercase
        normalized = normalized.lower()
        # Remove leading/trailing underscores
        normalized = normalized.strip("_")
        # Ensure non-empty
        if not normalized:
            normalized = "mcp"
        return normalized
    
    def filter_tools(self, tools: list[dict]) -> list[dict]:
        """Filter tools by allowlist/denylist.
        
        Args:
            tools: Raw tool definitions from MCP server
        
        Returns:
            Filtered list of tool definitions
        
        Rules:
        - If allowlist provided: only include listed tools
        - If denylist provided: exclude listed tools
        - If both: allowlist wins (intersection)
        - If neither: all tools included
        """
        if not self._allowed and not self._denied:
            # No filtering - all tools allowed
            return tools
        
        filtered = []
        for tool in tools:
            name = tool.get("name", "")
            
            # Check denylist first
            if self._denied and name in self._denied:
                logger.debug(
                    "MCPToolNamespace:%s excluded tool '%s' (in denylist)",
                    self._prefix,
                    name
                )
                continue
            
            # Check allowlist
            if self._allowed:
                if name not in self._allowed:
                    logger.debug(
                        "MCPToolNamespace:%s excluded tool '%s' (not in allowlist)",
                        self._prefix,
                        name
                    )
                    continue
            
            filtered.append(tool)
        
        logger.info(
            "MCPToolNamespace:%s filtered %d/%d tools (allowed=%s, denied=%s)",
            self._prefix,
            len(filtered),
            len(tools),
            self._allowed is not None,
            self._denied is not None
        )
        
        return filtered
    
    def apply_prefix(self, tools: list[dict]) -> list[MappedTool]:
        """Apply prefix to tool names and create MappedTool objects.
        
        Args:
            tools: Filtered tool definitions from MCP server
        
        Returns:
            List of MappedTool objects with prefixed names
        
        Raises:
            MCPToolNameCollisionError: If duplicate local names within this namespace
        """
        mapped = []
        seen_local_names: set[str] = set()
        
        for tool in tools:
            remote_name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema", {})
            
            # Generate local (prefixed) name
            local_name = f"{self._prefix}_{remote_name}"
            
            # Collision detection within this namespace
            if local_name in seen_local_names:
                raise MCPToolNameCollisionError(
                    tool_name=local_name,
                    source_nodes=[self._prefix]
                )
            
            seen_local_names.add(local_name)
            
            # Store mapping
            self._name_map[local_name] = remote_name
            
            mapped.append(MappedTool(
                local_name=local_name,
                remote_name=remote_name,
                description=description,
                input_schema=input_schema
            ))
        
        logger.info(
            "MCPToolNamespace:%s prefixed %d tools (prefix='%s_')",
            self._prefix,
            len(mapped),
            self._prefix
        )
        
        return mapped
    
    def get_remote_name(self, local_name: str) -> Optional[str]:
        """Get remote (original) tool name from local (prefixed) name.
        
        Args:
            local_name: Prefixed tool name
        
        Returns:
            Original tool name on MCP server, or None if not found
        """
        return self._name_map.get(local_name)
    
    def get_local_name(self, remote_name: str) -> str:
        """Get local (prefixed) name from remote (original) name.
        
        Args:
            remote_name: Original tool name on MCP server
        
        Returns:
            Prefixed tool name
        """
        return f"{self._prefix}_{remote_name}"
    
    def validate_collision_across_namespaces(
        self,
        other_namespaces: list["MCPToolNamespace"]
    ) -> None:
        """Check for collisions across multiple namespaces.
        
        This is called when multiple MCP nodes feed into one LLM node.
        
        Args:
            other_namespaces: Other namespace instances to check against
        
        Raises:
            MCPToolNameCollisionError: If same local name exists in multiple namespaces
        """
        # Collect all local names from other namespaces
        other_local_names: dict[str, list[str]] = {}
        for ns in other_namespaces:
            for local_name in ns._name_map.keys():
                if local_name not in other_local_names:
                    other_local_names[local_name] = []
                other_local_names[local_name].append(ns._prefix)
        
        # Check our names against others
        for local_name in self._name_map.keys():
            if local_name in other_local_names:
                sources = other_local_names[local_name] + [self._prefix]
                raise MCPToolNameCollisionError(
                    tool_name=local_name,
                    source_nodes=sources
                )