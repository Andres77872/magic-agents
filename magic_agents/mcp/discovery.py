"""MCP Tool Discovery.

Handles paginated tools/list requests and tool caching.
"""
import logging
from typing import Optional, Any

from magic_agents.mcp.session import MCPSessionManager
from magic_agents.mcp.errors import MCPTransportError

logger = logging.getLogger(__name__)


class MCPToolDiscovery:
    """Discover tools from MCP server via paginated tools/list.
    
    Handles:
    - Paginated discovery (cursor loop)
    - Tool caching for session duration
    - Discovery timeout handling
    
    Deferred (v2):
    - tools/list_changed notification handling (dynamic refresh)
    """
    
    def __init__(
        self,
        session: MCPSessionManager,
        timeout: Optional[float] = None
    ):
        self._session = session
        self._timeout = timeout or session._config.discovery_timeout
        
        # Cache discovered tools
        self._cached_tools: Optional[list[dict]] = None
        self._discovered_at: Optional[float] = None  # Timestamp
    
    @property
    def tools(self) -> Optional[list[dict]]:
        """Cached tools if discovery completed."""
        return self._cached_tools
    
    async def list_tools(self) -> list[dict]:
        """Discover all tools from MCP server.
        
        Paginated loop until nextCursor is absent.
        
        Returns:
            List of raw tool definitions from MCP server
            Each tool: {name, description, inputSchema}
        
        Raises:
            MCPTransportError: If session unhealthy or timeout
        """
        if self._cached_tools is not None:
            # Return cached tools
            logger.debug(
                "MCPToolDiscovery:%s returning %d cached tools",
                self._session._node_id,
                len(self._cached_tools)
            )
            return self._cached_tools
        
        all_tools: list[dict] = []
        cursor: Optional[str] = None
        page_count = 0
        
        try:
            while True:
                page_count += 1
                
                # Request tools/list with cursor
                result = await self._session.list_tools(cursor=cursor)
                
                # Extract tools from result
                tools = self._extract_tools(result)
                all_tools.extend(tools)
                
                # Check for pagination cursor
                next_cursor = self._extract_cursor(result)
                
                if not next_cursor:
                    # No more pages
                    break
                
                cursor = next_cursor
                
                logger.debug(
                    "MCPToolDiscovery:%s page %d: %d tools, cursor=%s",
                    self._session._node_id,
                    page_count,
                    len(tools),
                    cursor[:20] + "..." if cursor else None
                )
                
                # Safety limit: don't loop forever
                if page_count > 100:
                    logger.warning(
                        "MCPToolDiscovery:%s exceeded 100 pages, stopping",
                        self._session._node_id
                    )
                    break
            
            # Cache discovered tools
            self._cached_tools = all_tools
            
            logger.info(
                "MCPToolDiscovery:%s discovered %d tools in %d pages from %s",
                self._session._node_id,
                len(all_tools),
                page_count,
                self._session.server_key
            )
            
            return all_tools
            
        except MCPTransportError:
            raise
        except Exception as e:
            logger.error(
                "MCPToolDiscovery:%s discovery failed: %s",
                self._session._node_id,
                e
            )
            raise MCPTransportError(
                message=f"Tool discovery failed: {str(e)}",
                server=self._session.server_key,
                transport_type=self._session._config.transport
            )
    
    def _extract_tools(self, result: Any) -> list[dict]:
        """Extract tool list from MCP tools/list result.
        
        Handles both SDK types and dict-like objects.
        """
        if hasattr(result, 'tools'):
            # SDK ListToolsResult type
            tools = result.tools
            if tools is None:
                return []
            
            # Convert SDK Tool types to dict
            tool_dicts = []
            for tool in tools:
                tool_dict = {
                    "name": tool.name if hasattr(tool, 'name') else tool.get("name", ""),
                    "description": tool.description if hasattr(tool, 'description') else tool.get("description", ""),
                    "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else tool.get("inputSchema", {})
                }
                tool_dicts.append(tool_dict)
            return tool_dicts
        
        # Dict-like result
        if isinstance(result, dict):
            return result.get("tools", [])
        
        return []
    
    def _extract_cursor(self, result: Any) -> Optional[str]:
        """Extract nextCursor from MCP tools/list result."""
        if hasattr(result, 'nextCursor'):
            return result.nextCursor
        
        if isinstance(result, dict):
            return result.get("nextCursor")
        
        return None
    
    def invalidate_cache(self) -> None:
        """Invalidate cached tools.
        
        Called when tools/list_changed notification received (deferred in v1).
        """
        self._cached_tools = None
        self._discovered_at = None
        
        logger.debug(
            "MCPToolDiscovery:%s cache invalidated",
            self._session._node_id
        )