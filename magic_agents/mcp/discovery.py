"""MCP Tool Discovery.

Handles paginated tools/list requests and tool caching.
"""
import logging
from pydantic import BaseModel
from typing import Optional, Any

from magic_agents.mcp.session import MCPSessionManager
from magic_agents.mcp.errors import MCPTransportError

logger = logging.getLogger(__name__)

MAX_DISCOVERY_PAGES = 100
MAX_DISCOVERY_CURSOR_BYTES = 4096
MAX_DISCOVERY_TOOLS = 1024
MAX_DISCOVERY_CACHE_BYTES = 1024 * 1024


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
        if getattr(self._session, '_dispatch', None) is not None:
            # A page is transport progress, not a new logical discovery budget.
            return await self._session._run_discovery(self._collect_tools, timeout=self._timeout)
        return await self._collect_tools()

    async def _collect_tools(self) -> list[dict]:
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
        coordinated = getattr(self._session, '_dispatch', None) is not None
        seen_cursors, seen_names = set(), set()
        
        try:
            while True:
                page_count += 1
                
                # Request tools/list with cursor
                result = await self._session.list_tools(cursor=cursor)
                
                # Extract tools from result
                tools = self._extract_tools(result)
                if coordinated:
                    from magic_agents.coordination.dispatch import _encoded
                    from magic_agents.coordination.service import CoordinationError
                    if len(all_tools) + len(tools) > MAX_DISCOVERY_TOOLS:
                        raise CoordinationError('mcp_discovery_limit', 'MCP discovery exceeds its retained tool count')
                    names = [tool['name'] for tool in tools]
                    if len(set(names)) != len(names) or any(name in seen_names for name in names):
                        raise CoordinationError('mcp_discovery_invalid', 'MCP discovery contains ambiguous duplicate tool names')
                    candidate = all_tools + tools
                    _encoded(candidate, max_bytes=MAX_DISCOVERY_CACHE_BYTES)
                    # Assign only after both retained-count and encoded-cache
                    # bounds pass; rejected discoveries never publish a cache.
                    all_tools = candidate
                    seen_names.update(names)
                else:
                    all_tools.extend(tools)
                
                # Check for pagination cursor
                next_cursor = self._extract_cursor(result)
                
                if not next_cursor:
                    # No more pages
                    break

                if coordinated:
                    if (type(next_cursor) is not str
                            or len(next_cursor.encode('utf-8')) > MAX_DISCOVERY_CURSOR_BYTES):
                        raise CoordinationError('mcp_discovery_limit', 'MCP discovery cursor exceeds its byte ceiling')
                    if next_cursor in seen_cursors:
                        raise CoordinationError('mcp_discovery_invalid', 'MCP discovery repeated a pagination cursor')
                    if page_count >= MAX_DISCOVERY_PAGES:
                        raise CoordinationError('mcp_discovery_limit', 'MCP discovery ended before its final page')
                    seen_cursors.add(next_cursor)
                
                cursor = next_cursor
                
                logger.debug(
                    "MCPToolDiscovery:%s page %d: %d tools, cursor=%s",
                    self._session._node_id,
                    page_count,
                    len(tools),
                    cursor[:20] + "..." if cursor else None
                )
                
                # Safety limit: don't loop forever
                if not coordinated and page_count > 100:
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
            if getattr(self._session, '_dispatch', None) is not None:
                from magic_agents.coordination.dispatch import is_protected
                if is_protected(e): raise
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
        if isinstance(result, BaseModel):
            result = result.model_dump(by_alias=True)
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
        if isinstance(result, BaseModel):
            result = result.model_dump(by_alias=True)
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
