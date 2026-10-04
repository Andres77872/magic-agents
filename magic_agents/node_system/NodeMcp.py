"""MCP Node implementation.

Tool-provider node that discovers MCP server tools and yields MCPToolBundle.
"""
import logging
from types import MappingProxyType
from typing import Optional, AsyncGenerator, Dict, Any

from magic_agents.models.factory.Nodes.McpNodeModel import McpNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.mcp import (
    MCPSessionManager,
    MCPToolDiscovery,
    MCPToolNamespace,
    MCPToolDispatcher,
    MCPToolBundle,
    MCPProtocolError,
    MCPTransportError,
)

logger = logging.getLogger(__name__)


class NodeMcp(Node):
    coordination_external_dispatch_version = 1
    coordination_external_transport_profiles = frozenset({'http-stateless-bounded-v1'})
    coordination_external_transport_versions = MappingProxyType({'mcp': '2.2.0', 'httpx2': '2.13.0', 'mcp-types': '2.2.0'})

    """MCP tool-provider node.
    
    Yields MCPToolBundle containing discovered tools from MCP server.
    Each tool name is prefixed with node_id to prevent collisions.
    
    Session lifecycle:
    - Per-run scope (fresh session each execution)
    - Cleanup in finally block guaranteed
    
    In v1:
    - One MCP server per node (servers list length = 1)
    - server.instructions collected and injected into tool manifest
    - tools/list_changed ignored during run (refresh on next execution)
    """
    
    # Default output handle for tool definitions
    DEFAULT_OUTPUT_HANDLE = 'handle-tool-definition'
    
    def __init__(
        self,
        data: McpNodeModel,
        node_id: str,
        handles: Optional[dict] = None,
        debug: bool = False,
        **kwargs
    ):
        # node_type is passed via kwargs from agt_flow.py create_node()
        # Do not pass it explicitly to avoid duplicate keyword argument error
        super().__init__(debug=debug, node_id=node_id, **kwargs)
        self._config = data
        self._session: Optional[MCPSessionManager] = None
        self._bundle: Optional[MCPToolBundle] = None
        
        # Allow JSON to override handle names
        handles = handles or {}
        self.OUTPUT_HANDLE = handles.get('output', self.DEFAULT_OUTPUT_HANDLE)
    
    async def process(self, chat_log) -> AsyncGenerator[Dict[str, Any], None]:
        """Initialize MCP session, discover tools, yield bundle.
        
        Steps:
        1. Validate single server (v1 constraint)
        2. Initialize MCP session (connect + init handshake)
        3. Discover tools via paginated tools/list
        4. Apply prefix and filter tools
        5. Build MCPToolBundle with schemas and wrappers
        6. Yield bundle to downstream LLM node
        7. Cleanup session in finally
        
        Errors:
        - Configuration, connection and discovery failures emit a diagnostic,
          then raise so the executor marks failure and bypasses dependents.
        - Cleanup runs before errors leave the node.
        """
        # Validate v1 constraint: exactly one server
        if len(self._config.servers) != 1:
            logger.error(
                "NodeMcp:%s v1 requires exactly 1 server (got %d)",
                self.node_id,
                len(self._config.servers)
            )
            yield self.yield_debug_error(
                error_type="ConfigurationError",
                error_message="MCP node requires exactly 1 server in v1",
                context={
                    "servers_count": len(self._config.servers),
                    "node_id": self.node_id
                }
            )
            raise ValueError("MCP node requires exactly 1 server in v1")
        
        server_config = self._config.servers[0]
        scope = getattr(chat_log, 'coordination', None)
        def create_session():
            options = {'dispatch_session': scope.dispatch_session(self.node_id)} if scope is not None else {}
            return MCPSessionManager(config=server_config, node_id=self.node_id, debug=self.debug, **options)
        
        try:
            # Step 1: Initialize session
            self._session = create_session()
            await self._session.connect()
            
            logger.info(
                "NodeMcp:%s connected to MCP server: %s",
                self.node_id,
                self._session.server_key
            )
            
            # Step 2: Discover tools
            discovery = MCPToolDiscovery(
                session=self._session,
                timeout=server_config.discovery_timeout
            )
            raw_tools = await discovery.list_tools()
            
            logger.info(
                "NodeMcp:%s discovered %d raw tools",
                self.node_id,
                len(raw_tools)
            )
            
            # Step 3: Apply prefix and filter
            namespace = MCPToolNamespace(
                prefix=server_config.prefix or self.node_id,
                allowed=server_config.tool_allowlist,
                denied=server_config.tool_denylist,
                server_key=self._session.server_key
            )
            
            # Filter first, then prefix
            filtered_tools = namespace.filter_tools(raw_tools)
            mapped_tools = namespace.apply_prefix(filtered_tools)
            
            logger.info(
                "NodeMcp:%s %d tools after filtering, prefix='%s'",
                self.node_id,
                len(mapped_tools),
                namespace.prefix
            )
            
            # Step 4: Build bundle
            dispatcher = MCPToolDispatcher(
                session=self._session,
                namespace=namespace,
                timeout=server_config.tool_timeout,
                session_factory=create_session,
                **({'coordinated': True} if scope is not None else {}),
            )
            self._bundle = dispatcher.build_bundle(mapped_tools, self.node_id)
            
            # Collect native MCP server.instructions (from InitializeResult)
            # Native MCP protocol field — no custom alias hacks or override wrappers
            if self._session and self._session.server_instructions:
                self._bundle.server_instructions = self._session.server_instructions
                logger.debug(
                    "NodeMcp:%s collected server.instructions (%d chars)",
                    self.node_id,
                    len(self._session.server_instructions)
                )
            
            # Step 5: Yield bundle to downstream LLM node
            yield self.yield_static(self._bundle, content_type=self.OUTPUT_HANDLE)
            
        except MCPProtocolError as e:
            logger.error(
                "NodeMcp:%s MCP protocol error: code=%d message=%s",
                self.node_id,
                e.code,
                e.message
            )
            yield self.yield_debug_error(
                error_type="MCPProtocolError",
                error_message=str(e),
                context={
                    "server": self._session.server_key if self._session else server_config.command or server_config.url,
                    "code": e.code,
                    "node_id": self.node_id
                }
            )
            raise
            
        except MCPTransportError as e:
            logger.error(
                "NodeMcp:%s MCP transport error: %s",
                self.node_id,
                e.message
            )
            yield self.yield_debug_error(
                error_type="MCPTransportError",
                error_message=str(e),
                context={
                    "server": self._session.server_key if self._session else server_config.command or server_config.url,
                    "transport_type": e.transport_type,
                    "node_id": self.node_id
                }
            )
            raise
            
        except Exception as e:
            logger.error(
                "NodeMcp:%s unexpected error: %s",
                self.node_id,
                e
            )
            yield self.yield_debug_error(
                error_type="UnexpectedError",
                error_message=f"MCP node execution failed: {str(e)}",
                context={
                    "exception_type": type(e).__name__,
                    "server": server_config.command or server_config.url,
                    "node_id": self.node_id
                }
            )
            raise
            
        finally:
            # Step 6: Cleanup session (guaranteed)
            if self._session:
                try:
                    await self._session.cleanup()
                except Exception as cleanup_error:
                    if scope is not None:
                        raise
                    logger.warning(
                        "NodeMcp:%s cleanup error: %s",
                        self.node_id,
                        cleanup_error
                    )
                self._session = None
    
    def _capture_internal_state(self) -> Dict[str, Any]:
        """Capture MCP-specific internal state for debugging."""
        state = super()._capture_internal_state()
        
        # Add MCP-specific state
        if self._session:
            state['server_key'] = self._session.server_key
            state['session_healthy'] = self._session.is_healthy
            state['server_info'] = self._session.server_info
        
        if self._bundle:
            state['bundle_tool_count'] = len(self._bundle.tool_schemas)
            state['bundle_prefix'] = self._bundle.prefix
        
        # Server config (safe to log)
        if self._config.servers:
            server = self._config.servers[0]
            state['transport_type'] = server.transport
            state['timeout_config'] = {
                "init": server.init_timeout,
                "tool": server.tool_timeout,
                "discovery": server.discovery_timeout
            }
        
        return state
