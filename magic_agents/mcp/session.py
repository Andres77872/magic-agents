"""MCP Session Manager.

Handles MCP session lifecycle: connect, initialize handshake, tool calls, cleanup.
Encapsulates mcp SDK to prevent dependency leakage.
"""
import asyncio
import logging
from typing import Optional, Any
from contextlib import asynccontextmanager

# MCP SDK imports isolated here
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamablehttp_client

from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig
from magic_agents.mcp.errors import MCPProtocolError, MCPTransportError

logger = logging.getLogger(__name__)

# MCP protocol version we support
MCP_PROTOCOL_VERSION = "2025-03-26"


class MCPSessionManager:
    """Manages MCP session lifecycle for one server connection.
    
    Supports:
    - stdio transport: spawn subprocess, communicate via stdin/stdout
    - Streamable HTTP transport: POST to MCP endpoint
    
    Session is per-run scoped by default (created fresh each graph execution).
    """
    
    def __init__(
        self,
        config: MCPServerConfig,
        node_id: str,
        debug: bool = False
    ):
        self._config = config
        self._node_id = node_id
        self._debug = debug
        
        # Session state
        self._session: Optional[ClientSession] = None
        self._read_stream: Optional[Any] = None
        self._write_stream: Optional[Any] = None
        # SDK owns Mcp-Session-Id capture+reuse for HTTP; access via session_id property
        self._session_id_callback: Optional[Any] = None  # SDK callback for session ID query
        self._server_info: Optional[dict] = None
        self._capabilities: Optional[dict] = None
        self._protocol_version: Optional[str] = None
        
        # Transport context for cleanup
        self._transport_context: Optional[Any] = None
        
        # Session health
        self._is_healthy: bool = False
        self._is_initialized: bool = False
        
    @property
    def server_key(self) -> str:
        """Unique identifier for this server connection."""
        if self._config.transport == "stdio":
            return f"stdio:{self._config.command}"
        else:
            return f"http:{self._config.url}"
    
    @property
    def is_healthy(self) -> bool:
        """Check if session is healthy and usable."""
        return self._is_healthy and self._is_initialized
    
    @property
    def server_info(self) -> Optional[dict]:
        """Server info from initialize response."""
        return self._server_info
    
    @property
    def capabilities(self) -> Optional[dict]:
        """Server capabilities from initialize response."""
        return self._capabilities
    
    @property
    def session_id(self) -> Optional[str]:
        """HTTP session ID from Mcp-Session-Id header (observability only).
        
        Ownership contract: The MCP SDK (StreamableHTTPTransport) owns capture+reuse.
        This property queries the SDK's get_session_id() callback for observability.
        
        Returns:
            Session ID string if HTTP transport and initialized, else None.
        """
        if self._session_id_callback is None:
            return None
        return self._session_id_callback()
    
    async def connect(self) -> None:
        """Connect to MCP server and complete initialization handshake.
        
        Steps:
        1. Establish transport (spawn subprocess or HTTP connection)
        2. Send initialize request with client capabilities
        3. Send notifications/initialized
        
        Raises:
            MCPTransportError: If connection fails
            MCPProtocolError: If initialization fails
        """
        try:
            if self._config.transport == "stdio":
                await self._connect_stdio()
            else:
                await self._connect_http()
            
            # Complete initialization handshake
            await self._initialize_handshake()
            
            self._is_healthy = True
            self._is_initialized = True
            
            logger.info(
                "MCPSessionManager:%s connected to %s (server=%s, protocol=%s)",
                self._node_id,
                self.server_key,
                self._server_info.get("name", "unknown") if self._server_info else "unknown",
                self._protocol_version
            )
            
        except Exception as e:
            self._is_healthy = False
            self._is_initialized = False
            if isinstance(e, (MCPProtocolError, MCPTransportError)):
                raise
            raise MCPTransportError(
                message=str(e),
                server=self.server_key,
                transport_type=self._config.transport
            )
    
    async def _connect_stdio(self) -> None:
        """Connect via stdio transport - spawn subprocess."""
        server_params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args or [],
            env=self._config.env or None,
            cwd=self._config.cwd or None
        )
        
        # Create stdio client context
        self._transport_context = stdio_client(server_params)
        self._read_stream, self._write_stream = await self._transport_context.__aenter__()
        
        # Create MCP client session
        self._session = ClientSession(self._read_stream, self._write_stream)
        await self._session.__aenter__()
        
        if self._debug:
            logger.debug(
                "MCPSessionManager:%s spawned stdio process: %s %s",
                self._node_id,
                self._config.command,
                self._config.args or []
            )
    
    async def _connect_http(self) -> None:
        """Connect via Streamable HTTP transport."""
        headers = self._config.headers or {}
        
        # Create HTTP client context
        self._transport_context = streamablehttp_client(
            self._config.url,
            headers=headers
        )
        
        # Enter context and get streams
        result = await self._transport_context.__aenter__()
        # streamablehttp returns (read_stream, write_stream, session_id_callback)
        self._read_stream = result[0]
        self._write_stream = result[1]
        self._session_id_callback = result[2] if len(result) > 2 else None
        
        # Create MCP client session
        self._session = ClientSession(self._read_stream, self._write_stream)
        await self._session.__aenter__()
        
        if self._debug:
            logger.debug(
                "MCPSessionManager:%s connected to HTTP endpoint: %s",
                self._node_id,
                self._config.url
            )
    
    async def _initialize_handshake(self) -> None:
        """Complete MCP initialization handshake.
        
        1. Send initialize request
        2. Receive response with server info
        3. Send notifications/initialized
        """
        if not self._session:
            raise MCPTransportError(
                message="Session not established before initialize",
                server=self.server_key,
                transport_type=self._config.transport
            )
        
        # Send initialize request with empty capabilities (tool-only client)
        init_result = await asyncio.wait_for(
            self._session.initialize(),
            timeout=self._config.init_timeout
        )
        
        # Extract server info and capabilities
        self._server_info = init_result.serverInfo if hasattr(init_result, 'serverInfo') else {}
        self._capabilities = init_result.capabilities if hasattr(init_result, 'capabilities') else {}
        self._protocol_version = init_result.protocolVersion if hasattr(init_result, 'protocolVersion') else MCP_PROTOCOL_VERSION
        
        # HTTP Mcp-Session-Id ownership contract (recommendation B):
        # The MCP SDK's StreamableHTTPTransport owns session-id capture+reuse:
        # 1. Extracts Mcp-Session-Id from initialize response header
        # 2. Includes it in ALL subsequent request headers via _prepare_headers()
        # 3. Provides get_session_id() callback for observability
        # Magic-agents queries the callback via the session_id property, does NOT duplicate ownership.
        
        # Send notifications/initialized (required by protocol)
        # Note: SDK may handle this automatically, but we ensure it's sent
        # The ClientSession should send initialized notification after initialize()
        
        if self._debug:
            logger.debug(
                "MCPSessionManager:%s initialization complete: server=%s, capabilities=%s",
                self._node_id,
                self._server_info.get("name", "unknown"),
                list(self._capabilities.keys()) if self._capabilities else []
            )
    
    async def call_tool(
        self,
        name: str,
        arguments: dict,
        timeout: Optional[float] = None
    ) -> Any:
        """Call an MCP tool via session.
        
        Args:
            name: Remote tool name (not prefixed)
            arguments: Tool arguments matching inputSchema
            timeout: Optional timeout override (uses config.tool_timeout if not provided)
        
        Returns:
            MCP tool result (SDK CallToolResult object)
        
        Raises:
            MCPTransportError: If session is unhealthy
            MCPProtocolError: If JSON-RPC error
        """
        if not self.is_healthy:
            raise MCPTransportError(
                message="Session not healthy, cannot call tool",
                server=self.server_key,
                transport_type=self._config.transport
            )
        
        timeout_val = timeout or self._config.tool_timeout
        
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=timeout_val
            )
            return result
            
        except asyncio.TimeoutError:
            # Timeout - session may still be healthy, but this call timed out
            logger.warning(
                "MCPSessionManager:%s tool call '%s' timed out after %ss",
                self._node_id,
                name,
                timeout_val
            )
            # Return an error result instead of raising - LLM should see timeout
            from mcp.types import CallToolResult
            return CallToolResult(
                content=[{"type": "text", "text": f"Tool call timed out after {timeout_val} seconds"}],
                isError=True
            )
            
        except Exception as e:
            # Check if this is a JSON-RPC error
            error_code = getattr(e, 'code', None)
            if error_code:
                raise MCPProtocolError(
                    code=error_code,
                    message=str(e),
                    server=self.server_key,
                    tool=name
                )
            # Other errors - treat as transport error
            self._is_healthy = False
            raise MCPTransportError(
                message=f"Tool call failed: {str(e)}",
                server=self.server_key,
                transport_type=self._config.transport
            )
    
    async def list_tools(self, cursor: Optional[str] = None) -> Any:
        """List tools from MCP server (paginated).
        
        Args:
            cursor: Optional pagination cursor
        
        Returns:
            MCP tools/list result (SDK ListToolsResult object)
        """
        if not self.is_healthy:
            raise MCPTransportError(
                message="Session not healthy, cannot list tools",
                server=self.server_key,
                transport_type=self._config.transport
            )
        
        try:
            # SDK handles pagination internally if we pass cursor
            result = await asyncio.wait_for(
                self._session.list_tools(cursor=cursor),
                timeout=self._config.discovery_timeout
            )
            return result
            
        except asyncio.TimeoutError:
            raise MCPTransportError(
                message=f"Tool discovery timed out after {self._config.discovery_timeout}s",
                server=self.server_key,
                transport_type=self._config.transport
            )
    
    async def cleanup(self) -> None:
        """Cleanup session resources.
        
        For stdio: close stdin, wait for process exit
        For HTTP: close connection (DELETE session endpoint if supported)
        """
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(
                    "MCPSessionManager:%s session cleanup error: %s",
                    self._node_id,
                    e
                )
            self._session = None
        
        if self._transport_context:
            try:
                await self._transport_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(
                    "MCPSessionManager:%s transport cleanup error: %s",
                    self._node_id,
                    e
                )
            self._transport_context = None
        
        self._read_stream = None
        self._write_stream = None
        self._is_healthy = False
        self._is_initialized = False
        
        logger.info(
            "MCPSessionManager:%s cleaned up session for %s",
            self._node_id,
            self.server_key
        )