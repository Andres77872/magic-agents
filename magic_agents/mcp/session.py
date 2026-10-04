"""MCP Session Manager.

Handles MCP session lifecycle: connect, initialize handshake, tool calls, cleanup.
Encapsulates mcp SDK to prevent dependency leakage.
"""
import asyncio
import copy
import logging
import inspect
from pydantic import BaseModel
from typing import Optional, Any
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass

# MCP SDK imports isolated here
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig
from magic_agents.mcp.errors import MCPProtocolError, MCPTransportError

logger = logging.getLogger(__name__)

# MCP protocol version we support
MCP_PROTOCOL_VERSION = "2025-03-26"


@dataclass(frozen=True)
class _DiscoveryOperation:
    session: object
    physical: object
    task: object


_DISCOVERY_OPERATION = ContextVar('mcp_complete_discovery_operation', default=None)


@asynccontextmanager
async def streamablehttp_client(url: str, headers: Optional[dict[str, str]] = None):
    """Adapt MCP SDK HTTP naming/header changes without affecting stdio imports.

    The SDK's current API takes an HTTP client rather than a headers keyword.
    Its factory supplies the matching HTTP implementation and timeout defaults.
    Older SDKs expose only the legacy streamablehttp_client helper.
    """
    from mcp.client import streamable_http

    modern_client = getattr(streamable_http, 'streamable_http_client', None)
    if modern_client is None:
        async with streamable_http.streamablehttp_client(url, headers=headers) as streams:
            yield streams
        return

    from mcp.shared._httpx_utils import create_mcp_http_client
    async with create_mcp_http_client(headers=headers) as http_client:
        async with modern_client(url, http_client=http_client) as streams:
            yield streams


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
        debug: bool = False,
        dispatch_session=None,
    ):
        self._config = config.model_copy(deep=True) if dispatch_session is not None else config
        self._node_id = node_id
        self._debug = debug
        self._dispatch = dispatch_session
        self._physical = None
        
        # Session state
        self._session: Optional[ClientSession] = None
        self._read_stream: Optional[Any] = None
        self._write_stream: Optional[Any] = None
        # SDK owns Mcp-Session-Id capture+reuse for HTTP; access via session_id property
        self._session_id_callback: Optional[Any] = None  # SDK callback for session ID query
        self._server_info: Optional[dict] = None
        self._capabilities: Optional[dict] = None
        self._protocol_version: Optional[str] = None
        self._server_instructions: Optional[str] = None
        
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
    def server_instructions(self) -> Optional[str]:
        """Server instructions from initialize response.
        
        Native MCP protocol field — provides instructions for using
        the server and its tools. Available when the MCP server
        includes an `instructions` field in its InitializeResult.
        Returns None when the server does not provide instructions.
        """
        return self._server_instructions
    
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
        if self._dispatch is not None:
            from magic_agents.coordination.service import CoordinationError
            self._dispatch.guard()
            if self._config.transport != 'http':
                raise CoordinationError('mcp_transport_unsupported', 'Coordinated MCP requires the qualified stateless HTTP profile')
            # Each physical HTTP request owns its reservation. A logical
            # connect must not hold the only job permit needed by its handshake.
        return await self._connect_uncontrolled()

    async def _connect_uncontrolled(self) -> None:
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
            if self._dispatch is not None:
                from magic_agents.coordination.dispatch import is_protected
                if is_protected(e):
                    raise
            if self._physical is not None:
                self._physical.check()
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
        if self._dispatch is not None:
            from magic_agents.mcp.controlled_http import controlled_http_client
            self._transport_context = controlled_http_client(self._config, self._dispatch)
        else:
            self._transport_context = streamablehttp_client(self._config.url, headers=headers)
        
        # Enter context and get streams
        result = await self._transport_context.__aenter__()
        if self._dispatch is not None:
            result, self._physical = result
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
        async def initialize():
            return await asyncio.wait_for(self._session.initialize(), timeout=self._config.init_timeout)
        init_result = await self._physical.run(initialize, timeout=self._config.init_timeout) if self._physical is not None else await initialize()
        
        # Extract server info, capabilities, and instructions
        # SDK 1.x/2.x Pydantic models expose different Python attribute names;
        # protocol aliases remain stable on the wire.
        if isinstance(init_result, BaseModel):
            init_data = init_result.model_dump(by_alias=True)
        elif isinstance(init_result, dict):
            init_data = init_result
        else:
            init_data = {key: getattr(init_result, key, None) for key in
                         ('serverInfo', 'capabilities', 'protocolVersion', 'instructions')}
        self._server_info = init_data.get('serverInfo') or {}
        self._capabilities = init_data.get('capabilities') or {}
        self._protocol_version = init_data.get('protocolVersion') or MCP_PROTOCOL_VERSION
        self._server_instructions = init_data.get('instructions')
        if self._server_instructions:
            logger.debug(
                "MCPSessionManager:%s server provides instructions (%d chars)",
                self._node_id,
                len(self._server_instructions)
            )
        
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
        if self._dispatch is not None:
            from magic_agents.coordination.dispatch import _encoded
            self._physical.check()
            arguments = copy.deepcopy(arguments)
            _encoded(arguments, max_bytes=1024 * 1024)
            timeout = timeout or self._config.tool_timeout
            async def operation():
                return await asyncio.wait_for(self._session.call_tool(name, arguments), timeout=timeout)
            return await self._physical.run(operation, timeout=timeout)
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
        if self._dispatch is not None:
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError('MCP discovery was cancelled before its next page')
            owner = _DISCOVERY_OPERATION.get()
            if (owner is not None and owner.session is self and owner.physical is self._physical
                    and owner.task is asyncio.current_task()):
                # Only the exact whole-discovery task can reuse this window.
                # Inherited contexts in unrelated child tasks do not qualify.
                self._physical.check()
                return await self._list_tools_uncontrolled(cursor)
            async def operation(): return await self._list_tools_uncontrolled(cursor)
            return await self._physical.run(operation, timeout=self._config.discovery_timeout)
        return await self._list_tools_uncontrolled(cursor)

    async def _run_discovery(self, operation, *, timeout):
        """Own one byte/deadline window across an entire paginated discovery."""
        if self._dispatch is None:
            return await operation()
        self._physical.check()
        async def bounded():
            owner = _DiscoveryOperation(self, self._physical, asyncio.current_task())
            token = _DISCOVERY_OPERATION.set(owner)
            try:
                return await operation()
            finally:
                _DISCOVERY_OPERATION.reset(token)
        try:
            return await self._physical.run(bounded, timeout=min(timeout, self._config.discovery_timeout))
        except BaseException as error:
            from magic_agents.coordination.dispatch import is_protected
            if is_protected(error):
                self._physical.fail(error)
            raise

    async def _list_tools_uncontrolled(self, cursor: Optional[str] = None) -> Any:
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
            # SDK 2.x replaced cursor= with PaginatedRequestParams. Detect the
            # signature before calling so TypeError from a request is not retried.
            parameters = inspect.signature(self._session.list_tools).parameters
            if 'params' in parameters and 'cursor' not in parameters:
                from mcp.types import PaginatedRequestParams
                request = self._session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
            else:
                request = self._session.list_tools(cursor=cursor)
            result = await asyncio.wait_for(
                request,
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
        if self._physical is not None:
            return await self._cleanup_controlled()
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

    async def _cleanup_controlled(self):
        """Exit SDK contexts in their owning task even if session exit fails."""
        from magic_agents.mcp.controlled_http import LOCAL_UNWIND_SECONDS
        failure = None
        for attribute in ('_session', '_transport_context'):
            context = getattr(self, attribute)
            if context is not None:
                try:
                    # asyncio's deadline adds no AnyIO cancel scope above the
                    # SDK's owner scope, preserving its strict LIFO exit order.
                    async with asyncio.timeout(LOCAL_UNWIND_SECONDS):
                        await context.__aexit__(None, None, None)
                except BaseException as error:
                    failure = failure or error
                finally:
                    setattr(self, attribute, None)
        self._read_stream = self._write_stream = None
        self._is_healthy = self._is_initialized = False
        if self._physical.failure is not None:
            raise self._physical.failure
        if failure is not None:
            if not isinstance(failure, Exception):
                raise failure
            from magic_agents.coordination.service import CoordinationError
            raise CoordinationError('mcp_cleanup_failed', 'MCP cleanup did not complete safely') from failure
