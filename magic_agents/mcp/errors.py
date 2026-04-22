"""MCP error types.

Distinct error types for protocol-level, transport-level, and tool-level failures.
"""
from typing import Optional


class MCPProtocolError(Exception):
    """Protocol-level MCP error (JSON-RPC).
    
    Examples:
    - -32601: Method not found
    - -32602: Invalid params
    - -32603: Internal error
    
    These are hard errors that should interrupt execution.
    """
    
    def __init__(
        self,
        code: int,
        message: str,
        server: Optional[str] = None,
        tool: Optional[str] = None
    ):
        self.code = code
        self.message = message
        self.server = server
        self.tool = tool
        super().__init__(
            f"MCP protocol error (code={code}): {message}"
            + (f" [server={server}]" if server else "")
            + (f" [tool={tool}]" if tool else "")
        )


class MCPTransportError(Exception):
    """Transport-level MCP error.
    
    Examples:
    - Connection closed
    - Timeout
    - Process exited (stdio)
    - HTTP 404 session not found
    
    These indicate session health issues, not tool execution failures.
    """
    
    def __init__(
        self,
        message: str,
        server: Optional[str] = None,
        transport_type: Optional[str] = None
    ):
        self.message = message
        self.server = server
        self.transport_type = transport_type
        super().__init__(
            f"MCP transport error: {message}"
            + (f" [server={server}]" if server else "")
            + (f" [transport={transport_type}]" if transport_type else "")
        )


class MCPToolError(Exception):
    """Tool-level MCP error.
    
    This represents an isError:true result from tools/call.
    Unlike protocol errors, these should be returned to the LLM
    as tool result content, not raised as hard exceptions.
    
    Note: This exception is primarily used internally for logging
    and should NOT be raised to the agent loop in normal flow.
    """
    
    def __init__(
        self,
        tool_name: str,
        content: str,
        server: Optional[str] = None
    ):
        self.tool_name = tool_name
        self.content = content
        self.server = server
        super().__init__(
            f"MCP tool '{tool_name}' returned error: {content}"
            + (f" [server={server}]" if server else "")
        )


class MCPToolNameCollisionError(Exception):
    """Tool name collision error.
    
    Raised when multiple MCP nodes expose tools with the same
    prefixed name, which would cause ambiguity in tool execution.
    """
    
    def __init__(
        self,
        tool_name: str,
        source_nodes: list[str]
    ):
        self.tool_name = tool_name
        self.source_nodes = source_nodes
        super().__init__(
            f"Tool name collision: '{tool_name}' exposed by nodes {', '.join(source_nodes)}"
        )