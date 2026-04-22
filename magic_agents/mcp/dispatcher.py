"""MCP Tool Dispatcher.

Creates async callable wrappers for MCP tools and builds MCPToolBundle.
"""
import json
import logging
from typing import Any, Callable, Optional

from magic_agents.mcp.session import MCPSessionManager
from magic_agents.mcp.namespace import MCPToolNamespace, MappedTool
from magic_agents.mcp.bundle import MCPToolBundle
from magic_agents.mcp.errors import MCPProtocolError, MCPToolError

logger = logging.getLogger(__name__)


def validate_arguments_against_schema(
    arguments: dict,
    schema: dict,
    tool_name: str
) -> tuple[bool, Optional[str]]:
    """Validate arguments against JSON Schema inputSchema.
    
    Args:
        arguments: Keyword arguments passed to tool
        schema: MCP inputSchema (JSON Schema format)
        tool_name: Tool name for error messages
    
    Returns:
        (is_valid, error_message): If invalid, error_message describes what's wrong
    """
    if not schema:
        # No schema defined - accept any arguments
        return True, None
    
    # Check required fields
    required_fields = schema.get("required", [])
    if required_fields:
        missing = []
        for field in required_fields:
            if field not in arguments or arguments[field] is None:
                missing.append(field)
        
        if missing:
            return False, f"Missing required arguments for tool '{tool_name}': {missing}"
    
    # Basic type validation for properties (optional but helpful)
    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name in properties:
            prop_schema = properties[arg_name]
            expected_type = prop_schema.get("type")
            
            if expected_type and arg_value is not None:
                # Simple type checking
                valid = True
                if expected_type == "string" and not isinstance(arg_value, str):
                    valid = False
                elif expected_type == "number" and not isinstance(arg_value, (int, float)):
                    valid = False
                elif expected_type == "integer" and not isinstance(arg_value, int):
                    valid = False
                elif expected_type == "boolean" and not isinstance(arg_value, bool):
                    valid = False
                elif expected_type == "array" and not isinstance(arg_value, list):
                    valid = False
                elif expected_type == "object" and not isinstance(arg_value, dict):
                    valid = False
                
                if not valid:
                    return False, f"Argument '{arg_name}' for tool '{tool_name}' has wrong type. Expected '{expected_type}', got '{type(arg_value).__name__}'"
    
    return True, None


class MCPToolDispatcher:
    """Creates async callable wrappers for MCP tools.
    
    Each wrapper:
    1. Accepts keyword arguments matching tool's inputSchema
    2. Calls MCP session.call_tool(remote_name, arguments)
    3. Returns result content as string
    
    isError handling:
    - isError:true → return content to LLM (not raise)
    - Protocol errors → raise MCPProtocolError
    
    structuredContent handling:
    - Prefer structuredContent when present
    - JSON-serialize for tool result pipeline
    """
    
    def __init__(
        self,
        session: MCPSessionManager,
        namespace: MCPToolNamespace,
        timeout: Optional[float] = None
    ):
        self._session = session
        self._namespace = namespace
        self._timeout = timeout or session._config.tool_timeout
    
    def build_bundle(
        self,
        mapped_tools: list[MappedTool],
        node_id: str
    ) -> MCPToolBundle:
        """Build MCPToolBundle with schemas and async callable wrappers.
        
        Args:
            mapped_tools: Tools with prefixed names
            node_id: Graph node ID for metadata
        
        Returns:
            MCPToolBundle ready for yield_static()
        """
        schemas: list[dict[str, Any]] = []
        functions: dict[str, Callable] = {}
        
        for mapped in mapped_tools:
            # OpenAI-compatible schema
            schema = self._build_schema(mapped)
            schemas.append(schema)
            
            # Create async wrapper callable
            wrapper = self._create_wrapper(mapped)
            functions[mapped.local_name] = wrapper
        
        logger.info(
            "MCPToolDispatcher:%s built bundle with %d tools",
            node_id,
            len(schemas)
        )
        
        return MCPToolBundle(
            tool_schemas=schemas,
            tool_functions=functions,
            server_key=self._session.server_key,
            node_id=node_id,
            discovered_count=len(mapped_tools),
            filtered_count=len(mapped_tools),
            prefix=self._namespace.prefix
        )
    
    def _build_schema(self, mapped: MappedTool) -> dict[str, Any]:
        """Build OpenAI-compatible tool schema from MappedTool.
        
        MCP inputSchema maps directly to OpenAI function.parameters.
        """
        return {
            "type": "function",
            "function": {
                "name": mapped.local_name,
                "description": mapped.description or f"MCP tool: {mapped.remote_name}",
                "parameters": mapped.input_schema
            }
        }
    
    def _create_wrapper(self, mapped: MappedTool) -> Callable:
        """Create async callable wrapper for MCP tool.
        
        The wrapper:
        1. Captures remote_name and session reference
        2. Validates arguments against inputSchema locally
        3. Calls session.call_tool(remote_name, kwargs) only if valid
        4. Returns serialized result
        
        isError:true handling: return content, not raise
        Local validation error: return error message, do NOT call remote
        """
        # Capture values for closure
        remote_name = mapped.remote_name
        local_name = mapped.local_name
        session = self._session
        timeout = self._timeout
        input_schema = mapped.input_schema
        
        async def wrapper(**kwargs) -> str:
            """Execute MCP tool call with local validation."""
            # Local validation against inputSchema BEFORE remote call
            is_valid, error_msg = validate_arguments_against_schema(
                arguments=kwargs,
                schema=input_schema,
                tool_name=local_name
            )
            
            if not is_valid:
                logger.warning(
                    "MCPToolDispatcher tool '%s' validation failed: %s",
                    local_name,
                    error_msg
                )
                # Return error message to LLM - DO NOT call remote tool
                return json.dumps({
                    "error": "validation_failed",
                    "tool": local_name,
                    "message": error_msg
                })
            
            # Arguments valid - proceed with remote call
            return await self._call_tool_internal(
                remote_name=remote_name,
                local_name=local_name,
                arguments=kwargs,
                timeout=timeout
            )
        
        # Set __name__ for magic-llm registration
        wrapper.__name__ = local_name
        
        return wrapper
    
    async def _call_tool_internal(
        self,
        remote_name: str,
        local_name: str,
        arguments: dict,
        timeout: float
    ) -> str:
        """Internal tool call implementation.
        
        Args:
            remote_name: Original tool name on MCP server
            local_name: Prefixed name for logging
            arguments: Tool arguments
            timeout: Call timeout
        
        Returns:
            Serialized result string for LLM
        
        Note:
            isError:true → returns error content (not raises)
            Protocol errors → raised as MCPProtocolError
        """
        logger.debug(
            "MCPToolDispatcher calling tool '%s' (remote='%s') with args=%s",
            local_name,
            remote_name,
            list(arguments.keys())
        )
        
        try:
            result = await self._session.call_tool(
                name=remote_name,
                arguments=arguments,
                timeout=timeout
            )
            
            # Process result
            return self._serialize_result(result, local_name)
            
        except MCPProtocolError:
            # Protocol error - re-raise
            raise
        except MCPToolError as e:
            # Tool-level error - return content (already handled in serialize)
            return e.content
        except Exception as e:
            # Unexpected error - return as tool error
            logger.error(
                "MCPToolDispatcher tool '%s' unexpected error: %s",
                local_name,
                e
            )
            return json.dumps({"error": f"Tool execution error: {str(e)}"})
    
    def _serialize_result(self, result: Any, tool_name: str) -> str:
        """Serialize MCP tool result for LLM.
        
        Priority:
        1. structuredContent (if present) → JSON-serialize
        2. content[] array → concatenate text items
        
        isError handling:
        - Log warning but return content for LLM to reason about
        """
        # Check isError flag
        is_error = False
        if hasattr(result, 'isError'):
            is_error = result.isError
        elif isinstance(result, dict):
            is_error = result.get("isError", False)
        
        if is_error:
            logger.warning(
                "MCPToolDispatcher tool '%s' returned isError=true",
                tool_name
            )
        
        # Prefer structuredContent if present
        structured = None
        if hasattr(result, 'structuredContent'):
            structured = result.structuredContent
        elif isinstance(result, dict):
            structured = result.get("structuredContent")
        
        if structured is not None:
            # JSON-serialize structured content
            try:
                return json.dumps(structured)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "MCPToolDispatcher failed to serialize structuredContent: %s",
                    e
                )
                # Fall back to content array
        
        # Extract from content array
        content = None
        if hasattr(result, 'content'):
            content = result.content
        elif isinstance(result, dict):
            content = result.get("content", [])
        
        if content is None:
            return json.dumps({"result": None})
        
        # Concatenate text content items
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    # Other content types (image, resource, etc.)
                    text_parts.append(json.dumps(item))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        
        result_text = "\n".join(text_parts)
        
        # Add error indicator if isError
        if is_error:
            # Return error content for LLM to handle
            return result_text
        
        return result_text