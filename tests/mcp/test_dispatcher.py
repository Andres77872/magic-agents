"""Tests for MCPToolDispatcher behavior."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from magic_agents.mcp.dispatcher import MCPToolDispatcher
from magic_agents.mcp.namespace import MCPToolNamespace, MappedTool
from magic_agents.mcp.session import MCPSessionManager


@pytest.fixture
def mock_session():
    """Create mock MCP session."""
    session = MagicMock(spec=MCPSessionManager)
    session.server_key = "stdio:test-server"
    session.is_healthy = True
    session._config = MagicMock()
    session._config.tool_timeout = 30.0
    session._node_id = "test-node"
    session.call_tool = AsyncMock()
    return session


@pytest.fixture
def namespace():
    """Create namespace with prefix."""
    return MCPToolNamespace(prefix="test")


@pytest.fixture
def mapped_tools():
    """Create sample mapped tools."""
    return [
        MappedTool(
            local_name="test_read",
            remote_name="read",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}}
        ),
        MappedTool(
            local_name="test_write",
            remote_name="write",
            description="Write a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}
        ),
    ]


class TestBuildBundle:
    
    def test_build_bundle_creates_functions(self, mock_session, namespace, mapped_tools):
        """Test build_bundle creates callable wrappers."""
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        bundle = dispatcher.build_bundle(mapped_tools, "test-node")
        
        assert len(bundle.tool_functions) == 2
        assert "test_read" in bundle.tool_functions
        assert "test_write" in bundle.tool_functions
        assert callable(bundle.tool_functions["test_read"])
    
    def test_wrapper_name_set(self, mock_session, namespace, mapped_tools):
        """Test wrapper __name__ is set correctly."""
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        bundle = dispatcher.build_bundle(mapped_tools, "test-node")
        
        assert bundle.tool_functions["test_read"].__name__ == "test_read"
    
    def test_bundle_metadata(self, mock_session, namespace, mapped_tools):
        """Test bundle metadata is populated."""
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        bundle = dispatcher.build_bundle(mapped_tools, "test-node")
        
        assert bundle.node_id == "test-node"
        assert bundle.prefix == "test"
        assert bundle.server_key == "stdio:test-server"


class TestSerializeResult:
    
    def test_serialize_text_content(self, mock_session, namespace):
        """Test serializing simple text content."""
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        
        result = MagicMock()
        result.isError = False
        result.content = [{"type": "text", "text": "Hello world"}]
        result.structuredContent = None
        
        serialized = dispatcher._serialize_result(result, "test_tool")
        assert serialized == "Hello world"
    
    def test_serialize_multiple_text_items(self, mock_session, namespace):
        """Test concatenating multiple text content items."""
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        
        result = MagicMock()
        result.isError = False
        result.content = [
            {"type": "text", "text": "Line 1"},
            {"type": "text", "text": "Line 2"},
        ]
        result.structuredContent = None
        
        serialized = dispatcher._serialize_result(result, "test_tool")
        assert serialized == "Line 1\nLine 2"


class TestBuildSchema:
    
    def test_schema_format(self, mock_session, namespace, mapped_tools):
        """Test schema follows OpenAI format."""
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        schema = dispatcher._build_schema(mapped_tools[0])
        
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "test_read"
        assert schema["function"]["description"] == "Read a file"
        assert schema["function"]["parameters"]["type"] == "object"
    
    def test_schema_with_missing_description(self, mock_session, namespace):
        """Test schema generates description when missing."""
        tool = MappedTool(
            local_name="test_tool",
            remote_name="tool",
            description="",  # Empty description
            input_schema={"type": "object"}
        )
        
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        schema = dispatcher._build_schema(tool)
        
        # Should have fallback description
        assert "MCP tool" in schema["function"]["description"]


@pytest.mark.asyncio
class TestLocalArgumentValidation:
    """Tests for local argument validation before remote call (regression for blocker #4)."""
    
    async def test_missing_required_args_returns_error_not_calls_remote(self, mock_session, namespace):
        """Test missing required arguments returns error without calling remote."""
        # Tool with required argument
        tool_with_required = MappedTool(
            local_name="test_read_required",
            remote_name="read",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        )
        
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        bundle = dispatcher.build_bundle([tool_with_required], "test-node")
        
        # Call without required argument
        output = await bundle.tool_functions["test_read_required"]()
        
        # Should NOT have called session.call_tool
        mock_session.call_tool.assert_not_called()
        
        # Should return error JSON
        result_data = json.loads(output)
        assert result_data["error"] == "validation_failed"
        assert "path" in result_data["message"]
    
    async def test_missing_required_args_with_other_args_present(self, mock_session, namespace):
        """Test missing required argument when other args are present."""
        tool = MappedTool(
            local_name="test_multi_required",
            remote_name="multi",
            description="Multi arg tool",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "optional": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        )
        
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        bundle = dispatcher.build_bundle([tool], "test-node")
        
        # Provide only one required arg
        output = await bundle.tool_functions["test_multi_required"](path="/test")
        
        # Should NOT have called remote
        mock_session.call_tool.assert_not_called()
        
        # Error should mention missing content
        result_data = json.loads(output)
        assert "content" in result_data["message"]
    
    async def test_all_required_args_present_calls_remote(self, mock_session, namespace):
        """Test valid arguments proceed to remote call."""
        tool = MappedTool(
            local_name="test_valid_call",
            remote_name="read",
            description="Read tool",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        )
        
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        bundle = dispatcher.build_bundle([tool], "test-node")
        
        # Mock successful result
        result = MagicMock()
        result.isError = False
        result.content = [{"type": "text", "text": "Success"}]
        result.structuredContent = None
        mock_session.call_tool.return_value = result
        
        # Call with required arg
        await bundle.tool_functions["test_valid_call"](path="/test")
        
        # Should have called remote
        mock_session.call_tool.assert_called_once()
    
    async def test_optional_args_not_required(self, mock_session, namespace):
        """Test optional arguments don't block call."""
        tool = MappedTool(
            local_name="test_optional",
            remote_name="optional",
            description="Tool with optional args",
            input_schema={
                "type": "object",
                "properties": {
                    "required": {"type": "string"},
                    "optional": {"type": "string"}
                },
                "required": ["required"]
            }
        )
        
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        bundle = dispatcher.build_bundle([tool], "test-node")
        
        result = MagicMock()
        result.isError = False
        result.content = [{"type": "text", "text": "OK"}]
        result.structuredContent = None
        mock_session.call_tool.return_value = result
        
        # Call with only required arg (optional omitted)
        await bundle.tool_functions["test_optional"](required="value")
        
        # Should call remote
        mock_session.call_tool.assert_called_once()
    
    async def test_wrong_type_argument_returns_error(self, mock_session, namespace):
        """Test wrong type argument returns validation error."""
        tool = MappedTool(
            local_name="test_type_check",
            remote_name="typed",
            description="Typed tool",
            input_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"]
            }
        )
        
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        bundle = dispatcher.build_bundle([tool], "test-node")
        
        # Call with wrong type (string instead of integer)
        output = await bundle.tool_functions["test_type_check"](count="not-an-int")
        
        # Should NOT call remote
        mock_session.call_tool.assert_not_called()
        
        # Error should mention type mismatch
        result_data = json.loads(output)
        assert result_data["error"] == "validation_failed"
        assert "type" in result_data["message"].lower()
    
    async def test_no_schema_allows_any_args(self, mock_session, namespace):
        """Test tool with no schema accepts any arguments."""
        tool_no_schema = MappedTool(
            local_name="test_no_schema",
            remote_name="free",
            description="No schema tool",
            input_schema={}  # Empty schema
        )
        
        dispatcher = MCPToolDispatcher(session=mock_session, namespace=namespace)
        bundle = dispatcher.build_bundle([tool_no_schema], "test-node")
        
        result = MagicMock()
        result.isError = False
        result.content = [{"type": "text", "text": "OK"}]
        result.structuredContent = None
        mock_session.call_tool.return_value = result
        
        # Call with arbitrary args
        await bundle.tool_functions["test_no_schema"](anything="value", another=123)
        
        # Should call remote (no validation blocking)
        mock_session.call_tool.assert_called_once()