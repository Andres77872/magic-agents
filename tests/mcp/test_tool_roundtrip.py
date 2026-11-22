"""Tests for MCP tool call roundtrip."""
import json
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock
from magic_agents.mcp.session import MCPSessionManager
from magic_agents.mcp.discovery import MCPToolDiscovery
from magic_agents.mcp.namespace import MCPToolNamespace
from magic_agents.mcp.dispatcher import MCPToolDispatcher
from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig


@pytest.fixture
def mock_session_with_tools():
    """Create mock session with tool discovery."""
    config = MCPServerConfig(
        transport="stdio",
        command="test-server",
        init_timeout=10.0,
        tool_timeout=30.0,
        discovery_timeout=30.0
    )
    
    mock_session = MagicMock(spec=MCPSessionManager)
    mock_session._config = config
    mock_session._node_id = "test-node"
    mock_session.server_key = "stdio:test-server"
    mock_session.is_healthy = True
    
    # Mock tools/list result - properly set attribute values
    mock_tool1 = MagicMock()
    mock_tool1.name = "get_weather"  # Set actual attribute value
    mock_tool1.description = "Get weather for location"
    mock_tool1.inputSchema = {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    }
    
    mock_tool2 = MagicMock()
    mock_tool2.name = "get_time"
    mock_tool2.description = "Get current time"
    mock_tool2.inputSchema = {"type": "object", "properties": {}}
    
    mock_list_result = MagicMock()
    mock_list_result.tools = [mock_tool1, mock_tool2]
    mock_list_result.nextCursor = None
    mock_session.list_tools = AsyncMock(return_value=mock_list_result)
    
    return mock_session


class TestToolRoundtrip:
    
    @pytest.mark.asyncio
    async def test_tool_call_returns_content(self, mock_session_with_tools):
        """Test wrapper callable returns content."""
        # Mock tool call result
        mock_call_result = MagicMock()
        mock_call_result.isError = False
        mock_call_result.content = [{"type": "text", "text": "22°C, sunny"}]
        mock_call_result.structuredContent = None
        mock_session_with_tools.call_tool = AsyncMock(return_value=mock_call_result)
        
        # Build dispatcher
        namespace = MCPToolNamespace(prefix="weather")
        discovery = MCPToolDiscovery(session=mock_session_with_tools)
        raw_tools = await discovery.list_tools()
        mapped_tools = namespace.apply_prefix(raw_tools)
        
        dispatcher = MCPToolDispatcher(
            session=mock_session_with_tools,
            namespace=namespace
        )
        bundle = dispatcher.build_bundle(mapped_tools, "test-node")
        
        # Call wrapper
        wrapper = bundle.tool_functions["weather_get_weather"]
        result = await wrapper(location="Buenos Aires")
        
        # Verify result
        assert "22°C" in result
        # Verify session was called with remote name (timeout may also be passed)
        mock_session_with_tools.call_tool.assert_called_once()
        call_args = mock_session_with_tools.call_tool.call_args
        assert call_args.kwargs["name"] == "get_weather"  # Remote name
        assert call_args.kwargs["arguments"]["location"] == "Buenos Aires"
    
    @pytest.mark.asyncio
    async def test_tool_error_returns_content_not_raises(self, mock_session_with_tools):
        """Test isError:true returns content (not raises)."""
        # Mock error result
        mock_error_result = MagicMock()
        mock_error_result.isError = True
        mock_error_result.content = [{"type": "text", "text": "Rate limit exceeded"}]
        mock_error_result.structuredContent = None
        mock_session_with_tools.call_tool = AsyncMock(return_value=mock_error_result)
        
        namespace = MCPToolNamespace(prefix="weather")
        discovery = MCPToolDiscovery(session=mock_session_with_tools)
        raw_tools = await discovery.list_tools()
        mapped_tools = namespace.apply_prefix(raw_tools)
        
        dispatcher = MCPToolDispatcher(
            session=mock_session_with_tools,
            namespace=namespace
        )
        bundle = dispatcher.build_bundle(mapped_tools, "test-node")
        
        wrapper = bundle.tool_functions["weather_get_weather"]
        
        # Should NOT raise, should return error content
        result = await wrapper(location="Invalid")
        
        assert "Rate limit exceeded" in result
    
    @pytest.mark.asyncio
    async def test_structured_content_preferred(self, mock_session_with_tools):
        """Test structuredContent is preferred and JSON-serialized."""
        # Mock result with structuredContent
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [{"type": "text", "text": "Fallback text"}]
        mock_result.structuredContent = {
            "temperature": 22,
            "conditions": "sunny",
            "humidity": 45
        }
        mock_session_with_tools.call_tool = AsyncMock(return_value=mock_result)
        
        namespace = MCPToolNamespace(prefix="weather")
        discovery = MCPToolDiscovery(session=mock_session_with_tools)
        raw_tools = await discovery.list_tools()
        mapped_tools = namespace.apply_prefix(raw_tools)
        
        dispatcher = MCPToolDispatcher(
            session=mock_session_with_tools,
            namespace=namespace
        )
        bundle = dispatcher.build_bundle(mapped_tools, "test-node")
        
        wrapper = bundle.tool_functions["weather_get_weather"]
        result = await wrapper(location="Test")
        
        # Should be JSON from structuredContent
        data = json.loads(result)
        assert data["temperature"] == 22
        assert data["conditions"] == "sunny"


class TestPaginatedDiscovery:
    
    @pytest.mark.asyncio
    async def test_pagination_loop(self):
        """Test discovery handles paginated tools/list."""
        config = MCPServerConfig(
            transport="stdio",
            command="test-server",
            init_timeout=10.0,
            tool_timeout=30.0,
            discovery_timeout=30.0
        )
        
        mock_session = MagicMock(spec=MCPSessionManager)
        mock_session._config = config
        mock_session._node_id = "test-node"
        mock_session.server_key = "stdio:test-server"
        mock_session.is_healthy = True
        
        # First page
        mock_page1 = MagicMock()
        # SimpleNamespace, not MagicMock(name=...): MagicMock's name= only sets its repr.
        mock_page1.tools = [SimpleNamespace(name=f"tool_{i}", description=f"Tool {i}", inputSchema={}) for i in range(10)]
        mock_page1.nextCursor = "page2"
        
        # Second page
        mock_page2 = MagicMock()
        mock_page2.tools = [SimpleNamespace(name=f"tool_{i}", description=f"Tool {i}", inputSchema={}) for i in range(10, 20)]
        mock_page2.nextCursor = None
        
        mock_session.list_tools = AsyncMock(side_effect=[mock_page1, mock_page2])
        
        discovery = MCPToolDiscovery(session=mock_session)
        tools = await discovery.list_tools()
        
        assert [tool["name"] if isinstance(tool, dict) else tool.name for tool in tools] == [f"tool_{i}" for i in range(20)]
        assert mock_session.list_tools.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_returns_cached_tools(self, mock_session_with_tools):
        """Test discovery caches tools."""
        discovery = MCPToolDiscovery(session=mock_session_with_tools)
        
        # First call
        tools1 = await discovery.list_tools()
        
        # Second call should use cache
        tools2 = await discovery.list_tools()
        
        # Session should only be called once
        assert mock_session_with_tools.list_tools.call_count == 1
        assert tools1 == tools2
    
    @pytest.mark.asyncio
    async def test_invalidate_cache(self, mock_session_with_tools):
        """Test cache invalidation."""
        discovery = MCPToolDiscovery(session=mock_session_with_tools)
        
        # First call
        await discovery.list_tools()
        
        # Invalidate
        discovery.invalidate_cache()
        
        # Second call should hit session again
        await discovery.list_tools()
        
        assert mock_session_with_tools.list_tools.call_count == 2