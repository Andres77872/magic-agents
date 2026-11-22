"""Tests for MCP stdio session management."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from magic_agents.mcp.session import MCPSessionManager, MCP_PROTOCOL_VERSION
from magic_agents.mcp.errors import MCPTransportError
from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig


@pytest.fixture
def stdio_config():
    """Create stdio server config."""
    return MCPServerConfig(
        transport="stdio",
        command="uvx",
        args=["mcp-server-filesystem", "/tmp"],
        init_timeout=10.0,
        tool_timeout=30.0,
        discovery_timeout=30.0
    )


class TestSessionConnect:
    
    @pytest.mark.asyncio
    async def test_connect_stdio_spawns_process(self, stdio_config):
        """Test stdio connect spawns subprocess."""
        # Mock the stdio_client and ClientSession
        mock_read_stream = MagicMock()
        mock_write_stream = MagicMock()
        
        with patch('magic_agents.mcp.session.stdio_client') as mock_stdio_client, \
             patch('magic_agents.mcp.session.ClientSession') as mock_client_session:
            
            # Setup mocks
            mock_transport_ctx = MagicMock()
            mock_transport_ctx.__aenter__ = AsyncMock(return_value=(mock_read_stream, mock_write_stream))
            mock_transport_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_stdio_client.return_value = mock_transport_ctx
            
            # Create session mock - ClientSession(...) returns session object
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            
            mock_init_result = MagicMock()
            mock_init_result.serverInfo = {"name": "filesystem-server", "version": "1.0"}
            mock_init_result.capabilities = {"tools": {}}
            mock_init_result.protocolVersion = MCP_PROTOCOL_VERSION
            mock_session.initialize = AsyncMock(return_value=mock_init_result)
            
            mock_client_session.return_value = mock_session
            
            session = MCPSessionManager(config=stdio_config, node_id="test-node")
            await session.connect()
            
            # Verify process spawn was called
            mock_stdio_client.assert_called_once()
            
            # Verify session is healthy
            assert session.is_healthy
            assert session.server_info["name"] == "filesystem-server"
    
    @pytest.mark.asyncio
    async def test_connect_timeout_raises_transport_error(self, stdio_config):
        """An initialize handshake that times out is a transport error that says so."""
        with patch('magic_agents.mcp.session.stdio_client') as mock_stdio_client, \
             patch('magic_agents.mcp.session.ClientSession') as mock_client_session:
            mock_transport_ctx = MagicMock()
            mock_transport_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            mock_transport_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_stdio_client.return_value = mock_transport_ctx

            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.initialize = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_client_session.return_value = mock_session

            session = MCPSessionManager(config=stdio_config, node_id="test-node")

            with pytest.raises(MCPTransportError, match="timed out"):
                await session.connect()

            assert session.is_healthy is False
            mock_session.initialize.assert_awaited_once()


class TestSessionCallTool:
    
    @pytest.mark.asyncio
    async def test_call_tool_success(self, stdio_config):
        """Test successful tool call."""
        with patch('magic_agents.mcp.session.stdio_client') as mock_stdio_client, \
             patch('magic_agents.mcp.session.ClientSession') as mock_client_session:
            
            mock_transport_ctx = MagicMock()
            mock_transport_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            mock_transport_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_stdio_client.return_value = mock_transport_ctx
            
            mock_result = MagicMock()
            mock_result.isError = False
            mock_result.content = [{"type": "text", "text": "File contents"}]
            
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_init_result = MagicMock()
            mock_init_result.serverInfo = {"name": "test"}
            mock_init_result.capabilities = {}
            mock_init_result.protocolVersion = MCP_PROTOCOL_VERSION
            mock_session.initialize = AsyncMock(return_value=mock_init_result)
            mock_session.call_tool = AsyncMock(return_value=mock_result)
            
            mock_client_session.return_value = mock_session
            
            session = MCPSessionManager(config=stdio_config, node_id="test-node")
            await session.connect()
            
            result = await session.call_tool("read_file", {"path": "/tmp/test.txt"})
            
            assert result.isError == False
            mock_session.call_tool.assert_called_once_with("read_file", {"path": "/tmp/test.txt"})
    
    @pytest.mark.asyncio
    async def test_call_tool_timeout_returns_error_result(self, stdio_config):
        """Test tool call timeout returns isError=true result."""
        with patch('magic_agents.mcp.session.stdio_client') as mock_stdio_client, \
             patch('magic_agents.mcp.session.ClientSession') as mock_client_session:
            
            mock_transport_ctx = MagicMock()
            mock_transport_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            mock_transport_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_stdio_client.return_value = mock_transport_ctx
            
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_init_result = MagicMock()
            mock_init_result.serverInfo = {"name": "test"}
            mock_init_result.capabilities = {}
            mock_init_result.protocolVersion = MCP_PROTOCOL_VERSION
            mock_session.initialize = AsyncMock(return_value=mock_init_result)
            mock_session.call_tool = AsyncMock(side_effect=asyncio.TimeoutError)
            
            mock_client_session.return_value = mock_session
            
            session = MCPSessionManager(config=stdio_config, node_id="test-node")
            await session.connect()
            
            result = await session.call_tool("slow_tool", {})
            
            # Should return error result, not raise
            assert result.model_dump(by_alias=True)['isError'] is True
            # content might be SDK TextContent objects or dicts
            content_text = result.content[0].text if hasattr(result.content[0], 'text') else result.content[0].get("text", "")
            assert "timed out" in content_text.lower()
    
    @pytest.mark.asyncio
    async def test_call_tool_on_unhealthy_session_raises(self, stdio_config):
        """Test calling tool on unhealthy session raises."""
        session = MCPSessionManager(config=stdio_config, node_id="test-node")
        # Don't connect - session remains unhealthy
        
        with pytest.raises(MCPTransportError) as exc_info:
            await session.call_tool("read_file", {})
        
        assert "not healthy" in str(exc_info.value).lower()


class TestSessionCleanup:
    
    @pytest.mark.asyncio
    async def test_cleanup_closes_session(self, stdio_config):
        """Test cleanup closes session properly."""
        with patch('magic_agents.mcp.session.stdio_client') as mock_stdio_client, \
             patch('magic_agents.mcp.session.ClientSession') as mock_client_session:
            
            mock_transport_ctx = MagicMock()
            mock_transport_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
            mock_transport_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_stdio_client.return_value = mock_transport_ctx
            
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_init_result = MagicMock()
            mock_init_result.serverInfo = {"name": "test"}
            mock_init_result.capabilities = {}
            mock_init_result.protocolVersion = MCP_PROTOCOL_VERSION
            mock_session.initialize = AsyncMock(return_value=mock_init_result)
            
            mock_client_session.return_value = mock_session
            
            session = MCPSessionManager(config=stdio_config, node_id="test-node")
            await session.connect()
            await session.cleanup()
            
            # Verify session context was exited
            mock_session.__aexit__.assert_called_once()
            mock_transport_ctx.__aexit__.assert_called_once()
            
            assert not session.is_healthy
    
    @pytest.mark.asyncio
    async def test_cleanup_after_error(self, stdio_config):
        """Cleanup is safe after a failed connect and leaves no session behind."""
        mock_transport_ctx = MagicMock()
        mock_transport_ctx.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
        mock_transport_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch('magic_agents.mcp.session.stdio_client', return_value=mock_transport_ctx):
            session = MCPSessionManager(config=stdio_config, node_id="test-node")

            with pytest.raises(MCPTransportError, match="Connection failed"):
                await session.connect()

            await session.cleanup()

            assert session._session is None
            assert session._transport_context is None


class TestServerKey:
    
    def test_server_key_stdio(self, stdio_config):
        """Test server_key format for stdio transport."""
        session = MCPSessionManager(config=stdio_config, node_id="test")
        assert session.server_key == "stdio:uvx"
    
