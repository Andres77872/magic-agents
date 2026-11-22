"""Tests for MCP HTTP session management."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from magic_agents.mcp.session import MCPSessionManager, MCP_PROTOCOL_VERSION
from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig


@pytest.fixture
def http_config():
    """Create HTTP server config."""
    return MCPServerConfig(
        transport="http",
        url="http://localhost:8080/mcp",
        headers={"Authorization": "Bearer test-token"},
        init_timeout=10.0,
        tool_timeout=30.0,
        discovery_timeout=30.0
    )


class TestHTTPSessionConnect:
    
    @pytest.mark.asyncio
    async def test_connect_http_endpoint(self, http_config):
        """Test HTTP connect to endpoint."""
        mock_read_stream = MagicMock()
        mock_write_stream = MagicMock()
        mock_session_id_callback = MagicMock()
        
        with patch('magic_agents.mcp.session.streamablehttp_client') as mock_http_client, \
             patch('magic_agents.mcp.session.ClientSession') as mock_client_session:
            
            # Setup mocks
            mock_transport_ctx = MagicMock()
            mock_transport_ctx.__aenter__ = AsyncMock(
                return_value=(mock_read_stream, mock_write_stream, mock_session_id_callback)
            )
            mock_transport_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_http_client.return_value = mock_transport_ctx
            
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_init_result = MagicMock()
            mock_init_result.serverInfo = {"name": "http-server", "version": "1.0"}
            mock_init_result.capabilities = {"tools": {}}
            mock_init_result.protocolVersion = MCP_PROTOCOL_VERSION
            mock_session.initialize = AsyncMock(return_value=mock_init_result)
            
            mock_client_session.return_value = mock_session
            
            session = MCPSessionManager(config=http_config, node_id="test-node")
            await session.connect()
            
            # Verify HTTP client was called
            mock_http_client.assert_called_once_with(
                http_config.url,
                headers=http_config.headers
            )
            
            # Verify session is healthy
            assert session.is_healthy
    
class TestHTTPSessionIdProperty:
    """Tests for session_id property (recommendation B: SDK owns capture+reuse).
    
    The session_id property provides observability by querying the SDK callback.
    This proves the ownership contract: SDK owns Mcp-Session-Id capture+reuse,
    magic-agents documents and validates the seam via the property.
    """
    
    @pytest.mark.asyncio
    async def test_session_id_property_none_without_callback(self, http_config):
        """Test session_id property returns None when no callback available."""
        with patch('magic_agents.mcp.session.streamablehttp_client') as mock_http_client, \
             patch('magic_agents.mcp.session.ClientSession') as mock_client_session:
            
            # Simulate no callback (only 2-tuple returned)
            mock_transport_ctx = MagicMock()
            mock_transport_ctx.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock())  # No callback
            )
            mock_transport_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_http_client.return_value = mock_transport_ctx
            
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_init_result = MagicMock()
            mock_init_result.serverInfo = {"name": "test"}
            mock_init_result.capabilities = {}
            mock_init_result.protocolVersion = MCP_PROTOCOL_VERSION
            mock_session.initialize = AsyncMock(return_value=mock_init_result)
            
            mock_client_session.return_value = mock_session
            
            manager = MCPSessionManager(config=http_config, node_id="test-node")
            await manager.connect()
            
            # Property returns None when callback unavailable
            assert manager.session_id is None
    
    @pytest.mark.asyncio
    async def test_session_id_property_none_for_stdio_transport(self):
        """Test session_id property returns None for stdio transport."""
        from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig
        
        stdio_config = MCPServerConfig(
            transport="stdio",
            command="python",
            args=["-m", "mcp_server"],
            init_timeout=10.0
        )
        
        with patch('magic_agents.mcp.session.stdio_client') as mock_stdio_client, \
             patch('magic_agents.mcp.session.ClientSession') as mock_client_session:
            
            mock_transport_ctx = MagicMock()
            mock_transport_ctx.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock())
            )
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
            
            manager = MCPSessionManager(config=stdio_config, node_id="test-node")
            await manager.connect()
            
            # Session ID is HTTP-only feature; stdio returns None
            assert manager.session_id is None


