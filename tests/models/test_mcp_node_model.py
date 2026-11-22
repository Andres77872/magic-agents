"""Tests for McpNodeModel and MCPServerConfig Pydantic validation."""
import pytest
from pydantic import ValidationError
from magic_agents.models.factory.Nodes.McpNodeModel import McpNodeModel, MCPServerConfig


class TestMCPServerConfigValidation:
    
    def test_stdio_requires_command(self):
        """Test stdio transport requires command field."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerConfig(transport="stdio")
        
        assert "command" in str(exc_info.value)
    
    def test_http_requires_url(self):
        """Test http transport requires url field."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerConfig(transport="http")
        
        assert "url" in str(exc_info.value)
    
    def test_stdio_valid_with_command(self):
        """Test stdio transport is valid with command."""
        config = MCPServerConfig(
            transport="stdio",
            command="uvx",
            args=["mcp-server-filesystem", "/tmp"]
        )
        assert config.transport == "stdio"
        assert config.command == "uvx"
    
    def test_http_valid_with_url(self):
        """Test http transport is valid with valid URL."""
        config = MCPServerConfig(
            transport="http",
            url="https://example.com/mcp"
        )
        assert config.transport == "http"
        assert config.url == "https://example.com/mcp"
    
    def test_http_rejects_invalid_url(self):
        """Test http transport rejects invalid URL format (regression for blocker #3)."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerConfig(
                transport="http",
                url="not-a-url"
            )
        
        # Should contain error about URL format
        error_str = str(exc_info.value)
        assert "url" in error_str.lower()
        assert "valid HTTP or HTTPS URL" in error_str
    
    def test_http_rejects_url_without_protocol(self):
        """Test http transport rejects URL without http/https protocol."""
        with pytest.raises(ValidationError) as exc_info:
            MCPServerConfig(
                transport="http",
                url="example.com/mcp"
            )
        
        error_str = str(exc_info.value)
        assert "url" in error_str.lower()
    
    def test_http_accepts_localhost_url(self):
        """Test http transport accepts localhost URL."""
        config = MCPServerConfig(
            transport="http",
            url="http://localhost:8080/mcp"
        )
        assert config.url == "http://localhost:8080/mcp"
    
    def test_http_accepts_ip_url(self):
        """Test http transport accepts IP address URL."""
        config = MCPServerConfig(
            transport="http",
            url="http://127.0.0.1:9000/mcp"
        )
        assert config.url == "http://127.0.0.1:9000/mcp"
    
    def test_timeout_bounds(self):
        """Test timeout values must be within bounds."""
        # init_timeout too small
        with pytest.raises(ValidationError):
            MCPServerConfig(transport="stdio", command="test", init_timeout=0.5)
        
        # init_timeout too large
        with pytest.raises(ValidationError):
            MCPServerConfig(transport="stdio", command="test", init_timeout=200.0)


class TestMcpNodeModelValidation:
    
    def test_rejects_empty_servers_list(self):
        """Test McpNodeModel rejects empty servers list (regression for blocker #2)."""
        with pytest.raises(ValidationError) as exc_info:
            McpNodeModel(servers=[])
        
        error_str = str(exc_info.value)
        assert "servers" in error_str.lower()
        assert "at least one" in error_str.lower() or "min_length" in error_str.lower()
    
    def test_rejects_missing_servers(self):
        """Test McpNodeModel requires servers field."""
        # servers is required (no default_factory when Field(...) is used)
        with pytest.raises(ValidationError) as exc_info:
            McpNodeModel()
        
        error_str = str(exc_info.value)
        assert "servers" in error_str.lower()
    
    def test_accepts_single_valid_server(self):
        """Test McpNodeModel accepts single valid server."""
        model = McpNodeModel(
            servers=[
                MCPServerConfig(
                    transport="stdio",
                    command="uvx",
                    args=["mcp-server-filesystem"]
                )
            ]
        )
        assert len(model.servers) == 1
        assert model.servers[0].command == "uvx"
    
    def test_accepts_multiple_servers(self):
        """Test McpNodeModel accepts multiple servers (schema allows, runtime enforces v1)."""
        model = McpNodeModel(
            servers=[
                MCPServerConfig(transport="stdio", command="server1"),
                MCPServerConfig(transport="stdio", command="server2"),
            ]
        )
        # Schema allows multiple (min_length=1), runtime enforces v1 constraint
        assert len(model.servers) == 2


class TestURLValidationEdgeCases:
    
    def test_valid_urls_accepted(self):
        """Test various valid URL formats."""
        valid_urls = [
            "https://api.example.com/mcp",
            "http://localhost:3000",
            "http://127.0.0.1:8080/mcp/v1",
            "https://subdomain.example.co.uk/path",
            "http://192.168.1.1:9000",
        ]
        
        for url in valid_urls:
            config = MCPServerConfig(transport="http", url=url)
            assert config.url == url
    
    def test_invalid_urls_rejected(self):
        """Test various invalid URL formats."""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Wrong protocol
            "example.com/mcp",  # No protocol
            "http://",  # Empty host
            "://broken",
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValidationError):
                MCPServerConfig(transport="http", url=url)