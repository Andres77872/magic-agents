"""MCP server identifiers never carry URL credentials into debug output or prompts."""
import pytest

from magic_agents.mcp import MCPSessionManager
from magic_agents.mcp.session import display_url
from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig


@pytest.mark.parametrize('url,expected', [
    ('https://user:pw@mcp.example.com:8443/v1/mcp?token=SECRET#frag', 'https://mcp.example.com:8443/v1/mcp'),
    ('https://mcp.example.com/mcp', 'https://mcp.example.com/mcp'),
    (None, ''),
])
def test_display_url_drops_credentials(url, expected):
    assert display_url(url) == expected


def test_http_server_key_has_no_query_token():
    config = MCPServerConfig(transport='http', url='https://mcp.example.com/mcp?api_key=SECRET')
    manager = MCPSessionManager(config, node_id='mcp')
    assert manager.server_key == 'http:https://mcp.example.com/mcp'
    assert 'SECRET' not in manager.server_key
