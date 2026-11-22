"""MCPServerConfig accepts real-world server URLs and ignores the inactive transport's fields."""
import pytest
from pydantic import ValidationError

from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig


@pytest.mark.parametrize('url', [
    'http://mcp-server:8000/mcp',            # docker compose service name
    'http://host.docker.internal:8000/mcp',
    'https://mcp.example.company/mcp',       # TLD longer than six letters
    'http://[::1]:8000/mcp',
    'http://127.0.0.1:8767/mcp',
    'https://api.example.com/mcp?key=abc',
])
def test_accepts_http_urls_with_a_host(url):
    assert MCPServerConfig(transport='http', url=url).url == url


@pytest.mark.parametrize('url', ['ftp://example.com/mcp', 'http://', 'not a url', 'http://exa mple.com', 'http://host:99999/mcp'])
def test_rejects_malformed_urls(url):
    with pytest.raises(ValidationError):
        MCPServerConfig(transport='http', url=url)


def test_stdio_server_ignores_leftover_http_fields():
    server = MCPServerConfig(transport='stdio', command='python', url='', headers={'Authorization': 'x'})
    assert server.url is None and server.headers is None
    assert server.command == 'python'


def test_http_server_ignores_leftover_stdio_fields_but_still_requires_a_url():
    server = MCPServerConfig(transport='http', url='http://mcp:1/mcp', command='python', args=['-c', 'pass'], cwd='')
    assert server.command is None and server.args is None and server.cwd is None
    with pytest.raises(ValidationError):
        MCPServerConfig(transport='http', url='')
