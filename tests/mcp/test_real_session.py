"""Run actual installed MCP SDK client/server through handshake, discovery and call."""
import sys

import pytest
from magic_agents.mcp.session import MCPSessionManager
from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig

SERVER = '''try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP
server = FastMCP("fixture", instructions="Use the echo tool.")
@server.tool()
def echo(message: str) -> str:
    return "MCP_ECHO:" + message
server.run(transport="stdio")
'''


@pytest.mark.asyncio
async def test_real_stdio_sdk_connects_discovers_calls_and_cleans_up():
    manager = MCPSessionManager(MCPServerConfig(command=sys.executable,args=['-c',SERVER]),node_id='mcp',debug=True)
    try:
        await manager.connect()
        assert manager.is_healthy
        assert manager.server_info['name'] == 'fixture'
        assert 'tools' in manager.capabilities
        assert manager.server_instructions == 'Use the echo tool.'
        discovered = await manager.list_tools()
        assert [tool.name for tool in discovered.tools] == ['echo']
        result = await manager.call_tool('echo',{'message':'hello'})
        assert result.content[0].text == 'MCP_ECHO:hello'
        assert not result.model_dump(by_alias=True).get('isError')
        from magic_agents.mcp.discovery import MCPToolDiscovery
        from magic_agents.mcp.dispatcher import MCPToolDispatcher
        tools = await MCPToolDiscovery(manager).list_tools()
        assert tools[0]['inputSchema']['properties']['message']['type'] == 'string'
        dispatcher = MCPToolDispatcher(manager, namespace=None, timeout=10)
        assert 'MCP_ECHO:hello' in dispatcher._serialize_result(result, 'echo')
    finally:
        await manager.cleanup()
    assert not manager.is_healthy


@pytest.mark.asyncio
async def test_modern_pagination_uses_params_and_protocol_aliases():
    from types import SimpleNamespace
    from mcp.types import ListToolsResult
    from magic_agents.mcp.discovery import MCPToolDiscovery
    cursors = []
    async def list_tools(*, params=None):
        cursor = params.cursor if params is not None else None
        cursors.append(cursor)
        name = 'first' if cursor is None else 'second'
        return ListToolsResult.model_validate({'tools':[{'name':name,'inputSchema':{'type':'object'}}],
                                              'nextCursor':'page-2' if cursor is None else None})
    manager = MCPSessionManager(MCPServerConfig(command='unused'),node_id='mcp')
    manager._session = SimpleNamespace(list_tools=list_tools)
    manager._is_healthy = manager._is_initialized = True
    tools = await MCPToolDiscovery(manager).list_tools()
    assert [tool['name'] for tool in tools] == ['first','second']
    assert cursors == [None,'page-2']
