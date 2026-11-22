"""NodeMcp end to end against a real MCP 2.x server over the SDK's in-memory transport.

Only the stdio subprocess is replaced: ``MCPSessionManager``, the SDK
``ClientSession`` handshake, discovery, namespacing, the dispatcher and the
``CallToolResult`` serialization all run for real.
"""

import json

import pytest

# InMemoryTransport is what ``mcp.client.Client(server)`` uses internally; the
# SDK is pinned exactly in setup.cfg, so this private import is stable.
from mcp.client._memory import InMemoryTransport
from mcp.server.mcpserver import MCPServer

from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig, McpNodeModel
from magic_agents.node_system.NodeMcp import NodeMcp


def _server() -> MCPServer:
    server = MCPServer("local-server", instructions="Use arithmetic tools exactly.")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool()
    def secret() -> str:
        return "should be filtered"

    return server


@pytest.fixture
def in_memory_stdio(monkeypatch):
    server = _server()
    opened = []

    def fake_stdio_client(params):
        opened.append(params.command)
        return InMemoryTransport(server)

    monkeypatch.setattr("magic_agents.mcp.session.stdio_client", fake_stdio_client)
    return opened


async def test_node_mcp_discovers_filters_and_calls_real_server_tools(in_memory_stdio):
    config = McpNodeModel(servers=[MCPServerConfig(
        command="in-memory", prefix="math", tool_denylist=["secret"],
        discovery_timeout=5.0, tool_timeout=5.0,
    )])
    node = NodeMcp(config, node_id="mcp-math")

    events = [event async for event in node.process([])]
    bundles = [event["content"]["content"] for event in events if event["type"] == node.OUTPUT_HANDLE]

    assert len(bundles) == 1
    bundle = bundles[0]
    assert [schema["function"]["name"] for schema in bundle.tool_schemas] == ["math_add"]
    assert bundle.tool_schemas[0]["function"]["parameters"]["required"] == ["a", "b"]
    assert bundle.server_instructions == "Use arithmetic tools exactly."

    result = await bundle.tool_functions["math_add"](a=2, b=3)
    assert json.loads(result) == {"result": 5}
    # Discovery and the tool call each opened their own transport session.
    assert in_memory_stdio == ["in-memory", "in-memory"]
