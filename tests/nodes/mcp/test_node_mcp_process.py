"""Behavior-level tests for NodeMcp.process()."""

import sys
import pytest

from magic_agents.mcp.bundle import MCPToolBundle
from magic_agents.mcp.errors import MCPProtocolError, MCPTransportError
from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig, McpNodeModel
from magic_agents.node_system.NodeMcp import NodeMcp


RAW_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]


class FakeSession:
    instances: list["FakeSession"] = []
    next_connect_error = None
    next_list_error = None
    next_cleanup_error = None

    def __init__(self, config, node_id, debug):
        self._config = config
        self._node_id = node_id
        self.node_id = node_id
        self.debug = debug
        self.server_key = f"{config.transport}:{config.command or config.url}"
        self.server_instructions = "server instructions"
        self.is_healthy = True
        self.server_info = {"name": "fake"}
        self.connect_calls = 0
        self.cleanup_calls = 0
        self.tool_calls = []
        self.list_calls = []
        self.list_error = self.next_list_error
        self.connect_error = self.next_connect_error
        self.cleanup_error = self.next_cleanup_error
        FakeSession.instances.append(self)

    async def connect(self):
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error

    async def cleanup(self):
        self.cleanup_calls += 1
        self.is_healthy = False
        if self.cleanup_error is not None:
            raise self.cleanup_error

    async def list_tools(self, cursor=None):
        self.list_calls.append(cursor)
        if self.list_error is not None:
            raise self.list_error
        if cursor is None:
            return {"tools": [RAW_TOOLS[0]], "nextCursor": "page-2"}
        assert cursor == "page-2"
        return {"tools": [RAW_TOOLS[1]], "nextCursor": None}

    async def call_tool(self, name, arguments, timeout=None):
        self.tool_calls.append((name, arguments, timeout))
        return {"content": [{"type": "text", "text": "ok"}]}


@pytest.fixture(autouse=True)
def fake_mcp_collaborators(monkeypatch):
    FakeSession.instances.clear()
    FakeSession.next_connect_error = None
    FakeSession.next_list_error = None
    FakeSession.next_cleanup_error = None
    monkeypatch.setattr(sys.modules["magic_agents.node_system.NodeMcp"], "MCPSessionManager", FakeSession)


def mcp_config(**server_overrides):
    server = {
        "transport": "stdio",
        "command": "test-server",
        "prefix": "fs",
        "init_timeout": 10.0,
        "tool_timeout": 30.0,
        "discovery_timeout": 30.0,
    }
    server.update(server_overrides)
    return McpNodeModel(servers=[MCPServerConfig(**server)])


async def collect_node_outputs(node):
    return [event async for event in node.process([])]


@pytest.mark.asyncio
async def test_process_yields_prefixed_tool_bundle_and_cleans_up():
    node = NodeMcp(data=mcp_config(), node_id="mcp-fs", debug=False)

    results = await collect_node_outputs(node)

    bundle_events = [event for event in results if event["type"] == "handle-tool-definition"]
    assert len(bundle_events) == 1
    bundle = bundle_events[0]["content"]["content"]
    assert isinstance(bundle, MCPToolBundle)
    assert bundle.node_id == "mcp-fs"
    assert bundle.prefix == "fs"
    assert bundle.server_instructions == "server instructions"
    assert [schema["function"]["name"] for schema in bundle.tool_schemas] == [
        "fs_read_file",
        "fs_write_file",
    ]
    assert set(bundle.tool_functions) == {"fs_read_file", "fs_write_file"}

    session = FakeSession.instances[0]
    assert session.connect_calls == 1
    assert session.list_calls == [None, "page-2"]
    assert session.cleanup_calls == 1

    result = await bundle.tool_functions["fs_read_file"](path="/tmp/example.txt")
    assert result == "ok"
    assert len(FakeSession.instances) == 2
    call_session = FakeSession.instances[1]
    assert call_session.connect_calls == 1
    assert call_session.tool_calls == [("read_file", {"path": "/tmp/example.txt"}, 30.0)]
    assert call_session.cleanup_calls == 1


@pytest.mark.asyncio
async def test_process_cleans_up_and_yields_debug_error_on_discovery_failure():
    FakeSession.next_list_error = MCPTransportError(
        message="Discovery failed",
        server="stdio:test-server",
        transport_type="stdio",
    )
    node = NodeMcp(data=mcp_config(), node_id="mcp-error", debug=False)
    results = []
    with pytest.raises(MCPTransportError, match="Discovery failed"):
        async for event in node.process([]):
            results.append(event)

    debug_events = [event for event in results if event["type"] == "debug"]
    assert len(debug_events) == 1
    assert debug_events[0]["content"]["error_type"] == "MCPTransportError"
    assert "Discovery failed" in debug_events[0]["content"]["error_message"]
    assert FakeSession.instances[0].cleanup_calls == 1


@pytest.mark.asyncio
async def test_process_reports_protocol_error_and_cleans_up():
    FakeSession.next_connect_error = MCPProtocolError(code=-32601, message="Method not found")
    node = NodeMcp(data=mcp_config(), node_id="mcp-protocol", debug=False)
    results = []
    with pytest.raises(MCPProtocolError, match="Method not found"):
        async for event in node.process([]):
            results.append(event)

    errors = [event for event in results if event["type"] == "debug"]
    assert len(errors) == 1
    assert errors[0]["content"]["error_type"] == "MCPProtocolError"
    assert errors[0]["content"]["context"]["code"] == -32601
    assert FakeSession.instances[0].cleanup_calls == 1


@pytest.mark.asyncio
async def test_process_sanitizes_unexpected_error_and_survives_cleanup_failure():
    FakeSession.next_connect_error = ValueError("bad session state")
    FakeSession.next_cleanup_error = RuntimeError("cleanup broke")
    node = NodeMcp(data=mcp_config(), node_id="mcp-unexpected", debug=False)
    results = []
    with pytest.raises(ValueError, match="bad session state"):
        async for event in node.process([]):
            results.append(event)

    errors = [event for event in results if event["type"] == "debug"]
    assert len(errors) == 1
    assert errors[0]["content"]["error_type"] == "UnexpectedError"
    assert errors[0]["content"]["context"]["exception_type"] == "ValueError"
    assert FakeSession.instances[0].cleanup_calls == 1


@pytest.mark.asyncio
async def test_capture_internal_state_reports_active_session_bundle_and_timeouts():
    node = NodeMcp(data=mcp_config(), node_id="mcp-state", debug=False)
    events = await collect_node_outputs(node)
    bundle = next(
        event["content"]["content"]
        for event in events
        if event["type"] == node.OUTPUT_HANDLE
    )
    session = FakeSession.instances[0]
    session.is_healthy = True
    node._session = session
    node._bundle = bundle

    state = node._capture_internal_state()

    assert state["server_key"] == "stdio:test-server"
    assert state["session_healthy"] is True
    assert state["server_info"] == {"name": "fake"}
    assert state["bundle_tool_count"] == 2
    assert state["bundle_prefix"] == "fs"
    assert state["transport_type"] == "stdio"
    assert state["timeout_config"] == {"init": 10.0, "tool": 30.0, "discovery": 30.0}


@pytest.mark.asyncio
async def test_process_rejects_multiple_servers_before_connecting():
    config = McpNodeModel(
        servers=[
            MCPServerConfig(transport="stdio", command="server1", prefix="s1"),
            MCPServerConfig(transport="stdio", command="server2", prefix="s2"),
        ]
    )
    node = NodeMcp(data=config, node_id="mcp-multi", debug=False)

    results = []
    with pytest.raises(ValueError, match="exactly 1 server"):
        async for event in node.process([]):
            results.append(event)

    debug_events = [event for event in results if event["type"] == "debug"]
    assert len(debug_events) == 1
    assert debug_events[0]["content"]["error_type"] == "ConfigurationError"
    assert "exactly 1 server" in debug_events[0]["content"]["error_message"]
    assert FakeSession.instances == []
