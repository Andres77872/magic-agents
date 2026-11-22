"""Tests for multi-MCP nodes feeding into single LLM."""
import pytest

from magic_agents.mcp.bundle import MCPToolBundle
from magic_agents.mcp.errors import ToolNameCollisionError
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.node_system.NodeFetch import FetchToolCallable
from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel


def create_bundle(node_id, prefix, tool_names):
    """Create a deterministic bundle using the production MCP contract."""
    schemas = []
    functions = {}

    def create_wrapper(tool_name):
        async def wrapper(**kwargs):
            return f"result from {tool_name}"

        wrapper.__name__ = f"{prefix}_{tool_name}"
        wrapper._mcp_node_id = node_id
        return wrapper

    for tool_name in tool_names:
        schemas.append({
            "type": "function",
            "function": {
                "name": f"{prefix}_{tool_name}",
                "description": f"Tool {tool_name}",
                "parameters": {"type": "object"}
            }
        })

        functions[f"{prefix}_{tool_name}"] = create_wrapper(tool_name)

    return MCPToolBundle(
        tool_schemas=schemas,
        tool_functions=functions,
        node_id=node_id,
        prefix=prefix,
        server_key=f"mock:{node_id}"
    )


def create_llm_node_with_inputs(inputs):
    """Create NodeLLM with specified inputs."""
    data = LlmNodeModel(stream=False, json_output=False)
    node = NodeLLM(data=data, node_id="test-llm", debug=False)
    node.inputs = inputs
    return node


class TestMultiMCPNodesToLLM:

    @pytest.mark.asyncio
    async def test_two_mcp_nodes_flatten_correctly(self):
        """Test two MCP nodes feeding one LLM."""
        # filesystem MCP node
        fs_bundle = create_bundle(
            node_id="mcp-filesystem",
            prefix="fs",
            tool_names=["read_file", "write_file", "list_dir"]
        )

        # database MCP node
        db_bundle = create_bundle(
            node_id="mcp-database",
            prefix="db",
            tool_names=["query", "insert", "update"]
        )

        llm = create_llm_node_with_inputs({
            "handle-tool-definition-0": fs_bundle,
            "handle-tool-definition-1": db_bundle
        })

        schemas, functions, mcp_instructions = await llm._collect_tools()

        assert len(schemas) == 6
        assert len(functions) == 6
        assert mcp_instructions == {}

        # Verify all tool names are present
        expected_tools = [
            "fs_read_file", "fs_write_file", "fs_list_dir",
            "db_query", "db_insert", "db_update"
        ]
        for tool in expected_tools:
            assert tool in functions

        assert await functions["fs_read_file"]() == "result from read_file"
        assert await functions["db_update"]() == "result from update"

    @pytest.mark.asyncio
    async def test_collision_different_prefixes_no_error(self):
        """Test different prefixes prevent collision."""
        # Both have "read" tool but different prefixes
        bundle1 = create_bundle(node_id="mcp-fs", prefix="fs", tool_names=["read"])
        bundle2 = create_bundle(node_id="mcp-db", prefix="db", tool_names=["read"])

        llm = create_llm_node_with_inputs({
            "handle-tool-definition-0": bundle1,
            "handle-tool-definition-1": bundle2
        })

        schemas, functions, _ = await llm._collect_tools()

        # Both should exist without collision
        assert "fs_read" in functions
        assert "db_read" in functions
        assert len(schemas) == 2

    @pytest.mark.asyncio
    async def test_collision_same_prefix_raises(self):
        """Test same prefix collision raises error."""
        # Same prefix = collision
        bundle1 = create_bundle(node_id="mcp-fs-1", prefix="fs", tool_names=["read"])
        bundle2 = create_bundle(node_id="mcp-fs-2", prefix="fs", tool_names=["read"])

        llm = create_llm_node_with_inputs({
            "handle-tool-definition-0": bundle1,
            "handle-tool-definition-1": bundle2
        })

        with pytest.raises(ToolNameCollisionError) as exc_info:
            await llm._collect_tools()

        assert "fs_read" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_mixed_mcp_and_fetch_tools(self):
        """Test MCP bundle + FetchToolCallable together."""
        mcp_bundle = create_bundle(
            node_id="mcp-api",
            prefix="api",
            tool_names=["get_user", "update_user"]
        )

        fetch_tool = FetchToolCallable(
            url_template="https://example.invalid/{{resource}}",
            tool_name="http_fetch",
        )

        llm = create_llm_node_with_inputs({
            "handle-tool-definition-0": mcp_bundle,
            "handle-tool-definition-1": fetch_tool
        })

        schemas, functions, _ = await llm._collect_tools()

        assert len(schemas) == 3
        assert "api_get_user" in functions
        assert "api_update_user" in functions
        assert "http_fetch" in functions

    @pytest.mark.asyncio
    async def test_empty_bundle_doesnt_break_collection(self):
        """Test empty MCP bundle doesn't break collection."""
        empty_bundle = MCPToolBundle(
            tool_schemas=[],
            tool_functions={},
            node_id="mcp-empty",
            prefix="empty"
        )

        other_bundle = create_bundle(
            node_id="mcp-real",
            prefix="real",
            tool_names=["tool1"]
        )

        llm = create_llm_node_with_inputs({
            "handle-tool-definition-0": empty_bundle,
            "handle-tool-definition-1": other_bundle
        })

        schemas, functions, _ = await llm._collect_tools()

        # Only the non-empty bundle should contribute
        assert len(schemas) == 1
        assert "real_tool1" in functions


class TestNodeMcpNodeRegistration:

    def test_mcp_in_tool_capable_types(self):
        """Test 'mcp' is in _TOOL_CAPABLE_TYPES."""
        from magic_agents.agt_flow import _TOOL_CAPABLE_TYPES
        from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel

        assert ModelAgentFlowTypesModel.MCP in _TOOL_CAPABLE_TYPES
        assert 'mcp' in _TOOL_CAPABLE_TYPES

