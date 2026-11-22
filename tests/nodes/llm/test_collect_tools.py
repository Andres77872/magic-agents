"""Tests for NodeLLM._collect_tools() bundle flattening."""
import pytest
from unittest.mock import MagicMock, patch
from magic_agents.mcp.bundle import MCPToolBundle
from magic_agents.mcp.errors import MCPToolNameCollisionError, ToolNameCollisionError


class MockToolCallable:
    """Mock tool callable conforming to ToolProvider protocol."""
    
    def __init__(self, name, schema):
        self._name = name
        self._schema = schema
    
    @property
    def __name__(self):
        return self._name
    
    @property
    def tool_schema(self):
        return self._schema
    
    @property
    def tool_callable(self):
        return self


    async def __call__(self, **kwargs):
        return kwargs


class MockFetchToolCallable(MockToolCallable):
    """Mock FetchToolCallable-like object."""
    
    def __init__(self, name="fetch"):
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": "HTTP fetch",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}
            }
        }
        super().__init__(name, schema)


class MockMCPToolBundle(MCPToolBundle):
    """Mock MCPToolBundle for testing."""
    
    def __init__(self, tools, node_id="mcp-node", prefix="mcp"):
        schemas = []
        functions = {}
        
        for tool_name, tool_func in tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Tool {tool_name}",
                    "parameters": {"type": "object"}
                }
            })
            functions[tool_name] = tool_func
        
        super().__init__(
            tool_schemas=schemas,
            tool_functions=functions,
            node_id=node_id,
            prefix=prefix
        )


class MockSubagentBundle:
    """Mock SubagentBundle for testing subagent collision detection."""
    
    def __init__(self, tool_schemas=None, registered_count=0):
        self.tool_schemas = tool_schemas or []
        self.registered_count = registered_count


def create_node_llm_with_inputs(inputs):
    """Create mock NodeLLM with specified inputs."""
    from magic_agents.node_system.NodeLLM import NodeLLM
    from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel
    
    data = LlmNodeModel(stream=False, json_output=False)
    node = NodeLLM(data=data, node_id="test-llm", debug=False)
    node.inputs = inputs
    return node


class TestBundleFlattening:
    
    @pytest.mark.asyncio
    async def test_flatten_single_bundle(self):
        """Test flattening a single MCPToolBundle."""
        async def tool_func(**kwargs):
            return "result"
        tool_func._mcp_node_id = "mcp-node-1"
        
        bundle = MockMCPToolBundle(
            tools={"mcp_read_file": tool_func},
            node_id="mcp-node-1",
            prefix="mcp"
        )
        
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": bundle
        })
        
        schemas, functions, mcp_instructions = await node._collect_tools()
        
        assert len(schemas) == 1
        assert len(functions) == 1
        assert "mcp_read_file" in functions
    
    @pytest.mark.asyncio
    async def test_flatten_multiple_bundles(self):
        """Test flattening multiple MCPToolBundles."""
        async def fs_read(**kwargs):
            return "fs result"
        fs_read._mcp_node_id = "fs-node"
        
        async def db_query(**kwargs):
            return "db result"
        db_query._mcp_node_id = "db-node"
        
        bundle1 = MockMCPToolBundle(
            tools={"fs_read_file": fs_read},
            node_id="fs-node",
            prefix="fs"
        )
        
        bundle2 = MockMCPToolBundle(
            tools={"db_query": db_query},
            node_id="db-node",
            prefix="db"
        )
        
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": bundle1,
            "handle-tool-definition-1": bundle2
        })
        
        schemas, functions, _ = await node._collect_tools()
        
        assert len(schemas) == 2
        assert len(functions) == 2
        assert "fs_read_file" in functions
        assert "db_query" in functions
    
    @pytest.mark.asyncio
    async def test_flatten_bundle_with_many_tools(self):
        """Test flattening bundle with multiple tools."""
        tools = {}
        for i in range(5):
            async def tool_func(**kwargs):
                return f"result{i}"
            tool_func._mcp_node_id = "multi-node"
            tools[f"tool_{i}"] = tool_func
        
        bundle = MockMCPToolBundle(
            tools=tools,
            node_id="multi-node",
            prefix="multi"
        )
        
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": bundle
        })
        
        schemas, functions, _ = await node._collect_tools()
        
        assert len(schemas) == 5
        assert len(functions) == 5
        for i in range(5):
            assert f"tool_{i}" in functions


class TestMixedInputs:
    
    @pytest.mark.asyncio
    async def test_bundle_and_single_tool(self):
        """Test mixed MCP bundle + FetchToolCallable."""
        async def mcp_read(**kwargs):
            return "mcp result"
        mcp_read._mcp_node_id = "mcp-node"
        
        bundle = MockMCPToolBundle(
            tools={"mcp_read": mcp_read},
            node_id="mcp-node"
        )
        
        fetch_tool = MockFetchToolCallable()
        
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": bundle,
            "handle-tool-definition-1": fetch_tool
        })
        
        schemas, functions, _ = await node._collect_tools()
        
        assert len(schemas) == 2
        assert len(functions) == 2


def schema_tool(name: str = "client_lookup") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Client-executed lookup",
            "parameters": {"type": "object", "properties": {}},
        },
    }


class TestSchemaOnlyNodeToolCollection:

    @pytest.mark.asyncio
    async def test_collects_single_schema_only_dict_without_callable(self):
        node = create_node_llm_with_inputs({"handle-tool-definition-0": schema_tool("client_lookup")})

        schemas, functions, _ = await node._collect_tools()

        assert schemas == [schema_tool("client_lookup")]
        assert functions == {}

    @pytest.mark.asyncio
    async def test_collects_multiple_schema_only_dicts_with_distinct_names(self):
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": schema_tool("client_lookup"),
            "handle-tool-definition-1": schema_tool("client_search"),
        })

        schemas, functions, _ = await node._collect_tools()

        assert [s["function"]["name"] for s in schemas] == ["client_lookup", "client_search"]
        assert functions == {}

    @pytest.mark.asyncio
    async def test_duplicate_schema_only_names_are_rejected(self):
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": schema_tool("dup_tool"),
            "handle-tool-definition-1": schema_tool("dup_tool"),
        })

        with pytest.raises(ToolNameCollisionError, match="dup_tool"):
            await node._collect_tools()

    @pytest.mark.asyncio
    async def test_schema_only_and_callable_duplicate_name_is_rejected(self):
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": schema_tool("fetch"),
            "handle-tool-definition-1": MockFetchToolCallable("fetch"),
        })

        with pytest.raises(ToolNameCollisionError, match="fetch"):
            await node._collect_tools()

    @pytest.mark.asyncio
    async def test_mixed_callable_and_schema_only_is_rejected_for_v1(self):
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": schema_tool("client_lookup"),
            "handle-tool-definition-1": MockFetchToolCallable("server_fetch"),
        })

        with pytest.raises(ValueError, match="mixed callable and schema-only tools"):
            await node._collect_tools()
    
    @pytest.mark.asyncio
    async def test_bundle_and_plain_callable(self):
        """Test mixed MCP bundle + plain callable."""
        async def mcp_tool(**kwargs):
            return "mcp result"
        mcp_tool._mcp_node_id = "mcp-node"
        
        bundle = MockMCPToolBundle(
            tools={"mcp_tool": mcp_tool},
            node_id="mcp-node"
        )
        
        async def plain_tool(**kwargs):
            return "plain result"
        plain_tool.__name__ = "plain_tool"
        
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": bundle,
            "handle-tool-definition-1": plain_tool
        })
        
        schemas, functions, _ = await node._collect_tools()
        
        assert len(schemas) == 2
        assert len(functions) == 2


class TestCollisionDetection:
    
    @pytest.mark.asyncio
    async def test_collision_between_bundles(self):
        """Test collision detection across bundles."""
        async def tool1(**kwargs):
            return "result1"
        tool1._mcp_node_id = "node-a"
        
        async def tool2(**kwargs):
            return "result2"
        tool2._mcp_node_id = "node-b"
        
        # Same tool name in both bundles
        bundle1 = MockMCPToolBundle(
            tools={"duplicate_name": tool1},
            node_id="node-a"
        )
        
        bundle2 = MockMCPToolBundle(
            tools={"duplicate_name": tool2},
            node_id="node-b"
        )
        
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": bundle1,
            "handle-tool-definition-1": bundle2
        })
        
        with pytest.raises(ToolNameCollisionError) as exc_info:
            await node._collect_tools()
        
        assert "duplicate_name" in str(exc_info.value)
    
class TestEmptyInputs:
    
    @pytest.mark.asyncio
    async def test_empty_inputs(self):
        """Test empty inputs returns empty collections."""
        node = create_node_llm_with_inputs({})
        
        schemas, functions, mcp_instructions = await node._collect_tools()
        
        assert schemas == []
        assert functions == {}
        assert mcp_instructions == {}
    
    @pytest.mark.asyncio
    async def test_non_tool_inputs_skipped(self):
        """Test non-tool-prefixed inputs are skipped."""
        node = create_node_llm_with_inputs({
            "handle-user-message": "Hello",
            "handle-system-context": "System prompt"
        })
        
        schemas, functions, _ = await node._collect_tools()
        
        assert schemas == []
        assert functions == {}
    
    @pytest.mark.asyncio
    async def test_none_value_skipped(self):
        """Test None values are skipped."""
        bundle = MockMCPToolBundle(
            tools={"tool": lambda: "result"},
            node_id="mcp"
        )
        
        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": bundle,
            "handle-tool-definition-1": None
        })
        
        schemas, functions, _ = await node._collect_tools()
        
        assert len(schemas) == 1
        assert len(functions) == 1


# ═══════════════════════════════════════════════════════════════════════
# Task 4.3: Subagent-to-graph collision detection
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Task 4.4: ToolNameCollisionError rename and backward compatibility
# ═══════════════════════════════════════════════════════════════════════

class TestToolNameCollisionErrorCompat:
    """Tests for ToolNameCollisionError rename and backward compatibility."""

    def test_isinstance_backward_compat(self):
        """isinstance(MCPToolNameCollisionError(...), ToolNameCollisionError) is True."""
        from magic_agents.mcp.errors import ToolNameCollisionError
        error = MCPToolNameCollisionError(tool_name="test", source_nodes=["a", "b"])
        assert isinstance(error, ToolNameCollisionError)

# ═══════════════════════════════════════════════════════════════════════
# Task 4.5: _build_tool_manifest() format and truncation
# ═══════════════════════════════════════════════════════════════════════

class TestToolManifestBuilder:
    """Tests for NodeLLM._build_tool_manifest()."""

    def _make_node(self):
        """Create a bare NodeLLM for manifest testing."""
        from magic_agents.node_system.NodeLLM import NodeLLM
        from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel
        data = LlmNodeModel(stream=False, json_output=False)
        return NodeLLM(data=data, node_id="test-llm", debug=False)

    def _make_schema(self, name, desc="A test tool", params=None):
        """Helper to create a tool schema dict."""
        p = params or {}
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": p,
                    "required": list(p.keys()),
                }
            }
        }

    def test_manifest_includes_available_tools_header(self):
        """Manifest includes '## Available Tools' header."""
        node = self._make_node()
        schema = self._make_schema("test_tool")

        manifest = node._build_tool_manifest(
            tools_schemas=[schema],
            tool_functions={"test_tool": MagicMock()},
        )

        assert "## Available Tools" in manifest

    def test_manifest_includes_tool_name_description_params_source(self):
        """Each tool has name, description, parameters, source."""
        from magic_agents.node_system.NodeFetch import FetchToolCallable

        node = self._make_node()
        schema = self._make_schema("web_fetch", "Fetch a URL", {"url": {"type": "string"}})

        callable_obj = FetchToolCallable(url_template="https://example.com", tool_name="web_fetch")
        callable_obj._source_node_id = "fetch-node-1"

        manifest = node._build_tool_manifest(
            tools_schemas=[schema],
            tool_functions={"web_fetch": callable_obj},
        )

        assert "### Tool: web_fetch" in manifest
        assert "Fetch a URL" in manifest
        assert "`url` (string)" in manifest
        assert "fetch-node-1" in manifest

    def test_manifest_subagent_tools_section(self):
        """Subagent tools appear in separate '## Subagent Tools' section."""
        node = self._make_node()
        subagent = MockSubagentBundle(
            tool_schemas=[self._make_schema("sub_tool", "Subagent helper")],
            registered_count=1,
        )

        manifest = node._build_tool_manifest(
            tools_schemas=[],
            tool_functions={},
            subagent_bundle=subagent,
        )

        assert "## Subagent Tools" in manifest
        assert "**sub_tool**" in manifest
        assert "Subagent helper" in manifest

    def test_manifest_truncation_at_4096_chars(self):
        """Manifest truncated at 4096 chars with warning logged."""
        node = self._make_node()

        # Create enough tools to exceed 4096 chars
        schemas = []
        functions = {}
        for i in range(100):
            name = f"very_long_tool_name_{i:04d}"
            schemas.append(self._make_schema(name, "A" * 80))
            functions[name] = MagicMock()

        with patch('magic_agents.node_system.NodeLLM.logger') as mock_logger:
            manifest = node._build_tool_manifest(
                tools_schemas=schemas,
                tool_functions=functions,
            )

            assert len(manifest) <= 4096
            mock_logger.warning.assert_called_once()
            assert "truncated" in mock_logger.warning.call_args[0][0]

    def test_manifest_empty_tools(self):
        """Empty tools list produces empty manifest."""
        node = self._make_node()

        manifest = node._build_tool_manifest(
            tools_schemas=[],
            tool_functions={},
        )

        assert manifest == ""

# ═══════════════════════════════════════════════════════════════════════
# Task 4.6: _UNSUPPORTED_ENGINES set modifications
# ═══════════════════════════════════════════════════════════════════════

class TestUnsupportedEngines:
    """Tests for _UNSUPPORTED_ENGINES set modifications."""

    def test_warning_logged_for_unsupported_engine(self):
        """Warning logged for Cohere engine."""
        from magic_agents.node_system.NodeLLM import NodeLLM
        from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel

        node = NodeLLM(data=LlmNodeModel(stream=False, json_output=False), node_id="test-llm", debug=False)

        # Mock client with cohere engine
        mock_client = MagicMock()
        mock_client.llm.engine_name = "cohere"

        with patch('magic_agents.node_system.NodeLLM.logger') as mock_logger:
            node._warn_unsupported_engine(mock_client)
            mock_logger.warning.assert_called_once()
            # logger.warning(fmt, node_id, engine) — engine is the third positional arg
            args = mock_logger.warning.call_args[0]
            assert len(args) >= 3
            engine_arg = args[2]
            assert "cohere" in engine_arg.lower()

    def test_no_warning_for_google_engine(self):
        """No warning logged for Google/Gemini engine."""
        from magic_agents.node_system.NodeLLM import NodeLLM
        from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel

        node = NodeLLM(data=LlmNodeModel(stream=False, json_output=False), node_id="test-llm", debug=False)

        # Mock client with google engine
        mock_client = MagicMock()
        mock_client.llm.engine_name = "google"

        with patch('magic_agents.node_system.NodeLLM.logger') as mock_logger:
            node._warn_unsupported_engine(mock_client)
            mock_logger.warning.assert_not_called()

    def test_no_warning_for_unknown_engine(self):
        """No warning logged for unknown provider."""
        from magic_agents.node_system.NodeLLM import NodeLLM
        from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel

        node = NodeLLM(data=LlmNodeModel(stream=False, json_output=False), node_id="test-llm", debug=False)

        mock_client = MagicMock()
        mock_client.llm.engine_name = "anthropic"

        with patch('magic_agents.node_system.NodeLLM.logger') as mock_logger:
            node._warn_unsupported_engine(mock_client)
            mock_logger.warning.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# Task 4.11: Source-node tagging for single-tool callables
# ═══════════════════════════════════════════════════════════════════════

class TestSourceNodeTagging:
    """Tests for source-node tagging of single-tool callables."""

    @pytest.mark.asyncio
    async def test_collision_reports_actual_source_ids(self):
        """Collision error between two Fetch nodes reports actual source node IDs."""
        tool1 = MockFetchToolCallable(name="colliding_tool")
        tool1.tool_callable._source_node_id = "fetch-node-1"
        tool2 = MockFetchToolCallable(name="colliding_tool")
        tool2.tool_callable._source_node_id = "fetch-node-2"

        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": tool1,
            "handle-tool-definition-1": tool2,
        })

        with pytest.raises(ToolNameCollisionError) as exc_info:
            await node._collect_tools()

        error = exc_info.value
        assert error.tool_name == "colliding_tool"
        assert sorted(error.source_nodes) == ["fetch-node-1", "fetch-node-2"]

    @pytest.mark.asyncio
    async def test_plain_callables_get_source_node_id(self):
        """Plain callables registered in tool_functions also get _source_node_id."""
        async def plain_tool(**kwargs):
            return "plain result"
        plain_tool.__name__ = "plain_helper"

        node = create_node_llm_with_inputs({
            "handle-tool-definition-0": plain_tool,
        })

        schemas, functions, _ = await node._collect_tools()

        assert "plain_helper" in functions
        assert hasattr(functions["plain_helper"], '_source_node_id')
        assert functions["plain_helper"]._source_node_id == "test-llm"
