"""Tests for MCP node factory/build happy path (regression for blocker #1).

Phase 2: All invalid node configs now hard-fail instead of returning stubs.
"""
import pytest
from pydantic import ValidationError
from magic_agents.agt_flow import create_node, build
from magic_agents.node_system import NodeMcp
from magic_agents.models.factory.Nodes.McpNodeModel import McpNodeModel


class TestCreateNodeMcp:
    """Tests for create_node() with MCP type."""
    
    def test_create_node_mcp_returns_node_mcp_instance(self):
        """Test create_node with type 'mcp' returns NodeMcp instance (regression for blocker #1)."""
        node_def = {
            "id": "mcp-node-1",
            "type": "mcp",
            "data": {
                "servers": [
                    {
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["mcp-server-filesystem", "/tmp"]
                    }
                ]
            }
        }
        
        # This should NOT raise TypeError about duplicate node_type
        node_instance = create_node(node_def, load_chat=lambda: None, debug=False)
        
        # Should be a NodeMcp instance
        assert isinstance(node_instance, NodeMcp)
        assert node_instance.node_id == "mcp-node-1"
        assert node_instance.node_type == "mcp"
    
    def test_create_node_mcp_with_http_config(self):
        """Test create_node with HTTP MCP config."""
        node_def = {
            "id": "mcp-http-node",
            "type": "mcp",
            "data": {
                "servers": [
                    {
                        "transport": "http",
                        "url": "http://localhost:8080/mcp"
                    }
                ]
            }
        }
        
        node_instance = create_node(node_def, load_chat=lambda: None, debug=False)
        
        assert isinstance(node_instance, NodeMcp)
        assert node_instance.node_id == "mcp-http-node"
    
    def test_create_node_mcp_with_custom_handles(self):
        """Test create_node MCP with custom handles."""
        node_def = {
            "id": "mcp-custom",
            "type": "mcp",
            "data": {
                "servers": [
                    {"transport": "stdio", "command": "test-server"}
                ],
                "handles": {
                    "output": "custom-tool-output"
                }
            }
        }
        
        node_instance = create_node(node_def, load_chat=lambda: None, debug=False)
        
        assert isinstance(node_instance, NodeMcp)
        assert node_instance.OUTPUT_HANDLE == "custom-tool-output"
    
    def test_create_node_mcp_config_accessible(self):
        """Test create_node MCP config is accessible on instance."""
        node_def = {
            "id": "mcp-config-test",
            "type": "mcp",
            "data": {
                "servers": [
                    {
                        "transport": "stdio",
                        "command": "my-server",
                        "args": ["arg1", "arg2"],
                        "prefix": "mytools",
                        "tool_timeout": 60.0
                    }
                ]
            }
        }
        
        node_instance = create_node(node_def, load_chat=lambda: None, debug=False)
        
        # Config should be validated McpNodeModel
        assert isinstance(node_instance._config, McpNodeModel)
        assert len(node_instance._config.servers) == 1
        assert node_instance._config.servers[0].command == "my-server"
        assert node_instance._config.servers[0].prefix == "mytools"
        assert node_instance._config.servers[0].tool_timeout == 60.0
    
class TestBuildGraphWithMcp:
    """Tests for build() with MCP nodes in graph."""
    
    def test_build_graph_with_mcp_node(self):
        """Test build() creates MCP node correctly in graph."""
        graph_data = {
            "nodes": [
                {"id": "start", "type": "user_input"},
                {"id": "mcp-1", "type": "mcp", "data": {
                    "servers": [{"transport": "stdio", "command": "mcp-server"}]
                }},
                {"id": "llm-1", "type": "llm"},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "mcp-1"},
                {"id": "e2", "source": "mcp-1", "target": "llm-1"},
                {"id": "e3", "source": "llm-1", "target": "end"}
            ]
        }
        
        # build() should create all nodes including MCP
        flow_model = build(graph_data, message="test")
        
        # MCP node should be in nodes dict
        assert "mcp-1" in flow_model.nodes
        assert isinstance(flow_model.nodes["mcp-1"], NodeMcp)
    
    def test_build_assigns_tool_handles_for_mcp_to_llm_edge(self):
        """Test build() auto-assigns tool handles for MCP→LLM edge."""
        graph_data = {
            "nodes": [
                {"id": "start", "type": "user_input"},
                {"id": "mcp-tools", "type": "mcp", "data": {
                    "servers": [{"transport": "stdio", "command": "server"}]
                }},
                {"id": "llm-main", "type": "llm"},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "mcp-tools"},
                {"id": "e2", "source": "mcp-tools", "target": "llm-main"},
                {"id": "e3", "source": "llm-main", "target": "end"}
            ]
        }
        
        flow_model = build(graph_data, message="test")
        
        # Find the MCP→LLM edge
        mcp_to_llm_edge = None
        for edge in flow_model.edges:
            if edge.source == "mcp-tools" and edge.target == "llm-main":
                mcp_to_llm_edge = edge
                break
        
        assert mcp_to_llm_edge is not None
        # Should have auto-assigned sourceHandle and targetHandle
        assert mcp_to_llm_edge.sourceHandle == "handle-tool-definition"
        assert mcp_to_llm_edge.targetHandle == "handle-tool-definition-0"
    
    def test_build_multiple_mcp_nodes_to_single_llm(self):
        """Test build() handles multiple MCP nodes feeding one LLM."""
        graph_data = {
            "nodes": [
                {"id": "start", "type": "user_input"},
                {"id": "mcp-fs", "type": "mcp", "data": {
                    "servers": [{"transport": "stdio", "command": "filesystem-server"}]
                }},
                {"id": "mcp-db", "type": "mcp", "data": {
                    "servers": [{"transport": "stdio", "command": "db-server"}]
                }},
                {"id": "llm-main", "type": "llm"},
                {"id": "end", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "mcp-fs"},
                {"id": "e2", "source": "start", "target": "mcp-db"},
                {"id": "e3", "source": "mcp-fs", "target": "llm-main"},
                {"id": "e4", "source": "mcp-db", "target": "llm-main"},
                {"id": "e5", "source": "llm-main", "target": "end"}
            ]
        }
        
        flow_model = build(graph_data, message="test")
        
        # Both MCP nodes should exist
        assert isinstance(flow_model.nodes["mcp-fs"], NodeMcp)
        assert isinstance(flow_model.nodes["mcp-db"], NodeMcp)
        
        # Should have sequential tool handles
        mcp_fs_edge = None
        mcp_db_edge = None
        for edge in flow_model.edges:
            if edge.source == "mcp-fs" and edge.target == "llm-main":
                mcp_fs_edge = edge
            if edge.source == "mcp-db" and edge.target == "llm-main":
                mcp_db_edge = edge
        
        assert mcp_fs_edge is not None
        assert mcp_db_edge is not None
        # Different target handles for each MCP source
        assert mcp_fs_edge.targetHandle.startswith("handle-tool-definition-")
        assert mcp_db_edge.targetHandle.startswith("handle-tool-definition-")


class TestInvalidMcpNode:
    """Tests for invalid MCP node configurations."""
    
    def test_create_node_mcp_empty_servers_raises(self):
        """Test create_node with empty servers raises ValidationError (no stub)."""
        node_def = {
            "id": "mcp-empty",
            "type": "mcp",
            "data": {
                "servers": []  # Empty servers - invalid config
            }
        }
        
        # Phase 2: Invalid node configs MUST hard-fail
        with pytest.raises(ValidationError) as exc_info:
            create_node(node_def, load_chat=lambda: None, debug=False)
        assert "servers" in str(exc_info.value).lower()
    
    def test_create_node_mcp_invalid_url_raises(self):
        """Test create_node with invalid HTTP URL raises ValidationError (no stub)."""
        node_def = {
            "id": "mcp-bad-url",
            "type": "mcp",
            "data": {
                "servers": [
                    {
                        "transport": "http",
                        "url": "not-a-valid-url"
                    }
                ]
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            create_node(node_def, load_chat=lambda: None, debug=False)
        assert "url" in str(exc_info.value).lower()
    
    def test_create_node_mcp_missing_command_raises(self):
        """Test create_node with stdio but missing command raises ValidationError (no stub)."""
        node_def = {
            "id": "mcp-no-command",
            "type": "mcp",
            "data": {
                "servers": [
                    {"transport": "stdio"}  # Missing required command
                ]
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            create_node(node_def, load_chat=lambda: None, debug=False)
        assert "command" in str(exc_info.value).lower()

def test_package_level_node_mcp_stays_a_class_after_the_submodule_is_imported():
    """Regression: importing magic_agents.node_system.NodeMcp replaced the package
    attribute with the module, so later ``isinstance(x, NodeMcp)`` raised TypeError."""
    import importlib
    import types

    module = importlib.import_module("magic_agents.node_system.NodeMcp")
    node = create_node({"id": "m", "type": "mcp", "data": {"servers": [{"command": "srv"}]}}, load_chat=lambda: None)

    from magic_agents.node_system import NodeMcp as exported

    assert not isinstance(exported, types.ModuleType)
    assert isinstance(node, exported) and isinstance(node, module.NodeMcp)
