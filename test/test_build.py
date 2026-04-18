"""
Unit tests for the build() function in agt_flow.py.

Tests cover:
- Flat structure building
- Nested content wrapper normalization
- Message injection into USER_INPUT and CHAT nodes
- Void sentinel node creation
- END node auto-edge creation
- Validation error preservation
- Inner graph recursive build
"""
import pytest
from copy import deepcopy
from unittest.mock import patch

from magic_agents.agt_flow import build, validate_graph
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.node_system import NodeUserInput, NodeChat, NodeEND, NodeText, NodeInner
from magic_agents.util.const import HANDLE_VOID


class TestBuildFlatStructure:
    """Test build() with flat JSON structure."""

    def test_build_flat_structure_returns_agent_flow_model(self):
        """build() returns an AgentFlowModel with correct node count."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "ui", "target": "end"},
            ],
        }
        result = build(agt, message="hello", load_chat=None)
        assert isinstance(result, AgentFlowModel)
        # Should have 3 nodes: user_input, end, + void sentinel
        assert len(result.nodes) == 3

    def test_build_flat_structure_preserves_graph_type(self):
        """build() preserves the graph type from input."""
        agt = {
            "type": "chat",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "ui", "target": "end"}],
        }
        result = build(agt, message="hello", load_chat=None)
        assert result.type == "chat"

    def test_build_flat_structure_with_text_node(self):
        """build() handles text nodes in flat structure."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "txt", "type": ModelAgentFlowTypesModel.TEXT,
                 "data": {"text": "Hello {{ handle_parser_input }}"}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "ui", "target": "txt",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "txt", "target": "end"},
            ],
        }
        result = build(agt, message="test", load_chat=None)
        assert len(result.nodes) == 4  # ui, txt, end, void
        txt_node = result.nodes.get("txt")
        assert txt_node is not None
        assert txt_node.__class__.__name__ == "NodeText"


class TestBuildNestedContentNormalization:
    """Test build() normalizes nested content wrapper."""

    def test_build_nested_content_normalization(self):
        """build() extracts nodes/edges from content wrapper."""
        agt = {
            "type": "graph",
            "debug": False,
            "content": {
                "nodes": [
                    {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                    {"id": "end", "type": ModelAgentFlowTypesModel.END},
                ],
                "edges": [{"id": "e1", "source": "ui", "target": "end"}],
            },
        }
        result = build(agt, message="hello", load_chat=None)
        assert isinstance(result, AgentFlowModel)
        assert len(result.nodes) == 3  # ui, end, void

    def test_build_nested_preserves_top_level_properties(self):
        """build() preserves additional top-level properties from nested structure."""
        agt = {
            "type": "graph",
            "debug": True,
            "custom_prop": "custom_value",
            "content": {
                "nodes": [
                    {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                    {"id": "end", "type": ModelAgentFlowTypesModel.END},
                ],
                "edges": [{"id": "e1", "source": "ui", "target": "end"}],
            },
        }
        result = build(agt, message="hello", load_chat=None)
        assert result.debug is True

    def test_build_flat_vs_nested_produce_same_structure(self):
        """Flat and nested structures should produce equivalent graphs."""
        flat = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "ui", "target": "end"}],
        }
        nested = {
            "type": "graph",
            "content": {
                "nodes": [
                    {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                    {"id": "end", "type": ModelAgentFlowTypesModel.END},
                ],
                "edges": [{"id": "e1", "source": "ui", "target": "end"}],
            },
        }
        flat_result = build(flat, message="hello", load_chat=None)
        nested_result = build(nested, message="hello", load_chat=None)
        # Both should have same number of nodes (including void)
        assert len(flat_result.nodes) == len(nested_result.nodes)

    def test_build_resolves_env_placeholders_without_touching_runtime_templates(self, monkeypatch):
        """build() resolves {{env.*}} while preserving runtime placeholders."""
        monkeypatch.setenv("OPENAI_API_KEY", "resolved-openai-key")

        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {
                    "id": "client",
                    "type": ModelAgentFlowTypesModel.CLIENT,
                    "data": {
                        "engine": "openai",
                        "model": "gpt-4o-mini",
                        "api_info": {
                            "api_key": "{{env.OPENAI_API_KEY}}",
                            "base_url": "https://api.openai.com/v1",
                        },
                    },
                },
                {
                    "id": "txt",
                    "type": ModelAgentFlowTypesModel.TEXT,
                    "data": {"text": "Hello {{ handle_parser_input }}"},
                },
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "ui", "target": "txt"},
                {"id": "e2", "source": "txt", "target": "end"},
            ],
        }

        with patch("magic_agents.node_system.NodeClientLLM.MagicLLM", return_value=object()):
            result = build(agt, message="hello", load_chat=None)

        client_node = result.nodes.get("client")
        text_node = result.nodes.get("txt")

        assert client_node is not None
        assert client_node.client is not None
        assert text_node is not None
        assert text_node._text == "Hello {{ handle_parser_input }}"


class TestBuildMessageInjection:
    """Test build() injects message into appropriate nodes."""

    def test_build_injects_message_into_user_input(self):
        """USER_INPUT node data has 'text' = message."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "ui", "target": "end"}],
        }
        result = build(agt, message="test message", load_chat=None)
        ui_node = result.nodes.get("ui")
        assert ui_node is not None
        assert isinstance(ui_node, NodeUserInput)
        # Message is stored in _text (set via UserInputNodeModel)
        assert ui_node._text == "test message"

    def test_build_injects_message_into_chat(self):
        """CHAT node receives message via load_chat call."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "chat", "type": ModelAgentFlowTypesModel.CHAT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "ui", "target": "chat"},
                {"id": "e2", "source": "chat", "target": "end"},
            ],
        }
        captured_message = None
        def capture_load_chat(**kw):
            nonlocal captured_message
            captured_message = kw.get("message")
            return "mock_chat"
        result = build(agt, message="chat message", load_chat=capture_load_chat)
        chat_node = result.nodes.get("chat")
        assert chat_node is not None
        assert isinstance(chat_node, NodeChat)
        assert captured_message == "chat message"

    def test_build_injects_images_into_user_input(self):
        """USER_INPUT node receives images list."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "ui", "target": "end"}],
        }
        images = ["img1.png", "img2.png"]
        result = build(agt, message="test", images=images, load_chat=None)
        ui_node = result.nodes.get("ui")
        assert ui_node is not None
        assert ui_node.images == images


class TestBuildVoidSentinelNode:
    """Test build() creates void sentinel node."""

    def test_build_creates_void_sentinel_node(self):
        """Graph has one extra void node."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "ui", "target": "end"}],
        }
        result = build(agt, message="hello", load_chat=None)
        # 2 user nodes + 1 void = 3
        assert len(result.nodes) == 3
        # Find the void node (it's a NodeEND with no user-facing id)
        void_nodes = [
            nid for nid, n in result.nodes.items()
            if isinstance(n, NodeEND) and nid not in ("ui", "end")
        ]
        assert len(void_nodes) == 1

    def test_build_auto_connects_end_nodes_to_void(self):
        """END nodes get auto-edge to void sentinel."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end1", "type": ModelAgentFlowTypesModel.END},
                {"id": "end2", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "ui", "target": "end1"},
                {"id": "e2", "source": "ui", "target": "end2"},
            ],
        }
        result = build(agt, message="hello", load_chat=None)
        # 3 user nodes + 1 void = 4
        assert len(result.nodes) == 4
        # Both END nodes should have auto-edges to void
        void_id = [
            nid for nid, n in result.nodes.items()
            if isinstance(n, NodeEND) and nid not in ("ui", "end1", "end2")
        ][0]
        end_edges = [e for e in result.edges if e.source in ("end1", "end2")]
        void_edges = [e for e in end_edges if e.target == void_id]
        assert len(void_edges) == 2

    def test_build_edges_without_target_handle_route_to_void(self):
        """Edges without targetHandle get routed to void."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "ui", "target": "end"}],
        }
        result = build(agt, message="hello", load_chat=None)
        # The edge without targetHandle should be routed to void
        void_id = [
            nid for nid, n in result.nodes.items()
            if isinstance(n, NodeEND) and nid not in ("ui", "end")
        ][0]
        void_edge = [e for e in result.edges if e.target == void_id and e.source == "ui"]
        assert len(void_edge) == 1


class TestBuildValidationErrors:
    """Test build() preserves validation errors."""

    def test_build_preserves_validation_errors(self):
        """Invalid graph has _validation_errors populated."""
        agt = {
            "type": "graph",
            "nodes": [
                # Missing USER_INPUT node — should fail validation
                {"id": "txt", "type": ModelAgentFlowTypesModel.TEXT,
                 "data": {"text": "hello"}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "txt", "target": "end"}],
        }
        result = build(agt, message="hello", load_chat=None)
        assert result._validation_errors is not None
        assert len(result._validation_errors) > 0
        assert any("USER_INPUT" in e.get("error_message", "")
                   for e in result._validation_errors)

    def test_build_valid_graph_has_no_validation_errors(self):
        """Valid graph has no validation errors."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "ui", "target": "end"}],
        }
        result = build(agt, message="hello", load_chat=None)
        assert result._validation_errors is None


class TestBuildInnerGraph:
    """Test build() recursively builds inner graphs."""

    def test_build_inner_graph_recursive(self):
        """Inner node's inner_graph is built recursively."""
        inner_flow = {
            "type": "graph",
            "nodes": [
                {"id": "inner-ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "inner-end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "inner-ui", "target": "inner-end"}],
        }
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "inner", "type": ModelAgentFlowTypesModel.INNER,
                 "data": {"magic_flow": inner_flow}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "ui", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "end"},
            ],
        }
        result = build(agt, message="hello", load_chat=None)
        inner_node = result.nodes.get("inner")
        assert inner_node is not None
        assert isinstance(inner_node, NodeInner)
        assert inner_node.inner_graph is not None
        assert isinstance(inner_node.inner_graph, AgentFlowModel)
        # Inner graph should have its own nodes built
        assert "inner-ui" in inner_node.inner_graph.nodes
        assert "inner-end" in inner_node.inner_graph.nodes

    def test_build_debug_flag_propagates_to_nodes(self):
        """debug=True is passed to all nodes."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "txt", "type": ModelAgentFlowTypesModel.TEXT,
                 "data": {"text": "hello"}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "ui", "target": "txt"},
                {"id": "e2", "source": "txt", "target": "end"},
            ],
        }
        result = build(agt, message="hello", load_chat=None)
        for node_id, node in result.nodes.items():
            assert node.debug is True, f"Node {node_id} should have debug=True"


class TestBuildEdgeConnectivityValidation:
    """Test that build() catches edge connectivity errors at build time."""

    def test_build_invalid_edge_source(self):
        """build() records error when edge references non-existent source."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e0", "source": "nonexistent", "target": "end",
                 "sourceHandle": "h1", "targetHandle": "h2"},
            ],
        }
        result = build(agt, message="test", load_chat=None)
        errors = result._validation_errors or []
        source_errors = [e for e in errors if e.get("error_type") == "InvalidEdgeSource"]
        assert len(source_errors) == 1
        assert "nonexistent" in source_errors[0]["error_message"]

    def test_build_invalid_edge_target(self):
        """build() records error when edge references non-existent target."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e0", "source": "input", "target": "nonexistent",
                 "sourceHandle": "h1", "targetHandle": "h2"},
            ],
        }
        result = build(agt, message="test", load_chat=None)
        errors = result._validation_errors or []
        target_errors = [e for e in errors if e.get("error_type") == "InvalidEdgeTarget"]
        assert len(target_errors) == 1

    def test_build_self_loop_edge(self):
        """build() records error for self-referencing edges AND filters them out."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "node-a", "type": ModelAgentFlowTypesModel.TEXT, "data": {"text": "hello"}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e0", "source": "node-a", "target": "node-a",
                 "sourceHandle": "h1", "targetHandle": "h2"},
            ],
        }
        result = build(agt, message="test", load_chat=None)
        errors = result._validation_errors or []
        self_loops = [e for e in errors if e.get("error_type") == "SelfLoopEdge"]
        assert len(self_loops) == 1

        # The self-loop edge must be filtered out — it should NOT appear in the graph
        self_loop_edges = [
            e for e in result.edges
            if e.source == "node-a" and e.target == "node-a"
        ]
        assert len(self_loop_edges) == 0, (
            "Self-loop edge should be filtered out at build time, not just recorded as error"
        )

    def test_build_valid_graph_no_connectivity_errors(self):
        """build() has no connectivity errors for valid graph."""
        agt = {
            "type": "graph",
            "nodes": [
                {"id": "input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "text", "type": ModelAgentFlowTypesModel.TEXT, "data": {"text": "hello"}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e0", "source": "input", "target": "text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e1", "source": "text", "target": "end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle-5"},
            ],
        }
        result = build(agt, message="test", load_chat=None)
        errors = result._validation_errors or []
        conn_errors = [
            e for e in errors
            if e.get("error_type") in ("InvalidEdgeSource", "InvalidEdgeTarget", "SelfLoopEdge")
        ]
        assert len(conn_errors) == 0


class TestBuildValidationFailFast:
    """Test that blocking validation errors prevent runtime execution."""

    @pytest.mark.asyncio
    async def test_missing_user_input_fails_fast_no_hang(self):
        """Graph with no USER_INPUT node yields errors and returns immediately, no hang."""
        from magic_agents import run_agent
        import asyncio

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "txt", "type": ModelAgentFlowTypesModel.TEXT, "data": {"text": "hello"}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e0", "source": "txt", "target": "end",
                 "sourceHandle": "handle_text_output", "targetHandle": "h1"},
            ],
        }
        result = build(agt, message="test", load_chat=None)
        assert result._validation_errors is not None
        assert any("USER_INPUT" in e.get("error_message", "") for e in result._validation_errors)

        # Run the agent — it should yield the validation error and return quickly
        # (not hang waiting for nodes that can never execute)
        start = asyncio.get_event_loop().time()
        events = []
        async with asyncio.timeout(10.0):
            async for item in run_agent(result):
                events.append(item)
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete in under 1 second (fail-fast, not hang)
        assert elapsed < 1.0, f"Should fail fast, took {elapsed:.2f}s"

        # Should have yielded the validation error as a debug event
        error_events = [
            e for e in events
            if e.get("type") == "debug"
            and e.get("content", {}).get("error_type") == "GraphValidationError"
        ]
        assert len(error_events) > 0, "Should have yielded validation error as debug event"

        # Should NOT have a debug_summary (execution was aborted before nodes ran)
        summary_events = [e for e in events if e.get("type") == "debug_summary"]
        assert len(summary_events) == 0, "Should not produce debug_summary when execution aborted"

    @pytest.mark.asyncio
    async def test_self_loop_filtered_no_runtime_hang(self):
        """Graph with self-loop edge has edge filtered at build, runs without hang."""
        from magic_agents import run_agent
        import asyncio

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "node-a", "type": ModelAgentFlowTypesModel.TEXT,
                 "data": {"text": "hello {{ handle_input }}"}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e0", "source": "input", "target": "node-a",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e-self", "source": "node-a", "target": "node-a",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_input"},
                {"id": "e1", "source": "node-a", "target": "end",
                 "sourceHandle": "handle_text_output", "targetHandle": "h1"},
            ],
        }
        result = build(agt, message="test", load_chat=None)

        # Self-loop error should be recorded
        assert result._validation_errors is not None
        self_loops = [e for e in result._validation_errors if e.get("error_type") == "SelfLoopEdge"]
        assert len(self_loops) == 1

        # Self-loop edge should be filtered out
        self_loop_edges = [
            e for e in result.edges
            if e.source == "node-a" and e.target == "node-a"
        ]
        assert len(self_loop_edges) == 0

        # Running the agent should complete quickly (no hang from self-loop)
        events = []
        async with asyncio.timeout(10.0):
            async for item in run_agent(result):
                events.append(item)

        # Should have completed without timeout
        event_types = [e.get("type") for e in events]
        assert "debug_summary" in event_types, "Should have completed with debug summary"
