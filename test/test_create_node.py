"""
Unit tests for the create_node() factory function in agt_flow.py.

Tests cover:
- All 13 node type branches
- Unsupported type stub
- Conditional validation stub
- Inner node recursive build
- Debug flag propagation
"""
from unittest.mock import patch

import pytest

from magic_agents.agt_flow import create_node
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from magic_agents.node_system import (
    NodeChat, NodeLLM, NodeEND, NodeText, NodeConstant, NodeUserInput,
    NodeFetch, NodeClientLLM, NodeSendMessage, NodeParser,
    NodeLoop, NodeInner, NodeConditional,
)


class TestCreateNodeAllTypes:
    """Test create_node() returns correct class for each node type."""

    @pytest.mark.parametrize("node_type,expected_class", [
        (ModelAgentFlowTypesModel.TEXT, NodeText),
        (ModelAgentFlowTypesModel.CONSTANT, NodeConstant),
        (ModelAgentFlowTypesModel.USER_INPUT, NodeUserInput),
        (ModelAgentFlowTypesModel.LLM, NodeLLM),
        (ModelAgentFlowTypesModel.CHAT, NodeChat),
        (ModelAgentFlowTypesModel.CLIENT, NodeClientLLM),
        (ModelAgentFlowTypesModel.PARSER, NodeParser),
        (ModelAgentFlowTypesModel.FETCH, NodeFetch),
        (ModelAgentFlowTypesModel.SEND_MESSAGE, NodeSendMessage),
        (ModelAgentFlowTypesModel.CONDITIONAL, NodeConditional),
        (ModelAgentFlowTypesModel.LOOP, NodeLoop),
        (ModelAgentFlowTypesModel.INNER, NodeInner),
        (ModelAgentFlowTypesModel.END, NodeEND),
    ])
    def test_create_node_type(self, node_type, expected_class):
        """Each node type returns the correct class."""
        node_def = {"id": f"test-{node_type}", "type": node_type}

        # Add required data for specific types
        if node_type == ModelAgentFlowTypesModel.PARSER:
            node_def["data"] = {"text": "hello {{ input }}"}
        elif node_type == ModelAgentFlowTypesModel.TEXT:
            node_def["data"] = {"text": "hello"}
        elif node_type == ModelAgentFlowTypesModel.CONSTANT:
            node_def["data"] = {"value_type": "int", "value": "42"}
        elif node_type == ModelAgentFlowTypesModel.CONDITIONAL:
            node_def["data"] = {"condition": "{{ value }}"}
        elif node_type == ModelAgentFlowTypesModel.INNER:
            node_def["data"] = {
                "magic_flow": {
                    "type": "graph",
                    "nodes": [
                        {"id": "ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                        {"id": "end", "type": ModelAgentFlowTypesModel.END},
                    ],
                    "edges": [{"id": "e1", "source": "ui", "target": "end"}],
                }
            }
        elif node_type == ModelAgentFlowTypesModel.CHAT:
            # Chat needs message in data and load_chat
            node_def["data"] = {"message": "hello"}
            node = create_node(node_def, load_chat=lambda **kw: None)
            assert isinstance(node, expected_class)
            return
        elif node_type == ModelAgentFlowTypesModel.LLM:
            node_def["data"] = {"model": "gpt-4"}

        node = create_node(node_def, load_chat=None)
        assert isinstance(node, expected_class), (
            f"Expected {expected_class.__name__} for type '{node_type}', "
            f"got {node.__class__.__name__}"
        )

    def test_create_node_chat_with_load_chat(self):
        """CHAT node receives load_chat callable and message."""
        captured = {}
        def capture_load_chat(**kw):
            captured.update(kw)
            return "mock_chat"
        node_def = {
            "id": "chat-1",
            "type": ModelAgentFlowTypesModel.CHAT,
            "data": {"message": "hello chat"},
        }
        node = create_node(node_def, load_chat=capture_load_chat)
        assert isinstance(node, NodeChat)
        assert captured.get("message") == "hello chat"


class TestCreateNodeUnsupportedType:
    """Test create_node() handles unsupported node types."""

    def test_create_node_unsupported_type(self):
        """Unsupported type returns NodeEND stub with _error_info."""
        node_def = {
            "id": "bad-node",
            "type": "nonexistent_type",
        }
        node = create_node(node_def, load_chat=None)
        assert isinstance(node, NodeEND)
        assert hasattr(node, "_error_info")
        assert node._error_info["error_type"] == "UnsupportedNodeType"
        assert node._error_info["node_id"] == "bad-node"
        assert "nonexistent_type" in node._error_info["error_message"]
        assert "available_types" in node._error_info


class TestCreateNodeConditionalValidation:
    """Test create_node() conditional validation."""

    def test_create_node_conditional_valid(self):
        """Valid conditional node is created successfully."""
        node_def = {
            "id": "cond-1",
            "type": ModelAgentFlowTypesModel.CONDITIONAL,
            "data": {
                "condition": "{{ value }}",
                "merge_strategy": "flat",
            },
        }
        node = create_node(node_def, load_chat=None)
        assert isinstance(node, NodeConditional)
        assert node.condition_template == "{{ value }}"
        assert node.merge_strategy == "flat"

    def test_create_node_conditional_invalid_config(self):
        """Invalid conditional config returns NodeEND stub with error."""
        node_def = {
            "id": "cond-bad",
            "type": ModelAgentFlowTypesModel.CONDITIONAL,
            "data": {
                # Missing required 'condition' field
                "merge_strategy": "invalid_strategy",
            },
        }
        node = create_node(node_def, load_chat=None)
        assert isinstance(node, NodeEND)
        assert hasattr(node, "_error_info")
        assert node._error_info["error_type"] == "ConditionalValidationError"

    def test_create_node_conditional_empty_condition(self):
        """Empty condition string returns NodeEND stub."""
        node_def = {
            "id": "cond-empty",
            "type": ModelAgentFlowTypesModel.CONDITIONAL,
            "data": {"condition": ""},
        }
        node = create_node(node_def, load_chat=None)
        assert isinstance(node, NodeEND)
        assert hasattr(node, "_error_info")
        # Pydantic validates string_too_short for empty condition
        assert "ConditionalValidationError" in node._error_info["error_type"]


class TestCreateNodeInnerRecursive:
    """Test create_node() with inner node type."""

    def test_create_node_inner_builds_inner_graph(self):
        """Inner node stores magic_flow for later build()."""
        inner_flow = {
            "type": "graph",
            "nodes": [
                {"id": "inner-ui", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "inner-end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [{"id": "e1", "source": "inner-ui", "target": "inner-end"}],
        }
        node_def = {
            "id": "inner-1",
            "type": ModelAgentFlowTypesModel.INNER,
            "data": {"magic_flow": inner_flow},
        }
        node = create_node(node_def, load_chat=None)
        assert isinstance(node, NodeInner)
        # The magic_flow should be stored (build() will process it later)
        assert node.magic_flow is not None
        assert node.magic_flow["nodes"] == inner_flow["nodes"]


class TestCreateNodeDebugFlag:
    """Test create_node() propagates debug flag."""

    def test_create_node_debug_flag_propagated(self):
        """debug=True is passed to all nodes."""
        node_def = {
            "id": "txt-1",
            "type": ModelAgentFlowTypesModel.TEXT,
            "data": {"text": "hello"},
        }
        node = create_node(node_def, load_chat=None, debug=True)
        assert node.debug is True

    def test_create_node_debug_false_by_default(self):
        """debug defaults to False."""
        node_def = {
            "id": "txt-1",
            "type": ModelAgentFlowTypesModel.TEXT,
            "data": {"text": "hello"},
        }
        node = create_node(node_def, load_chat=None)
        assert node.debug is False

    def test_create_node_debug_propagates_to_conditional(self):
        """debug=True propagates to conditional nodes."""
        node_def = {
            "id": "cond-1",
            "type": ModelAgentFlowTypesModel.CONDITIONAL,
            "data": {"condition": "{{ value }}"},
        }
        node = create_node(node_def, load_chat=None, debug=True)
        assert node.debug is True

    def test_create_node_debug_propagates_to_loop(self):
        """debug=True propagates to loop nodes."""
        node_def = {
            "id": "loop-1",
            "type": ModelAgentFlowTypesModel.LOOP,
            "data": {},
        }
        node = create_node(node_def, load_chat=None, debug=True)
        assert node.debug is True


class TestCreateNodeHandles:
    """Test create_node() handle configuration."""

    def test_create_node_parser_custom_output_handle(self):
        """Parser node respects custom output handle."""
        node_def = {
            "id": "parser-1",
            "type": ModelAgentFlowTypesModel.PARSER,
            "data": {
                "text": "hello",
                "handles": {"output": "custom_output"},
            },
        }
        node = create_node(node_def, load_chat=None)
        assert node.OUTPUT_HANDLE == "custom_output"

    def test_create_node_loop_custom_handles(self):
        """Loop node respects custom handle configuration."""
        node_def = {
            "id": "loop-1",
            "type": ModelAgentFlowTypesModel.LOOP,
            "data": {
                "handles": {
                    "input_list": "custom_list",
                    "output_item": "custom_item",
                },
            },
        }
        node = create_node(node_def, load_chat=None)
        assert node.INPUT_HANDLE_LIST == "custom_list"
        assert node.OUTPUT_HANDLE_ITEM == "custom_item"

    def test_create_node_conditional_custom_handles(self):
        """Conditional node respects custom handle configuration."""
        node_def = {
            "id": "cond-1",
            "type": ModelAgentFlowTypesModel.CONDITIONAL,
            "data": {
                "condition": "{{ value }}",
                "handles": {"input": "custom_input"},
            },
        }
        node = create_node(node_def, load_chat=None)
        assert node.INPUT_HANDLE_CTX == "custom_input"


class TestCreateNodeEnvResolution:
    """Tests env placeholder handling during client node creation."""

    def test_create_client_resolves_env_placeholders_before_magicllm(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "resolved-openai-key")
        captured = {}

        def fake_magic_llm(**kwargs):
            captured.update(kwargs)
            return object()

        node_def = {
            "id": "client-env",
            "type": ModelAgentFlowTypesModel.CLIENT,
            "data": {
                "engine": "openai",
                "model": "gpt-4o-mini",
                "api_info": {
                    "api_key": "{{env.OPENAI_API_KEY}}",
                    "base_url": "https://api.openai.com/v1",
                },
            },
        }

        with patch("magic_agents.node_system.NodeClientLLM.MagicLLM", side_effect=fake_magic_llm):
            node = create_node(node_def, load_chat=None)

        assert isinstance(node, NodeClientLLM)
        assert captured["api_key"] == "resolved-openai-key"
        assert captured["private_key"] == "resolved-openai-key"
