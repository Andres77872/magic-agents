"""Hard-fail unit tests for create_node() and build().

Phase 2: All invalid node configs MUST raise exceptions (never return NodeEND stubs).
"""
import pytest
from pydantic import ValidationError

from magic_agents.agt_flow import create_node, build
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from magic_agents.node_system import (
    NodeLLM,
)


class TestCreateNodeHardFailsOnExtraFields:
    """create_node() MUST raise on extra/unknown fields (not return stubs)."""

    def test_llm_node_with_model_field_succeeds(self):
        """LLM node accepts 'model' as a runtime/logging field."""
        node_def = {
            "id": "llm-bad",
            "type": ModelAgentFlowTypesModel.LLM,
            "data": {"model": "gpt-4"},
        }
        node = create_node(node_def, load_chat=None)
        assert isinstance(node, NodeLLM)

    def test_llm_node_with_extra_field_tool_inputs_raises(self):
        """LLM node with 'toolInputs' field raises ValidationError (not NodeEND stub)."""
        node_def = {
            "id": "llm-bad",
            "type": ModelAgentFlowTypesModel.LLM,
            "data": {"toolInputs": ["handle-tool-definition-0"]},
        }
        with pytest.raises(ValidationError):
            create_node(node_def, load_chat=None)

    def test_fetch_node_with_extra_field_raises(self):
        """Fetch node with unknown extra field raises ValidationError (not stub)."""
        node_def = {
            "id": "fetch-bad",
            "type": ModelAgentFlowTypesModel.FETCH,
            "data": {"bogus_field": "i-do-not-exist"},
        }
        with pytest.raises(ValidationError):
            create_node(node_def, load_chat=None)

    @pytest.mark.parametrize("ui_only_type", ["note", "section"])
    def test_ui_only_studio_node_types_raise_value_error(self, ui_only_type):
        """Studio-only node types must be filtered by API before magic-agents build."""
        node_def = {
            "id": f"{ui_only_type}-1",
            "type": ui_only_type,
            "data": {},
        }
        with pytest.raises(ValueError, match=f"Unsupported node type: {ui_only_type}"):
            create_node(node_def, load_chat=None)

    def test_conditional_node_with_extra_field_raises(self):
        """Conditional node with extra field raises ValidationError (not stub)."""
        node_def = {
            "id": "cond-bad",
            "type": ModelAgentFlowTypesModel.CONDITIONAL,
            "data": {"condition": "{{ value }}", "bogus_field": "nope"},
        }
        with pytest.raises(ValidationError):
            create_node(node_def, load_chat=None)

    def test_inner_node_with_extra_field_raises(self):
        """Inner node with extra field raises ValidationError (not stub)."""
        node_def = {
            "id": "inner-bad",
            "type": ModelAgentFlowTypesModel.INNER,
            "data": {"bogus_field": "nope"},
        }
        with pytest.raises(ValidationError):
            create_node(node_def, load_chat=None)

    def test_memory_node_with_extra_field_raises(self):
        """Memory node with extra field raises ValidationError (not stub)."""
        node_def = {
            "id": "mem-bad",
            "type": ModelAgentFlowTypesModel.MEMORY,
            "data": {"bogus_field": "nope"},
        }
        with pytest.raises(ValidationError):
            create_node(node_def, load_chat=None)

    def test_parser_node_with_extra_field_raises(self):
        """Parser node with extra field raises ValidationError (not stub)."""
        node_def = {
            "id": "parser-bad",
            "type": ModelAgentFlowTypesModel.PARSER,
            "data": {"text": "hello {{ input }}", "bogus_field": "nope"},
        }
        with pytest.raises(ValidationError):
            create_node(node_def, load_chat=None)


class TestBuildHardFails:
    """build() MUST propagate exceptions from create_node()."""

    def test_build_accepts_llm_node_with_model_field(self):
        """build() accepts LLM model metadata."""
        graph_data = {
            "nodes": [
                {"id": "start", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "llm-bad", "type": ModelAgentFlowTypesModel.LLM, "data": {"model": "gpt-4"}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "llm-bad"},
                {"id": "e2", "source": "llm-bad", "target": "end"},
            ],
        }
        flow = build(graph_data, message="test")

        assert flow is not None
        assert "llm-bad" in flow.nodes
        assert isinstance(flow.nodes["llm-bad"], NodeLLM)

    def test_build_raises_on_fetch_node_with_extra_field(self):
        """build() raises ValidationError when Fetch node has extra field."""
        graph_data = {
            "nodes": [
                {"id": "start", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "fetch-bad", "type": ModelAgentFlowTypesModel.FETCH, "data": {"bogus": "x"}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "fetch-bad"},
                {"id": "e2", "source": "fetch-bad", "target": "end"},
            ],
        }
        with pytest.raises(ValidationError):
            build(graph_data, message="test")

    def test_build_raises_on_unsupported_node_type(self):
        """build() raises ValueError for unsupported node type."""
        graph_data = {
            "nodes": [
                {"id": "start", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "weird", "type": "alien_type"},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "weird"},
                {"id": "e2", "source": "weird", "target": "end"},
            ],
        }
        with pytest.raises(ValueError, match="Unsupported node type: alien_type"):
            build(graph_data, message="test")

    def test_build_succeeds_with_clean_graph(self):
        """build() succeeds with a fully clean graph (no exceptions)."""
        graph_data = {
            "nodes": [
                {"id": "start", "type": ModelAgentFlowTypesModel.USER_INPUT},
                {"id": "llm-1", "type": ModelAgentFlowTypesModel.LLM, "data": {"temperature": 0.7}},
                {"id": "end", "type": ModelAgentFlowTypesModel.END},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "llm-1"},
                {"id": "e2", "source": "llm-1", "target": "end"},
            ],
        }
        # Should not raise
        flow = build(graph_data, message="test")
        assert flow is not None
        assert "llm-1" in flow.nodes
        assert isinstance(flow.nodes["llm-1"], NodeLLM)
