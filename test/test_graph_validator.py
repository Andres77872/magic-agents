"""
Unit tests for graph validators.

Tests cover:
- ConditionalEdgeValidator.validate()
- validate_edge_connectivity()
- run_all_validations()
"""
import pytest

from magic_agents.util.graph_validator import (
    ConditionalEdgeValidator,
    validate_edge_connectivity,
    run_all_validations,
)
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.node_system import NodeConditional, NodeText, NodeEND


def make_mock_conditional(node_id: str, output_handles=None, default_handle=None, condition="{{ value }}"):
    """Create a mock conditional node."""
    node = NodeConditional(
        condition=condition,
        output_handles=output_handles,
        default_handle=default_handle,
        node_id=node_id,
        node_type="conditional",
    )
    return node


def make_mock_node(node_id: str, node_type="text"):
    """Create a mock non-conditional node."""
    if node_type == "text":
        from magic_agents.node_system import NodeText
        from magic_agents.models.factory.Nodes import TextNodeModel
        return NodeText(
            data=TextNodeModel(text="hello"),
            node_id=node_id,
            node_type="text",
        )
    return NodeEND(node_id=node_id, node_type="end")


class TestConditionalEdgeValidator:
    """Test ConditionalEdgeValidator.validate()."""

    def test_conditional_validator_missing_edge(self):
        """Error for declared handle with no edge."""
        nodes = {
            "cond": make_mock_conditional("cond", output_handles=["handle_yes", "handle_no"]),
        }
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="target", sourceHandle="handle_yes", targetHandle="in"),
            # Missing edge for "handle_no"
        ]
        errors = ConditionalEdgeValidator.validate(nodes, edges)
        assert len(errors) >= 1
        missing_errors = [e for e in errors if e.get("type") == "MissingConditionalEdge"]
        assert len(missing_errors) == 1
        assert "handle_no" in missing_errors[0]["missing_handles"]

    def test_conditional_validator_missing_default(self):
        """Error for default_handle with no edge."""
        nodes = {
            "cond": make_mock_conditional("cond", default_handle="handle_fallback"),
        }
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="target", sourceHandle="handle_other", targetHandle="in"),
        ]
        errors = ConditionalEdgeValidator.validate(nodes, edges)
        missing_default = [e for e in errors if e.get("type") == "MissingDefaultEdge"]
        assert len(missing_default) == 1
        assert missing_default[0]["default_handle"] == "handle_fallback"

    def test_conditional_validator_undeclared_outputs(self):
        """Warning when edges exist but no output_handles."""
        nodes = {
            "cond": make_mock_conditional("cond", output_handles=None),  # No declared outputs
        }
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="target", sourceHandle="handle_yes", targetHandle="in"),
        ]
        errors = ConditionalEdgeValidator.validate(nodes, edges)
        warnings = [e for e in errors if e.get("severity") == "warning"]
        undeclared = [w for w in warnings if w.get("type") == "UndeclaredOutputs"]
        assert len(undeclared) == 1

    def test_conditional_validator_valid(self):
        """Empty errors for correct config."""
        nodes = {
            "cond": make_mock_conditional("cond", output_handles=["handle_yes", "handle_no"],
                                          default_handle="handle_no"),
        }
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="a", sourceHandle="handle_yes", targetHandle="in"),
            EdgeNodeModel(id="e2", source="cond", target="b", sourceHandle="handle_no", targetHandle="in"),
        ]
        errors = ConditionalEdgeValidator.validate(nodes, edges)
        assert errors == []

    def test_conditional_validator_ignores_non_conditional_nodes(self):
        """Non-conditional nodes are skipped."""
        nodes = {
            "txt": make_mock_node("txt"),
            "end": make_mock_node("end", "end"),
        }
        edges = []
        errors = ConditionalEdgeValidator.validate(nodes, edges)
        assert errors == []


class TestValidateEdgeConnectivity:
    """Test validate_edge_connectivity()."""

    def test_validate_edge_connectivity_valid(self):
        """No errors for valid connectivity."""
        nodes = {"a": make_mock_node("a"), "b": make_mock_node("b")}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="out", targetHandle="in"),
        ]
        errors = validate_edge_connectivity(nodes, edges)
        assert errors == []

    def test_validate_edge_connectivity_invalid_source(self):
        """Error for non-existent source."""
        nodes = {"b": make_mock_node("b")}
        edges = [
            EdgeNodeModel(id="e1", source="nonexistent", target="b", sourceHandle="out", targetHandle="in"),
        ]
        errors = validate_edge_connectivity(nodes, edges)
        source_errors = [e for e in errors if e.get("type") == "InvalidEdgeSource"]
        assert len(source_errors) == 1
        assert "nonexistent" in source_errors[0]["error_message"]

    def test_validate_edge_connectivity_invalid_target(self):
        """Error for non-existent target."""
        nodes = {"a": make_mock_node("a")}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="nonexistent", sourceHandle="out", targetHandle="in"),
        ]
        errors = validate_edge_connectivity(nodes, edges)
        target_errors = [e for e in errors if e.get("type") == "InvalidEdgeTarget"]
        assert len(target_errors) == 1

    def test_validate_edge_connectivity_self_loop(self):
        """Warning for self-referencing edge."""
        nodes = {"a": make_mock_node("a")}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="a", sourceHandle="out", targetHandle="in"),
        ]
        errors = validate_edge_connectivity(nodes, edges)
        self_loop = [e for e in errors if e.get("type") == "SelfLoopEdge"]
        assert len(self_loop) == 1
        assert self_loop[0]["severity"] == "warning"

    def test_validate_edge_connectivity_duplicate(self):
        """Warning for duplicate edges."""
        nodes = {"a": make_mock_node("a"), "b": make_mock_node("b")}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="out", targetHandle="in"),
            EdgeNodeModel(id="e2", source="a", target="b", sourceHandle="out", targetHandle="in"),
        ]
        errors = validate_edge_connectivity(nodes, edges)
        dupes = [e for e in errors if e.get("type") == "DuplicateEdge"]
        assert len(dupes) == 1
        assert dupes[0]["severity"] == "warning"


class TestRunAllValidations:
    """Test run_all_validations()."""

    def test_run_all_validations_combines_results(self):
        """Both validators run, errors combined."""
        from magic_agents.models.factory.AgentFlowModel import AgentFlowModel

        nodes = {
            "cond": make_mock_conditional("cond", output_handles=["handle_yes"]),
            "txt": make_mock_node("txt"),
        }
        edges = [
            EdgeNodeModel(id="e1", source="nonexistent", target="txt", sourceHandle="out", targetHandle="in"),
        ]
        graph = AgentFlowModel(nodes=nodes, edges=edges)
        errors = run_all_validations(graph)

        # Should have both connectivity errors and conditional validation
        assert len(errors) >= 1
        # At least the invalid source error
        source_errors = [e for e in errors if e.get("type") == "InvalidEdgeSource"]
        assert len(source_errors) == 1

    def test_run_all_validations_empty_graph(self):
        """No errors for empty graph."""
        from magic_agents.models.factory.AgentFlowModel import AgentFlowModel

        graph = AgentFlowModel(nodes={}, edges=[])
        errors = run_all_validations(graph)
        assert errors == []
