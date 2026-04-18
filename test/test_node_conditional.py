"""
Unit tests for NodeConditional methods.

Tests cover:
- get_possible_outputs()
- validate_against_edges()
- _merge_inputs() collision detection
"""
import pytest

from magic_agents.node_system import NodeConditional
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel


def make_conditional(condition="{{ value }}", output_handles=None, default_handle=None,
                     merge_strategy="flat", handles=None, debug=False):
    """Create a NodeConditional with configurable params."""
    kwargs = {
        "condition": condition,
        "output_handles": output_handles,
        "default_handle": default_handle,
        "merge_strategy": merge_strategy,
        "node_id": "cond-test",
        "node_type": "conditional",
        "debug": debug,
    }
    if handles:
        kwargs["handles"] = handles
    return NodeConditional(**kwargs)


class TestConditionalGetPossibleOutputs:
    """Test get_possible_outputs()."""

    def test_conditional_get_possible_outputs_from_declared(self):
        """Returns declared output_handles."""
        node = make_conditional(output_handles=["handle_yes", "handle_no"])
        outputs = node.get_possible_outputs()
        assert set(outputs) == {"handle_yes", "handle_no"}

    def test_conditional_get_possible_outputs_from_template(self):
        """Infers handles from Jinja template string literals."""
        node = make_conditional(condition="{{ 'handle_yes' if value else 'handle_no' }}")
        outputs = node.get_possible_outputs()
        assert "handle_yes" in outputs
        assert "handle_no" in outputs

    def test_conditional_get_possible_outputs_empty(self):
        """Returns empty list when no handles can be determined."""
        node = make_conditional(condition="{{ value }}", output_handles=None)
        outputs = node.get_possible_outputs()
        assert outputs == []

    def test_conditional_get_possible_outputs_double_quotes(self):
        """Infers handles from double-quoted strings in template."""
        node = make_conditional(condition='{{ "handle_a" if x else "handle_b" }}')
        outputs = node.get_possible_outputs()
        assert "handle_a" in outputs
        assert "handle_b" in outputs


class TestConditionalValidateAgainstEdges:
    """Test validate_against_edges()."""

    def test_conditional_validate_against_edges_valid(self):
        """No errors when edges match declared outputs."""
        node = make_conditional(
            output_handles=["handle_yes", "handle_no"],
            default_handle="handle_no",
        )
        node.node_id = "cond-1"
        edges = [
            EdgeNodeModel(id="e1", source="cond-1", target="a", sourceHandle="handle_yes", targetHandle="in"),
            EdgeNodeModel(id="e2", source="cond-1", target="b", sourceHandle="handle_no", targetHandle="in"),
        ]
        result = node.validate_against_edges(edges)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_conditional_validate_against_edges_invalid(self):
        """Errors when edges don't match declared outputs."""
        node = make_conditional(output_handles=["handle_yes", "handle_no"])
        node.node_id = "cond-1"
        edges = [
            EdgeNodeModel(id="e1", source="cond-1", target="a", sourceHandle="handle_yes", targetHandle="in"),
            # Missing edge for "handle_no"
        ]
        result = node.validate_against_edges(edges)
        assert result["valid"] is False
        assert len(result["errors"]) >= 1
        assert result["errors"][0]["type"] == "missing_edges"
        assert "handle_no" in result["errors"][0]["handles"]

    def test_conditional_validate_against_edges_missing_default(self):
        """Error when default_handle has no edge."""
        node = make_conditional(default_handle="handle_fallback")
        node.node_id = "cond-1"
        edges = [
            EdgeNodeModel(id="e1", source="cond-1", target="a", sourceHandle="handle_other", targetHandle="in"),
        ]
        result = node.validate_against_edges(edges)
        assert result["valid"] is False
        default_errors = [e for e in result["errors"] if e["type"] == "missing_default_edge"]
        assert len(default_errors) == 1


class TestConditionalMergeInputs:
    """Test _merge_inputs() collision detection."""

    def test_conditional_merge_flat_no_collision(self):
        """Clean merge with no collisions."""
        node = make_conditional()
        node.inputs = {
            "handle_input": '{"key1": "value1"}',
        }
        ctx = node._merge_inputs()
        assert ctx["key1"] == "value1"
        assert node._merge_collisions == []

    def test_conditional_merge_flat_with_collision(self):
        """Later value wins, collision tracked in debug."""
        node = make_conditional(debug=True)
        node.inputs = {
            "handle_input": '{"shared": "first"}',
            "handle_extra": '{"shared": "second", "extra": "data"}',
        }
        ctx = node._merge_inputs()
        # Later value wins
        assert ctx["shared"] == "second"
        assert ctx["extra"] == "data"
        # Collision should be tracked
        assert len(node._merge_collisions) == 1
        assert node._merge_collisions[0]["key"] == "shared"

    def test_conditional_merge_namespaced(self):
        """Keys namespaced under handle name."""
        node = make_conditional(merge_strategy="namespaced")
        node.inputs = {
            "handle_input": '{"key1": "value1"}',
            "handle_extra": '{"key2": "value2"}',
        }
        ctx = node._merge_inputs()
        assert "handle_input" in ctx
        assert "handle_extra" in ctx
        assert ctx["handle_input"]["key1"] == "value1"
        assert ctx["handle_extra"]["key2"] == "value2"

    def test_conditional_merge_no_inputs(self):
        """Returns None when no inputs available."""
        node = make_conditional()
        node.inputs = {}
        ctx = node._merge_inputs()
        assert ctx is None

    def test_conditional_merge_value_alias(self):
        """Primary input is aliased as 'value'."""
        node = make_conditional()
        node.inputs = {
            "handle_input": '{"data": "test"}',
        }
        ctx = node._merge_inputs()
        assert ctx["value"] == {"data": "test"}

    def test_conditional_merge_value_alias_namespaced(self):
        """In namespaced mode, 'value' alias is set for primary input."""
        node = make_conditional(merge_strategy="namespaced")
        node.inputs = {
            "handle_input": '{"data": "test"}',
        }
        ctx = node._merge_inputs()
        assert ctx["value"] == {"data": "test"}
        assert ctx["handle_input"] == {"data": "test"}
