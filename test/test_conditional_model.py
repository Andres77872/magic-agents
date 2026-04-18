"""
Unit tests for ConditionalNodeModel validation.

Tests Pydantic validation of conditional node configuration including:
- Valid configurations
- Invalid configurations
- Jinja2 syntax validation
- Output handle validation
- Default handle validation
"""

import pytest
from pydantic import ValidationError

from magic_agents.models.factory.Nodes.ConditionalNodeModel import (
    ConditionalNodeModel,
    ConditionalSignalTypes,
)


class TestConditionalNodeModel:
    """Unit tests for ConditionalNodeModel."""
    
    def test_valid_simple_condition(self):
        """Test valid simple if/else condition."""
        model = ConditionalNodeModel(
            condition="{{ 'yes' if flag else 'no' }}"
        )
        assert model.condition == "{{ 'yes' if flag else 'no' }}"
        assert model.merge_strategy == "flat"
        assert model.output_handles is None
        assert model.default_handle is None
    
    def test_valid_with_all_options(self):
        """Test valid config with all options specified."""
        model = ConditionalNodeModel(
            condition="{{ status }}",
            merge_strategy="namespaced",
            output_handles=["success", "error", "timeout"],
            default_handle="error"
        )
        assert model.condition == "{{ status }}"
        assert model.merge_strategy == "namespaced"
        assert model.output_handles == ["success", "error", "timeout"]
        assert model.default_handle == "error"
    
    def test_valid_with_handles_mapping(self):
        """Test valid config with custom handles mapping."""
        model = ConditionalNodeModel(
            condition="{{ result }}",
            handles={"input": "my_custom_input", "context": "my_context"}
        )
        assert model.handles == {"input": "my_custom_input", "context": "my_context"}
    
    def test_valid_namespaced_strategy(self):
        """Test valid config with namespaced merge strategy."""
        model = ConditionalNodeModel(
            condition="{{ 'branch_a' if handle_a.status == 'ok' else 'branch_b' }}",
            merge_strategy="namespaced"
        )
        assert model.merge_strategy == "namespaced"
    
    def test_invalid_empty_condition(self):
        """Test that empty condition is rejected."""
        with pytest.raises(ValidationError) as exc:
            ConditionalNodeModel(condition="")
        assert "String should have at least 1 character" in str(exc.value)
    
    def test_invalid_missing_condition(self):
        """Test that missing condition is rejected."""
        with pytest.raises(ValidationError) as exc:
            ConditionalNodeModel()
        assert "condition" in str(exc.value).lower()
    
    def test_invalid_jinja2_syntax(self):
        """Test that invalid Jinja2 syntax is rejected."""
        with pytest.raises(ValidationError) as exc:
            ConditionalNodeModel(condition="{{ if broken }}")
        assert "jinja2" in str(exc.value).lower() or "syntax" in str(exc.value).lower()
    
    def test_invalid_jinja2_unclosed_block(self):
        """Test that unclosed Jinja2 block is rejected."""
        with pytest.raises(ValidationError) as exc:
            ConditionalNodeModel(condition="{% if x %}")
        assert "jinja2" in str(exc.value).lower() or "syntax" in str(exc.value).lower()
    
    def test_invalid_merge_strategy(self):
        """Test that invalid merge strategy is rejected."""
        with pytest.raises(ValidationError) as exc:
            ConditionalNodeModel(
                condition="{{ x }}",
                merge_strategy="invalid"
            )
        assert "merge_strategy" in str(exc.value).lower()
    
    def test_invalid_output_handle_empty(self):
        """Test that empty output handle name is rejected."""
        with pytest.raises(ValidationError) as exc:
            ConditionalNodeModel(
                condition="{{ x }}",
                output_handles=["valid", ""]
            )
        assert "empty" in str(exc.value).lower()
    
    def test_invalid_output_handle_system_reserved(self):
        """Test that system-reserved handle names are rejected."""
        with pytest.raises(ValidationError) as exc:
            ConditionalNodeModel(
                condition="{{ x }}",
                output_handles=["valid", "__bypass_all__"]
            )
        assert "reserved" in str(exc.value).lower()
    
    def test_invalid_default_handle_not_in_outputs(self):
        """Test that default_handle must be in output_handles if specified."""
        with pytest.raises(ValidationError) as exc:
            ConditionalNodeModel(
                condition="{{ x }}",
                output_handles=["a", "b"],
                default_handle="c"  # Not in output_handles
            )
        assert "default_handle" in str(exc.value).lower()
    
    def test_invalid_default_handle_system_reserved(self):
        """Test that system-reserved default handle is rejected."""
        with pytest.raises(ValidationError) as exc:
            ConditionalNodeModel(
                condition="{{ x }}",
                default_handle="__error__"
            )
        assert "reserved" in str(exc.value).lower()
    
    def test_valid_default_handle_without_outputs(self):
        """Test that default_handle can be specified without output_handles."""
        model = ConditionalNodeModel(
            condition="{{ x }}",
            default_handle="fallback"
        )
        assert model.default_handle == "fallback"
    
    def test_extra_fields_allowed(self):
        """Test that extra fields from JSON are allowed."""
        model = ConditionalNodeModel(
            condition="{{ x }}",
            position={"x": 100, "y": 200},
            custom_field="value"
        )
        assert model.condition == "{{ x }}"
        # Extra fields should be accessible
        assert model.model_extra.get("position") == {"x": 100, "y": 200}


class TestConditionalSignalTypes:
    """Unit tests for ConditionalSignalTypes."""
    
    def test_bypass_all_constant(self):
        """Test BYPASS_ALL signal constant."""
        assert ConditionalSignalTypes.BYPASS_ALL == "__bypass_all__"
    
    def test_default_constant(self):
        """Test DEFAULT signal constant."""
        assert ConditionalSignalTypes.DEFAULT == "__default__"
    
    def test_error_constant(self):
        """Test ERROR signal constant."""
        assert ConditionalSignalTypes.ERROR == "__error__"
    
    def test_is_system_signal_true(self):
        """Test is_system_signal returns True for system signals."""
        assert ConditionalSignalTypes.is_system_signal("__bypass_all__") is True
        assert ConditionalSignalTypes.is_system_signal("__error__") is True
        assert ConditionalSignalTypes.is_system_signal("__custom__") is True
    
    def test_is_system_signal_false(self):
        """Test is_system_signal returns False for user signals."""
        assert ConditionalSignalTypes.is_system_signal("success") is False
        assert ConditionalSignalTypes.is_system_signal("error") is False  # User-defined
        assert ConditionalSignalTypes.is_system_signal("__partial") is False
        assert ConditionalSignalTypes.is_system_signal("partial__") is False
        assert ConditionalSignalTypes.is_system_signal("_single_") is False
