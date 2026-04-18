"""
Tests for ConditionEvaluator protocol and Jinja2Evaluator implementation.

Slice 3a: Define ConditionEvaluator protocol
Slice 3b: Refactor NodeConditional.process() to use evaluator strategy
"""

import pytest

from magic_agents.execution.condition_evaluator import ConditionEvaluator
from magic_agents.execution.condition_evaluator_jinja2 import (
    Jinja2Evaluator,
    Jinja2StrictEvaluator,
)


class TestConditionEvaluatorProtocol:
    """Slice 3a: Test ConditionEvaluator protocol."""

    def test_jinja2_evaluator_implements_protocol(self):
        """Jinja2Evaluator implements ConditionEvaluator protocol."""
        evaluator = Jinja2Evaluator()
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(evaluator, 'validate_syntax')
        assert callable(evaluator.evaluate)
        assert callable(evaluator.validate_syntax)

    def test_jinja2_strict_evaluator_implements_protocol(self):
        """Jinja2StrictEvaluator implements ConditionEvaluator protocol."""
        evaluator = Jinja2StrictEvaluator()
        assert hasattr(evaluator, 'evaluate')
        assert hasattr(evaluator, 'validate_syntax')
        assert callable(evaluator.evaluate)
        assert callable(evaluator.validate_syntax)


class TestJinja2Evaluator:
    """Test Jinja2Evaluator matches current NodeConditional behavior."""

    def test_evaluate_simple_variable(self):
        """Evaluate simple variable reference."""
        evaluator = Jinja2Evaluator()
        result = evaluator.evaluate("{{ value }}", {"value": "handle_yes"})
        assert result == "handle_yes"

    def test_evaluate_conditional_expression(self):
        """Evaluate conditional expression."""
        evaluator = Jinja2Evaluator()
        result = evaluator.evaluate("{{ 'yes' if value else 'no' }}", {"value": True})
        assert result == "yes"

        result = evaluator.evaluate("{{ 'yes' if value else 'no' }}", {"value": False})
        assert result == "no"

    def test_evaluate_undefined_variable_renders_empty(self):
        """Undefined variable renders to empty string (default Jinja2 behavior)."""
        evaluator = Jinja2Evaluator()
        result = evaluator.evaluate("{{ undefined_var }}", {"other": "value"})
        assert result == ""

    def test_evaluate_strips_whitespace(self):
        """Result is stripped of leading/trailing whitespace."""
        evaluator = Jinja2Evaluator()
        result = evaluator.evaluate("  {{ value }}  ", {"value": "handle_yes"})
        assert result == "handle_yes"

    def test_validate_syntax_valid_template(self):
        """Valid template returns True."""
        evaluator = Jinja2Evaluator()
        assert evaluator.validate_syntax("{{ value }}") is True
        assert evaluator.validate_syntax("{{ 'yes' if value else 'no' }}") is True

    def test_validate_syntax_invalid_template(self):
        """Invalid template returns False."""
        evaluator = Jinja2Evaluator()
        assert evaluator.validate_syntax("{{ invalid syntax }}") is False

    def test_evaluate_with_dict_context(self):
        """Evaluate with dict context."""
        evaluator = Jinja2Evaluator()
        result = evaluator.evaluate(
            "{{ 'yes' if data.approved else 'no' }}",
            {"data": {"approved": True}}
        )
        assert result == "yes"


class TestJinja2StrictEvaluator:
    """Test Jinja2StrictEvaluator raises errors for undefined variables."""

    def test_evaluate_undefined_variable_raises_error(self):
        """Undefined variable raises UndefinedError in strict mode."""
        import jinja2
        evaluator = Jinja2StrictEvaluator()
        with pytest.raises(jinja2.UndefinedError):
            evaluator.evaluate("{{ undefined_var }}", {"other": "value"})

    def test_evaluate_defined_variable_works(self):
        """Defined variable works in strict mode."""
        evaluator = Jinja2StrictEvaluator()
        result = evaluator.evaluate("{{ value }}", {"value": "handle_yes"})
        assert result == "handle_yes"

    def test_validate_syntax_valid_template(self):
        """Valid template returns True."""
        evaluator = Jinja2StrictEvaluator()
        assert evaluator.validate_syntax("{{ value }}") is True

    def test_validate_syntax_invalid_template(self):
        """Invalid template returns False."""
        evaluator = Jinja2StrictEvaluator()
        assert evaluator.validate_syntax("{{ invalid syntax }}") is False
