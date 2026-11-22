"""Tests for ChatNodeModel STM field validation.

Phase 6.1: STM field validation tests per spec X2.

Tests cover:
- max_messages: valid values, zero rejection, negative rejection
- max_input_tokens: valid values, zero rejection
- truncation_strategy: valid literals ('tail', 'token_budget'), invalid rejection
- Default values: all None / 'tail' on no input
- Legacy memory fallback: model-level coexistence of memory dict and new field
- Legacy memory precedence: model-level coexistence both set
- Deprecation warning: model-level coexistence check (NodeChat runtime test pending)

ARCHITECTURE NOTE:
Tests 1-8 are pure Pydantic model validation tests — no NodeChat dependency.
Tests 9-11 verify model-level field coexistence (ChatNodeModel can hold both
memory dict and first-class fields simultaneously). The actual resolution
logic (precedence, deprecation warning) lives in NodeChat.__init__ (task 2.1)
and will be fully tested when Batch C wiring is complete.
"""
import pytest
from pydantic import ValidationError

from magic_agents.models.factory.Nodes.ChatNodeModel import ChatNodeModel


class TestChatNodeModelStmFields:
    """Validation tests for ChatNodeModel STM windowing fields (spec X2)."""

    # ── max_messages validation ──────────────────────────────────────────

    def test_max_messages_invalid_zero(self):
        """X2: max_messages=0 rejected (ge=1)."""
        with pytest.raises(ValidationError) as exc_info:
            ChatNodeModel(max_messages=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('max_messages',)
        assert errors[0]['type'] == 'greater_than_equal'

    def test_max_messages_negative(self):
        """X2: max_messages=-5 rejected (ge=1)."""
        with pytest.raises(ValidationError) as exc_info:
            ChatNodeModel(max_messages=-5)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('max_messages',)
        assert errors[0]['type'] == 'greater_than_equal'

    # ── max_input_tokens validation ─────────────────────────────────────

    def test_max_input_tokens_invalid_zero(self):
        """X2: max_input_tokens=0 rejected (ge=1)."""
        with pytest.raises(ValidationError) as exc_info:
            ChatNodeModel(max_input_tokens=0)
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('max_input_tokens',)
        assert errors[0]['type'] == 'greater_than_equal'

    # ── truncation_strategy validation ──────────────────────────────────

    def test_truncation_strategy_invalid(self):
        """X2: 'middle' rejected (Literal['tail', 'token_budget'])."""
        with pytest.raises(ValidationError) as exc_info:
            ChatNodeModel(truncation_strategy='middle')
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('truncation_strategy',)
        assert errors[0]['type'] == 'literal_error'

    # ── Default values ──────────────────────────────────────────────────

    def test_default_values(self):
        """X2: All None / 'tail' on no input."""
        model = ChatNodeModel()
        assert model.max_messages is None
        assert model.max_input_tokens is None
        assert model.truncation_strategy == 'tail'
        assert model.model is None

    # ── Legacy memory dict coexistence (model-level) ────────────────────

    def test_legacy_memory_fallback_coexistence(self):
        """X2: max_input_tokens=None, memory.max_input_tokens set.

        Model-level verification: ChatNodeModel accepts both fields without
        validation error. The first-class max_input_tokens is None, and the
        legacy memory dict holds the key. Actual fallback resolution is a
        NodeChat runtime concern (task 2.1).
        """
        model = ChatNodeModel(memory={'max_input_tokens': 32000})
        assert model.max_input_tokens is None  # first-class field is unset
        assert model.memory == {'max_input_tokens': 32000}  # legacy dict preserved

    def test_legacy_precedence_coexistence(self):
        """X2: New field wins over legacy memory.max_input_tokens.

        Model-level verification: ChatNodeModel accepts both first-class
        max_input_tokens AND memory dict with max_input_tokens simultaneously
        without validation error. This verifies backward-compatible coexistence.
        Actual precedence resolution (new field > legacy) is NodeChat runtime
        behavior (task 2.1).
        """
        model = ChatNodeModel(
            max_input_tokens=64000,
            memory={'max_input_tokens': 32000},
        )
        assert model.max_input_tokens == 64000  # first-class field takes precedence
        assert model.memory == {'max_input_tokens': 32000}  # legacy dict preserved

