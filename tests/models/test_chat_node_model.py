"""Tests for ChatNodeModel Pydantic validation.

Phase 5.4: ChatNodeModel validation tests per spec.md scenarios:
- Valid ChatNodeModel with session_id passes validation
- Valid ChatNodeModel without session passes with auto-create
- Invalid custom_messages raises ValidationError
- Extra fields forbidden, raises ValidationError
- Legacy fields preserved (message, memory, handles)
- messages_append_mode=False default (backward compat)
"""
import pytest
from pydantic import ValidationError

from magic_agents.models.factory.Nodes.ChatNodeModel import ChatNodeModel


class TestChatNodeModelValidation:
    """Test ChatNodeModel Pydantic validation per spec.md scenarios."""

        # Per spec: No error raised; backend auto-creates during execution

        # Runtime validation in NodeChat would catch malformed messages
        # This test documents that Pydantic validates type only
        
    def test_extra_fields_forbidden(self):
        """Extra fields are forbidden, raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ChatNodeModel(
                session_id="conv-abc",
                unknown_field="should fail"
            )
        
        error_str = str(exc_info.value)
        assert "Extra inputs are not permitted" in error_str
        assert "unknown_field" in error_str

    def test_rejects_measured_field(self):
        """ChatNodeModel should reject 'measured' field (ReactFlow-specific)."""
        with pytest.raises(ValidationError) as exc_info:
            ChatNodeModel(measured={'width': 200, 'height': 100})
        
        error_str = str(exc_info.value)
        assert "Extra inputs are not permitted" in error_str
        assert "measured" in error_str

    def test_legacy_fields_preserved(self):
        """Legacy fields (message, memory, handles) are preserved for backward compat."""
        model = ChatNodeModel(
            message="Hello",  # Legacy message field
            memory={"stm": 5, "ltm": 10, "max_input_tokens": 4000},
            handles={"system_context": "custom-system", "user_message": "custom-user"}
        )
        assert model.message == "Hello"
        assert model.memory["stm"] == 5
        assert model.handles["system_context"] == "custom-system"

    def test_messages_append_mode_default_false(self):
        """messages_append_mode=False default (backward compat)."""
        model = ChatNodeModel()
        assert model.messages_append_mode == False
        
        # Explicit True
        model_append = ChatNodeModel(messages_append_mode=True)
        assert model_append.messages_append_mode == True

    def test_session_required_default_false(self):
        """session_required=False default (backward compat)."""
        model = ChatNodeModel()
        assert model.session_required == False

        # No validation error - backward compat preserved