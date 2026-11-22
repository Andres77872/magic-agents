"""Unit tests for EmitInterface helper methods.

Phase 9.4: Tests for:
- emit.user creates ChatCompletionModel with correct content
- emit.user returns dict with 'type': 'content'
- emit.debug returns dict with 'type': 'debug'
- emit.debug includes event_type, node_id, timestamp
- emit.feedback returns placeholder dict
"""
import pytest
from unittest.mock import MagicMock
from datetime import datetime, UTC

from magic_agents.hooks.emit_context import EmitInterface


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_node():
    """Create a minimal mock Node for EmitInterface."""
    node = MagicMock()
    node.node_id = "hook-node-1"
    node.__class__.__name__ = "NodeHook"
    node.prep.return_value = {
        "node": "NodeHook",
        "content": "processed_content",
    }
    return node


@pytest.fixture
def emit(mock_node):
    """Create an EmitInterface with mocked Node."""
    return EmitInterface(node=mock_node, node_id="hook-node-1")


class TestEmitUser:
    """Tests for emit.user() concrete implementation."""

    def test_user_returns_dict_with_type_content(self, emit):
        """emit.user() returns a dict with type='content'."""
        result = emit.user("Hello, world!")

        assert isinstance(result, dict)
        assert result["type"] == "content", (
            f"Expected type='content', got '{result.get('type')}'"
        )

    def test_user_includes_content_key(self, emit):
        """emit.user() includes 'content' key with prep-wrapped value."""
        result = emit.user("Hello, world!")

        assert "content" in result
        assert result["content"] == {
            "node": "NodeHook",
            "content": "processed_content",
        }

    def test_user_calls_node_prep(self, mock_node, emit):
        """emit.user() calls self._node.prep() with a ChatCompletionModel."""
        emit.user("Test message")

        mock_node.prep.assert_called_once()
        prep_arg = mock_node.prep.call_args[0][0]

        # Verify the ChatCompletionModel structure
        assert prep_arg.id == ''
        assert prep_arg.model == ''
        assert len(prep_arg.choices) == 1
        assert prep_arg.choices[0].delta.content == "Test message"
        assert prep_arg.choices[0].delta.role == "assistant"

    def test_user_passes_extras_to_model(self, mock_node, emit):
        """Extras dict is passed to ChatCompletionModel.extras."""
        extras = {"session_id": "abc-123"}
        emit.user("Test", extras=extras)

        prep_arg = mock_node.prep.call_args[0][0]
        assert prep_arg.extras == extras

    def test_user_default_extras_is_empty_dict(self, mock_node, emit):
        """When extras is None, ChatCompletionModel.extras is empty dict."""
        emit.user("Test")

        prep_arg = mock_node.prep.call_args[0][0]
        assert prep_arg.extras == {}

class TestEmitDebug:
    """Tests for emit.debug() structure contract."""

    def test_debug_returns_dict_with_type_debug(self, emit):
        """emit.debug() returns a dict with type='debug'."""
        result = emit.debug("CUSTOM_EVENT", {"value": 42})

        assert isinstance(result, dict)
        assert result["type"] == "debug", (
            f"Expected type='debug', got '{result.get('type')}'"
        )

    def test_debug_includes_event_type(self, emit):
        """emit.debug() content includes event_type."""
        result = emit.debug("CUSTOM_EVENT", {"value": 42})

        assert result["content"]["event_type"] == "CUSTOM_EVENT"

    def test_debug_includes_node_id(self, emit):
        """emit.debug() content includes node_id."""
        result = emit.debug("METRIC", {"value": 1})

        assert result["content"]["node_id"] == "hook-node-1"

    def test_debug_includes_iso_timestamp(self, emit):
        """emit.debug() content includes ISO format timestamp."""
        before = datetime.now(UTC).isoformat()
        result = emit.debug("EVENT", {})
        after = datetime.now(UTC).isoformat()

        timestamp = result["content"]["timestamp"]
        assert isinstance(timestamp, str)
        # UTC ISO-8601 timestamps sort chronologically, so the emitted value
        # must fall inside the interval measured around the production call.
        assert before <= timestamp <= after
        parsed = datetime.fromisoformat(timestamp)
        assert parsed.tzinfo == UTC

    def test_debug_merges_payload_into_content(self, emit):
        """emit.debug() merges payload fields into content."""
        payload = {"step": 2, "percent": 50.0, "status": "ok"}

        result = emit.debug("PROGRESS", payload)

        for key, value in payload.items():
            assert result["content"][key] == value, (
                f"Payload field '{key}' should be merged into content"
            )

    def test_debug_overlapping_keys(self, emit):
        """If payload contains event_type or node_id, payload takes precedence."""
        result = emit.debug("ORIGINAL", {"event_type": "OVERRIDE"})

        # Payload merge happens last, so it overrides
        assert result["content"]["event_type"] == "OVERRIDE"


class TestEmitFeedback:
    """Tests for emit.feedback() placeholder contract."""

    def test_feedback_returns_dict_with_type_feedback(self, emit):
        """emit.feedback() returns a dict with type='feedback'."""
        result = emit.feedback({"rating": 5})

        assert isinstance(result, dict)
        assert result["type"] == "feedback", (
            f"Expected type='feedback', got '{result.get('type')}'"
        )

    def test_feedback_includes_feedback_event_type(self, emit):
        """emit.feedback() content includes event_type='FEEDBACK'."""
        result = emit.feedback({"rating": 5})

        assert result["content"]["event_type"] == "FEEDBACK"

    def test_feedback_includes_node_id(self, emit):
        """emit.feedback() content includes node_id."""
        result = emit.feedback({"approved": True})

        assert result["content"]["node_id"] == "hook-node-1"

    def test_feedback_merges_event_data(self, emit):
        """emit.feedback() content includes all event_data fields."""
        data = {"rating": 5, "comment": "Great work!", "approved": True}

        result = emit.feedback(data)

        for key, value in data.items():
            assert result["content"][key] == value, (
                f"Event data field '{key}' should be in content"
            )

class TestEmitInterfaceEdgeCases:
    """Tests for edge cases across all emit methods."""

    def test_empty_message(self, emit):
        """emit.user() handles empty string message."""
        result = emit.user("")
        assert result["type"] == "content"

    def test_empty_event_data(self, emit):
        """emit.feedback() handles empty event data."""
        result = emit.feedback({})
        assert result["type"] == "feedback"
        assert result["content"]["event_type"] == "FEEDBACK"
        assert result["content"]["node_id"] == "hook-node-1"

