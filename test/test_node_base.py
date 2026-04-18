"""
Unit tests for the Node base class.

Tests cover:
- add_parent() routing
- get_input() with defaults/required
- mark_bypassed()
- yield_static output format
- Error handling in __call__
"""
import asyncio
import pytest

from magic_agents.node_system.Node import Node
from magic_agents.models.model_agent_run_log import ModelAgentRunLog


class ConcreteNode(Node):
    """Concrete test double for Node abstract class."""
    async def process(self, chat_log):
        yield self.yield_static("result", content_type="end")


class ConcreteErrorNode(Node):
    """Node that raises an exception during process."""
    async def process(self, chat_log):
        # Must be an async generator — yield first, then raise
        yield self.yield_static("before_error", content_type="end")
        raise ValueError("test error")


class TestNodeAddParent:
    """Test add_parent() routing."""

    def test_add_parent_routes_content_by_handle(self):
        """Correct target_handle key in inputs."""
        node = ConcreteNode(node_id="test")
        parent_outputs = {
            "out1": {"node": "ParentNode", "content": "hello"},
        }
        node.add_parent(parent_outputs, source_handle="out1", target_handle="my_input")
        assert node.inputs["my_input"] == "hello"

    def test_add_parent_missing_source_handle_noop(self):
        """No crash, no input set when source handle missing."""
        node = ConcreteNode(node_id="test")
        parent_outputs = {"other": {"node": "ParentNode", "content": "hello"}}
        node.add_parent(parent_outputs, source_handle="missing", target_handle="my_input")
        assert "my_input" not in node.inputs

    def test_add_parent_content_is_none_noop(self):
        """No input set when parent output content is None."""
        node = ConcreteNode(node_id="test")
        parent_outputs = {"out1": None}
        node.add_parent(parent_outputs, source_handle="out1", target_handle="my_input")
        assert "my_input" not in node.inputs


class TestNodeGetInput:
    """Test get_input() behavior."""

    def test_get_input_returns_value(self):
        """Returns stored input."""
        node = ConcreteNode(node_id="test")
        node.inputs["key"] = "value"
        assert node.get_input("key") == "value"

    def test_get_input_returns_default(self):
        """Returns default when key missing."""
        node = ConcreteNode(node_id="test")
        assert node.get_input("missing", default="fallback") == "fallback"

    def test_get_input_required_raises(self):
        """Raises ValueError when required and missing."""
        node = ConcreteNode(node_id="test")
        with pytest.raises(ValueError) as exc_info:
            node.get_input("missing", required=True)
        assert "Required input 'missing' not found" in str(exc_info.value)


class TestNodeMarkBypassed:
    """Test mark_bypassed() behavior."""

    def test_mark_bypassed_sets_flags(self):
        """was_bypassed=True, was_executed=False (only in debug mode)."""
        node = ConcreteNode(node_id="test", debug=True)
        node.mark_bypassed()
        assert node._debug_info.was_bypassed is True
        assert node._debug_info.was_executed is False

    def test_mark_bypassed_noop_without_debug(self):
        """mark_bypassed does nothing when debug=False."""
        node = ConcreteNode(node_id="test", debug=False)
        node.mark_bypassed()
        assert node._debug_info is None


class TestNodeYieldStatic:
    """Test yield_static output format."""

    def test_yield_static_produces_correct_format(self):
        """Produces {"type": ..., "content": {"node": ..., "content": ...}}."""
        node = ConcreteNode(node_id="test")
        result = node.yield_static("my_data", content_type="custom_type")
        assert result["type"] == "custom_type"
        assert result["content"]["node"] == "ConcreteNode"
        assert result["content"]["content"] == "my_data"

    def test_yield_static_default_content_type(self):
        """Default content_type is 'end'."""
        node = ConcreteNode(node_id="test")
        result = node.yield_static("data")
        assert result["type"] == "end"


class TestNodeCall:
    """Test __call__ async execution."""

    def test_node_call_yields_results(self):
        """__call__ yields process results."""
        node = ConcreteNode(node_id="test")
        chat_log = ModelAgentRunLog()

        async def _test():
            results = []
            async for item in node(chat_log):
                results.append(item)
            # Node yields at least the "end" content event
            # (telemetry may add additional debug events)
            end_events = [r for r in results if r.get("type") == "end"]
            assert len(end_events) >= 1
            assert end_events[0]["content"]["content"] == "result"

        asyncio.get_event_loop().run_until_complete(_test())

    def test_node_call_yields_precomputed_response(self):
        """If _response is set, yields immediately without calling process."""
        node = ConcreteNode(node_id="test")
        node._response = "precomputed"
        chat_log = ModelAgentRunLog()

        async def _test():
            results = []
            async for item in node(chat_log):
                results.append(item)
            assert len(results) == 1
            assert results[0]["type"] == "end"
            assert results[0]["content"]["content"] == "precomputed"

        asyncio.get_event_loop().run_until_complete(_test())

    def test_node_call_handles_exception(self):
        """Exception in process yields debug error, doesn't propagate."""
        node = ConcreteErrorNode(node_id="test", debug=True)
        chat_log = ModelAgentRunLog()

        async def _test():
            results = []
            async for item in node(chat_log):
                results.append(item)
            # Should get at least the pre-error yield and a debug error event
            assert len(results) >= 1
            debug_events = [r for r in results if r.get("type") == "debug"]
            assert len(debug_events) >= 1
            # Debug event content has error_type at the content level
            content = debug_events[0].get("content", {})
            assert "error" in content or "error_type" in content

        asyncio.get_event_loop().run_until_complete(_test())


class TestNodePrep:
    """Test prep() content wrapping."""

    def test_prep_wraps_content(self):
        """prep() returns standardized response structure."""
        node = ConcreteNode(node_id="test")
        result = node.prep("data")
        assert result == {"node": "ConcreteNode", "content": "data"}
        assert node._response == "data"

    def test_prep_sets_response(self):
        """prep() stores content in _response."""
        node = ConcreteNode(node_id="test")
        node.prep("stored")
        assert node.response == "stored"
