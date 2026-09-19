"""
Unit tests for NodeLoop.process() standalone behavior.

Note: During real loop execution, process() is bypassed — the executor
(execute_graph_loop_reactive) handles iteration directly. These tests
prove the node's standalone behavior, not the executor's loop logic.
"""
import json
import pytest

from magic_agents.node_system import NodeLoop
from magic_agents.models.model_agent_run_log import ModelAgentRunLog


def make_loop_node(inputs=None, handles=None, debug=False):
    """Create a NodeLoop with configurable inputs."""
    kwargs = {"node_id": "loop-test", "node_type": "loop", "debug": debug}
    if handles:
        kwargs["handles"] = handles
    node = NodeLoop(**kwargs)
    if inputs:
        node.inputs.update(inputs)
    return node


async def collect_process(node, chat_log):
    return [item async for item in node.process(chat_log)]


class TestNodeLoopProcessNoInput:
    """Test process() with no input."""

    @pytest.mark.asyncio
    async def test_loop_process_no_input(self):
        """Yields debug error about missing input."""
        node = make_loop_node(inputs={})
        chat_log = ModelAgentRunLog()

        results = await collect_process(node, chat_log)
        # Should yield a debug error
        assert len(results) >= 1
        debug_errors = [r for r in results
                        if r.get("type") == "debug"
                        and isinstance(r.get("content"), dict)
                        and r["content"].get("error_type") == "InputError"]
        assert len(debug_errors) == 1


class TestNodeLoopProcessInvalidJson:
    """Test process() with invalid JSON string."""

    @pytest.mark.asyncio
    async def test_loop_process_invalid_json(self):
        """Yields JSONParseError debug event."""
        node = make_loop_node(inputs={"handle_list": "not valid json {{{"})
        chat_log = ModelAgentRunLog()

        results = await collect_process(node, chat_log)
        json_errors = [r for r in results
                       if r.get("type") == "debug"
                       and isinstance(r.get("content"), dict)
                       and r["content"].get("error_type") == "JSONParseError"]
        assert len(json_errors) == 1

    @pytest.mark.asyncio
    async def test_loop_process_non_list(self):
        """Yields ValidationError debug event for valid JSON that isn't a list."""
        # "hello" is valid JSON (a string) but not a list
        node = make_loop_node(inputs={"handle_list": '"hello"'})
        chat_log = ModelAgentRunLog()

        results = await collect_process(node, chat_log)
        validation_errors = [r for r in results
                             if r.get("type") == "debug"
                             and isinstance(r.get("content"), dict)
                             and r["content"].get("error_type") == "ValidationError"]
        assert len(validation_errors) == 1


class TestNodeLoopProcessEmptyList:
    """Test process() with empty list."""

    @pytest.mark.asyncio
    async def test_loop_process_empty_list(self):
        """Yields items (none) then aggregation (empty)."""
        node = make_loop_node(inputs={"handle_list": []})
        chat_log = ModelAgentRunLog()

        results = await collect_process(node, chat_log)
        # Should yield the aggregation event (empty list)
        end_events = [r for r in results if r.get("type") == "handle_end"]
        assert len(end_events) == 1
        # Aggregation should be empty
        assert end_events[0]["content"]["content"] == []


class TestNodeLoopProcessValidList:
    """Test process() with valid list input."""

    @pytest.mark.asyncio
    async def test_loop_process_valid_list(self):
        """Yields items on handle_item, aggregation on handle_end."""
        node = make_loop_node(inputs={"handle_list": [1, 2, 3]})
        chat_log = ModelAgentRunLog()

        results = await collect_process(node, chat_log)

        # Should yield 3 item events + 1 end event
        item_events = [r for r in results if r.get("type") == "handle_item"]
        end_events = [r for r in results if r.get("type") == "handle_end"]
        assert len(item_events) == 3
        assert len(end_events) == 1
        # Check item content
        assert item_events[0]["content"]["content"] == 1
        assert item_events[1]["content"]["content"] == 2
        assert item_events[2]["content"]["content"] == 3

    @pytest.mark.asyncio
    async def test_loop_process_json_string_list(self):
        """JSON string list is parsed correctly."""
        node = make_loop_node(inputs={"handle_list": json.dumps(["a", "b"])})
        chat_log = ModelAgentRunLog()

        results = await collect_process(node, chat_log)
        item_events = [r for r in results if r.get("type") == "handle_item"]
        assert len(item_events) == 2
        assert item_events[0]["content"]["content"] == "a"
        assert item_events[1]["content"]["content"] == "b"

    @pytest.mark.asyncio
    async def test_loop_process_with_loop_feedback(self):
        """Loop feedback input is included in aggregation."""
        node = make_loop_node(inputs={
            "handle_list": [1, 2],
            "handle_loop": ["feedback1", "feedback2"],
        })
        chat_log = ModelAgentRunLog()

        results = await collect_process(node, chat_log)
        end_events = [r for r in results if r.get("type") == "handle_end"]
        assert len(end_events) == 1
        assert end_events[0]["content"]["content"] == ["feedback1", "feedback2"]

    @pytest.mark.asyncio
    async def test_loop_process_custom_handles(self):
        """Custom handle names are respected."""
        node = make_loop_node(
            inputs={"custom_list": ["x", "y"]},
            handles={"input_list": "custom_list", "output_item": "custom_item", "output_end": "custom_end"},
        )
        chat_log = ModelAgentRunLog()

        results = await collect_process(node, chat_log)
        item_events = [r for r in results if r.get("type") == "custom_item"]
        end_events = [r for r in results if r.get("type") == "custom_end"]
        assert len(item_events) == 2
        assert len(end_events) == 1
