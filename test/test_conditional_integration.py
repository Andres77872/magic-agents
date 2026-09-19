"""
Slice 12 — Conditional routing integration tests (no API keys).

Tests build + execute conditional flows using only text, parser, send_message,
and end nodes — no LLM needed. Uses JSON definitions as the primary test driver.
"""
import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.debug.events import DebugEventType


def extract_streamed_content(item):
    """Extract streamed content from send_message or LLM output."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "content":
        return ""
    content = item.get("content")
    if content is None:
        return ""
    if hasattr(content, "choices") and content.choices:
        delta = content.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            return delta.content
    return ""


class DebugCapture(list):
    """Async callback collector for typed direct-runtime debug events."""

    async def __call__(self, event):
        self.append(event)


def get_executed_nodes(debug_events) -> set:
    """Extract node IDs with a successful NODE_END event."""
    return {
        event.node_id
        for event in debug_events
        if event.event_type == DebugEventType.NODE_END
    }


def get_bypassed_nodes(debug_events) -> set:
    """Extract node IDs with a NODE_BYPASS event."""
    return {
        event.node_id
        for event in debug_events
        if event.event_type == DebugEventType.NODE_BYPASS
    }


def get_debug_items(results: list) -> list:
    """Extract debug items from collected results."""
    return [r for r in results if isinstance(r, dict) and r.get("type") == "debug"]


class TestConditionalEmptyCondition:
    """Tests for empty condition routing to fallback."""

    @pytest.mark.asyncio
    async def test_conditional_empty_condition_routes_to_fallback(self):
        """Empty condition result routes to the fallback/default path."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ '' }}",
                    "default_handle": "fallback",
                    "output_handles": ["primary", "fallback"],
                }},
                {"id": "send_primary", "type": "send_message", "data": {"message": "", "json_extras": "PRIMARY"}},
                {"id": "send_fallback", "type": "send_message", "data": {"message": "", "json_extras": "FALLBACK"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "send_primary",
                 "sourceHandle": "primary", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_fallback",
                 "sourceHandle": "fallback", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send_primary", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e5", "source": "send_fallback", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "FALLBACK" in content_str
        assert "PRIMARY" not in content_str
        executed = get_executed_nodes(debug_events)
        assert "send_fallback" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_primary" in bypassed

    @pytest.mark.asyncio
    async def test_conditional_nonempty_condition_routes_to_primary(self):
        """Non-empty condition result routes to the primary path, fallback bypassed."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'primary' }}",
                    "output_handles": ["primary", "fallback"],
                }},
                {"id": "send_primary", "type": "send_message", "data": {"message": "", "json_extras": "PRIMARY"}},
                {"id": "send_fallback", "type": "send_message", "data": {"message": "", "json_extras": "FALLBACK"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "send_primary",
                 "sourceHandle": "primary", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_fallback",
                 "sourceHandle": "fallback", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send_primary", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e5", "source": "send_fallback", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "PRIMARY" in content_str
        assert "FALLBACK" not in content_str
        executed = get_executed_nodes(debug_events)
        assert "send_primary" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_fallback" in bypassed


class TestConditionalFanOut:
    """Tests for fan-out (multiple targets from same handle)."""

    @pytest.mark.asyncio
    async def test_conditional_fan_out_three_targets(self):
        """All 3 targets of the selected handle execute."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'multi' }}",
                    "output_handles": ["multi", "single"],
                }},
                {"id": "send_1", "type": "send_message", "data": {"message": "", "json_extras": "T1"}},
                {"id": "send_2", "type": "send_message", "data": {"message": "", "json_extras": "T2"}},
                {"id": "send_3", "type": "send_message", "data": {"message": "", "json_extras": "T3"}},
                {"id": "send_single", "type": "send_message", "data": {"message": "", "json_extras": "SINGLE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "send_1",
                 "sourceHandle": "multi", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_2",
                 "sourceHandle": "multi", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "cond", "target": "send_3",
                 "sourceHandle": "multi", "targetHandle": "handle_send_extra"},
                {"id": "e5", "source": "cond", "target": "send_single",
                 "sourceHandle": "single", "targetHandle": "handle_send_extra"},
                {"id": "e6", "source": "send_1", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e7", "source": "send_2", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e8", "source": "send_3", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e9", "source": "send_single", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "T1" in content_str
        assert "T2" in content_str
        assert "T3" in content_str
        assert "SINGLE" not in content_str

        executed = get_executed_nodes(debug_events)
        assert "send_1" in executed
        assert "send_2" in executed
        assert "send_3" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_single" in bypassed


class TestConditionalMultiInput:
    """Tests for multi-input conditional merge strategies."""

    @pytest.mark.asyncio
    async def test_conditional_multi_input_flat_merge(self):
        """Both inputs merged flat, non-colliding keys accessible in condition."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser1", "type": "parser", "data": {"text": '{"status": "ok"}'}},
                {"id": "parser2", "type": "parser", "data": {"text": '{"score": 95}'}},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'pass' if status == 'ok' and score > 90 else 'fail' }}",
                    "merge_strategy": "flat",
                    "output_handles": ["pass", "fail"],
                }},
                {"id": "send_pass", "type": "send_message", "data": {"message": "", "json_extras": "PASSED"}},
                {"id": "send_fail", "type": "send_message", "data": {"message": "", "json_extras": "FAILED"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser1",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser2",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "parser1", "target": "cond",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_input_0"},
                {"id": "e4", "source": "parser2", "target": "cond",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_input_1"},
                {"id": "e5", "source": "cond", "target": "send_pass",
                 "sourceHandle": "pass", "targetHandle": "handle_send_extra"},
                {"id": "e6", "source": "cond", "target": "send_fail",
                 "sourceHandle": "fail", "targetHandle": "handle_send_extra"},
                {"id": "e7", "source": "send_pass", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e8", "source": "send_fail", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "PASSED" in content_str
        assert "FAILED" not in content_str
        executed = get_executed_nodes(debug_events)
        assert "send_pass" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_fail" in bypassed

    @pytest.mark.asyncio
    async def test_conditional_multi_input_namespaced_merge(self):
        """Namespaced merge keeps inputs separate under handle names."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser1", "type": "parser", "data": {"text": '{"status": "ok"}'}},
                {"id": "parser2", "type": "parser", "data": {"text": '{"status": "error"}'}},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'branch_a' if handle_input_0.status == 'ok' else 'branch_b' }}",
                    "merge_strategy": "namespaced",
                    "output_handles": ["branch_a", "branch_b"],
                }},
                {"id": "send_a", "type": "send_message", "data": {"message": "", "json_extras": "BRANCH_A"}},
                {"id": "send_b", "type": "send_message", "data": {"message": "", "json_extras": "BRANCH_B"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser1",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser2",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "parser1", "target": "cond",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_input_0"},
                {"id": "e4", "source": "parser2", "target": "cond",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_input_1"},
                {"id": "e5", "source": "cond", "target": "send_a",
                 "sourceHandle": "branch_a", "targetHandle": "handle_send_extra"},
                {"id": "e6", "source": "cond", "target": "send_b",
                 "sourceHandle": "branch_b", "targetHandle": "handle_send_extra"},
                {"id": "e7", "source": "send_a", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e8", "source": "send_b", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "BRANCH_A" in content_str
        assert "BRANCH_B" not in content_str
        executed = get_executed_nodes(debug_events)
        assert "send_a" in executed
        bypassed = get_bypassed_nodes(debug_events)
        assert "send_b" in bypassed


class TestConditionalErrorPaths:
    """Tests for conditional error and edge-case routing."""

    @pytest.mark.asyncio
    async def test_conditional_undefined_variable_bypasses_all(self):
        """Undefined variable in condition triggers BYPASS_ALL signal."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    # Nested access raises under Jinja's default Undefined policy.
                    "condition": "{{ undefined_var.required_attribute }}",
                    "output_handles": ["yes", "no"],
                }},
                {"id": "send_yes", "type": "send_message", "data": {"message": "", "json_extras": "YES"}},
                {"id": "send_no", "type": "send_message", "data": {"message": "", "json_extras": "NO"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "send_yes",
                 "sourceHandle": "yes", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "cond", "target": "send_no",
                 "sourceHandle": "no", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send_yes", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e5", "source": "send_no", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_items = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
            if (
                isinstance(item, dict)
                and item.get("type") == "debug"
                and item.get("content", {}).get("error_type") == "TemplateError"
            ):
                debug_items.append(item)

        # Both branches are bypassed after the template error.
        content_str = "".join(content_output)
        assert "YES" not in content_str
        assert "NO" not in content_str
        assert {"send_yes", "send_no"} <= get_bypassed_nodes(debug_events)
        assert len(debug_items) == 1

    @pytest.mark.asyncio
    async def test_conditional_no_default_empty_result_bypasses_all(self):
        """No default_handle and empty condition result → BYPASS_ALL + debug error.

        Uses a simpler graph with a single downstream node to avoid
        potential race conditions in multi-input bypass cascades.
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ '' }}",
                    "output_handles": ["yes"],
                }},
                {"id": "send_yes", "type": "send_message", "data": {"message": "", "json_extras": "YES"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "send_yes",
                 "sourceHandle": "yes", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send_yes", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        debug_items = []
        debug_events = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_events):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
            if (
                isinstance(item, dict)
                and item.get("type") == "debug"
                and item.get("content", {}).get("error_type") == "EmptyHandleError"
            ):
                debug_items.append(item)

        assert any(
            event.event_type == DebugEventType.GRAPH_END
            for event in debug_events
        )

        # send_yes should be bypassed
        content_str = "".join(content_output)
        assert "YES" not in content_str
        assert "send_yes" in get_bypassed_nodes(debug_events)
        # Should have debug error about routing
        assert len(debug_items) == 1
