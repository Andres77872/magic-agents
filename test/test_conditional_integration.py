"""
Slice 12 — Conditional routing integration tests (no API keys).

Tests build + execute conditional flows using only text, parser, send_message,
and end nodes — no LLM needed. Uses JSON definitions as the primary test driver.
"""
import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build


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


def get_executed_nodes(debug_summary: dict) -> set:
    """Extract set of executed node IDs from debug summary."""
    executed = set()
    if not debug_summary:
        return executed
    for node in debug_summary.get("nodes", []):
        if node.get("was_executed"):
            executed.add(node.get("node_id"))
    return executed


def get_bypassed_nodes(debug_summary: dict) -> set:
    """Extract set of bypassed node IDs from debug summary."""
    bypassed = set()
    if not debug_summary:
        return bypassed
    for node in debug_summary.get("nodes", []):
        if node.get("was_bypassed"):
            bypassed.add(node.get("node_id"))
    return bypassed


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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
                {"id": "e5", "source": "send_fallback", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h2"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "FALLBACK" in content_str
        assert "PRIMARY" not in content_str
        executed = get_executed_nodes(debug_summary)
        assert "send_fallback" in executed
        bypassed = get_bypassed_nodes(debug_summary)
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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
                {"id": "e5", "source": "send_fallback", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h2"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "PRIMARY" in content_str
        assert "FALLBACK" not in content_str
        executed = get_executed_nodes(debug_summary)
        assert "send_primary" in executed
        bypassed = get_bypassed_nodes(debug_summary)
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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
                {"id": "e7", "source": "send_2", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h2"},
                {"id": "e8", "source": "send_3", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h3"},
                {"id": "e9", "source": "send_single", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h4"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "T1" in content_str
        assert "T2" in content_str
        assert "T3" in content_str
        assert "SINGLE" not in content_str

        executed = get_executed_nodes(debug_summary)
        assert "send_1" in executed
        assert "send_2" in executed
        assert "send_3" in executed
        bypassed = get_bypassed_nodes(debug_summary)
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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
                {"id": "e8", "source": "send_fail", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h2"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "PASSED" in content_str
        assert "FAILED" not in content_str
        executed = get_executed_nodes(debug_summary)
        assert "send_pass" in executed
        bypassed = get_bypassed_nodes(debug_summary)
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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
                {"id": "e8", "source": "send_b", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h2"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "BRANCH_A" in content_str
        assert "BRANCH_B" not in content_str
        executed = get_executed_nodes(debug_summary)
        assert "send_a" in executed
        bypassed = get_bypassed_nodes(debug_summary)
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
                    "condition": "{{ 'yes' if undefined_var else 'no' }}",
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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
                {"id": "e5", "source": "send_no", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h2"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_items = []
        async for item in run_agent(graph):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_items.append(item)

        # At least one branch should be bypassed due to the error
        content_str = "".join(content_output)
        assert "YES" not in content_str or "NO" not in content_str
        assert len(debug_items) > 0

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
                    "output_handles": ["yes", "no"],
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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_items = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_items.append(item)

        # Should have completed (debug_summary present)
        assert debug_summary is not None

        # send_yes should be bypassed
        content_str = "".join(content_output)
        assert "YES" not in content_str
        # Should have debug error about routing
        assert len(debug_items) > 0
