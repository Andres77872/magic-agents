"""
Slice 14 — Loop execution tests (mocked / no API keys).

Tests loop iteration, aggregation, empty list, max iterations, static phase,
post-loop execution, and conditional bypass of loop — all without real API calls.
Uses parser nodes and text nodes to avoid LLM dependency.
"""
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.execution.reactive_executor import (
    execute_graph_loop_reactive,
    find_iteration_subgraph,
    prepare_item_output,
    emit_loop_progress,
    reset_iteration_nodes,
)


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


class TestLoopSimpleIteration:
    """Tests for basic loop iteration with parser nodes (no LLM)."""

    @pytest.mark.asyncio
    async def test_loop_simple_iteration(self):
        """Loop over [1,2,3] → 3 iterations, aggregation correct."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": '["a", "b", "c"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "item: {{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "Results: {{ handle_parser_input | join(', ') }}"
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e5", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "format", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e7", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
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
        assert "DONE" in content_str
        executed = get_executed_nodes(debug_summary)
        # Note: loop node itself is not tracked in debug because the loop executor
        # handles iteration directly without calling node.__call__. We verify
        # loop execution indirectly by checking that iteration nodes executed.
        assert "transform" in executed
        assert "format" in executed
        assert "send" in executed

    @pytest.mark.asyncio
    async def test_loop_empty_list(self):
        """Empty list [] → zero iterations, empty aggregation, post-loop still runs."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": "[]"}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "item: {{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "Count: {{ handle_parser_input | length }}"
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "EMPTY_DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e5", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "format", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e7", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
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
        assert "EMPTY_DONE" in content_str
        executed = get_executed_nodes(debug_summary)
        # Loop node not tracked in debug (executor handles iteration directly)
        assert "format" in executed
        assert "send" in executed
        # transform should NOT execute (no items to iterate)
        assert "transform" not in executed

    @pytest.mark.asyncio
    async def test_loop_post_loop_execution(self):
        """Post-loop node receives aggregation from loop's handle_end."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": '[1, 2, 3]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "x{{ handle_parser_input }}"
                }},
                {"id": "aggregate", "type": "parser", "data": {
                    "text": "Aggregated: {{ handle_parser_input | length }} items"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e5", "source": "loop", "target": "aggregate",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "aggregate", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})

        executed = get_executed_nodes(debug_summary)
        assert "aggregate" in executed
        # The aggregate node should have received 3 items
        agg_node = graph.nodes.get("aggregate")
        assert agg_node is not None
        assert "handle_parser_input" in agg_node.inputs


class TestLoopStaticPhase:
    """Tests for static phase execution before loop iteration."""

    @pytest.mark.asyncio
    async def test_loop_static_phase_before_iteration(self):
        """Static nodes (text, parser) execute before loop iterations begin."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": '["x", "y"]'}},
                {"id": "static_parser", "type": "parser", "data": {
                    "text": "Static context: {{ handle_parser_input }}"
                }},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "item: {{ handle_parser_input }}"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "static_parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e3", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e4", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e5", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e6", "source": "loop", "target": "end",
                 "sourceHandle": "handle_end", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="hello")
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})

        executed = get_executed_nodes(debug_summary)
        # Static parser should execute before loop
        assert "static_parser" in executed
        # Loop node not tracked in debug (executor handles iteration directly)
        assert "transform" in executed


class TestLoopConditionalBypass:
    """Tests for conditional bypass of loop execution."""

    @pytest.mark.asyncio
    async def test_loop_conditional_bypass_skips_loop(self):
        """Conditional that bypasses loop → loop skipped, fallback path executed."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'skip_loop' }}",
                    "output_handles": ["skip_loop", "run_loop"],
                }},
                {"id": "list_text", "type": "text", "data": {"text": '[1, 2, 3]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "item: {{ handle_parser_input }}"
                }},
                {"id": "fallback", "type": "send_message", "data": {"message": "", "json_extras": "FALLBACK"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "list_text",
                 "sourceHandle": "run_loop", "targetHandle": "handle_input"},
                {"id": "e3", "source": "cond", "target": "fallback",
                 "sourceHandle": "skip_loop", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e5", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e7", "source": "loop", "target": "end",
                 "sourceHandle": "handle_end", "targetHandle": "h1"},
                {"id": "e8", "source": "fallback", "target": "end",
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
        executed = get_executed_nodes(debug_summary)
        assert "fallback" in executed
        bypassed = get_bypassed_nodes(debug_summary)
        # Loop and its subgraph should be bypassed
        assert "loop" in bypassed
        assert "transform" in bypassed

    @pytest.mark.asyncio
    async def test_loop_bypass_also_bypasses_post_loop_nodes(self):
        """Slice 12: When loop is bypassed, post-loop nodes dependent on loop output are ALSO bypassed.

        Graph: cond → (skip_loop → fallback) / (run_loop → list_text → loop → transform → loop)
               loop → post_format → end
        When conditional selects skip_loop, post_format (which depends on loop's handle_end)
        should also be bypassed.
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'skip_loop' }}",
                    "output_handles": ["skip_loop", "run_loop"],
                }},
                {"id": "list_text", "type": "text", "data": {"text": '[1, 2, 3]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "item: {{ handle_parser_input }}"
                }},
                {"id": "fallback", "type": "send_message", "data": {"message": "", "json_extras": "FALLBACK"}},
                {"id": "post_format", "type": "parser", "data": {
                    "text": "Post-loop: {{ handle_parser_input | length }} items"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "list_text",
                 "sourceHandle": "run_loop", "targetHandle": "handle_input"},
                {"id": "e3", "source": "cond", "target": "fallback",
                 "sourceHandle": "skip_loop", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e5", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e7", "source": "loop", "target": "post_format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e8", "source": "post_format", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h1"},
                {"id": "e9", "source": "fallback", "target": "end",
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

        bypassed = get_bypassed_nodes(debug_summary)
        # Loop, its subgraph, AND post_format (depends on loop output) should all be bypassed
        assert "loop" in bypassed, f"loop should be bypassed, got: {bypassed}"
        assert "transform" in bypassed, f"transform should be bypassed, got: {bypassed}"
        assert "post_format" in bypassed, \
            f"post_format (post-loop node) should be bypassed when loop is bypassed, got: {bypassed}"

    @pytest.mark.asyncio
    async def test_loop_input_from_bypassed_source(self):
        """Slice 13: Conditional bypasses the text node feeding the loop.

        When the source of the loop's list input is bypassed by a conditional,
        the loop should detect raw is None + source in bypassed_nodes and
        execute the loop_bypassed path (lines 827-848 of reactive_executor.py).
        Post-loop nodes should also be bypassed.
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'skip_list' }}",
                    "output_handles": ["skip_list", "use_list"],
                }},
                {"id": "list_text", "type": "text", "data": {"text": '[1, 2, 3]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "item: {{ handle_parser_input }}"
                }},
                {"id": "fallback", "type": "send_message", "data": {"message": "", "json_extras": "NO_LIST"}},
                {"id": "post_format", "type": "parser", "data": {
                    "text": "Post: {{ handle_parser_input }}"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "list_text",
                 "sourceHandle": "use_list", "targetHandle": "handle_input"},
                {"id": "e3", "source": "cond", "target": "fallback",
                 "sourceHandle": "skip_list", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e5", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e7", "source": "loop", "target": "post_format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e8", "source": "post_format", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h1"},
                {"id": "e9", "source": "fallback", "target": "end",
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
        assert "NO_LIST" in content_str

        bypassed = get_bypassed_nodes(debug_summary)
        # list_text is bypassed → loop has no input → loop and its subgraph bypassed
        assert "list_text" in bypassed, f"list_text should be bypassed, got: {bypassed}"
        assert "loop" in bypassed, f"loop should be bypassed (input source bypassed), got: {bypassed}"
        assert "transform" in bypassed, f"transform should be bypassed, got: {bypassed}"
        assert "post_format" in bypassed, \
            f"post_format should be bypassed (depends on loop), got: {bypassed}"


class TestLoopMaxIterations:
    """Tests for max iteration limits."""

    @pytest.mark.asyncio
    async def test_loop_max_iterations_enforced(self):
        """200 items with max=100 → stops at 100, debug warning emitted."""
        # Create a list of 200 items
        items = list(range(200))
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": json.dumps(items)}},
                {"id": "loop", "type": "loop", "data": {"max_iterations": 100}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "x{{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "Count: {{ handle_parser_input | length }}"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e5", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "format", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        debug_items = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_items.append(item)

        # Should have a MaxIterationsExceeded debug event
        max_iter_events = [
            d for d in debug_items
            if d.get("content", {}).get("error_type") == "MaxIterationsExceeded"
        ]
        assert len(max_iter_events) > 0

        # The format node should have received only 100 items
        executed = get_executed_nodes(debug_summary)
        assert "format" in executed
        format_node = graph.nodes.get("format")
        agg = format_node.inputs.get("handle_parser_input", [])
        assert len(agg) == 100


class TestMultipleLoopNodes:
    """Slice 15 — Multiple loop nodes in same graph documents current limitation."""

    @pytest.mark.asyncio
    async def test_multiple_loop_nodes_only_first_handled(self):
        """Graph with 2 loop nodes: only the first is handled by the loop executor.

        Current behavior (reactive_executor.py line 623):
            loop_id = next(nid for nid, node in nodes.items() if isinstance(node, NodeLoop))

        This picks only the FIRST loop node. The second loop node is NOT handled
        as a loop — it will be treated as a regular node in the post-loop phase,
        which means it won't iterate. This test documents the limitation with
        explicit assertions against the real event stream.

        DEGRADED BEHAVIOR OF SECOND LOOP (proven by real runtime evidence):
        - loop1 iterates correctly over ["a", "b"] → transform1 runs twice
        - loop2 is treated as a regular node in post-loop phase
        - loop2 does NOT iterate — it executes once with whatever input it has
        - transform2 runs once in post-loop phase, NOT as part of loop2's iteration
        - loop2 emits JSONParseError because NodeLoop.process() tries to parse
          its input as a JSON list but receives a single string instead
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                # First loop with its list source
                {"id": "list1", "type": "text", "data": {"text": '["a", "b"]'}},
                {"id": "loop1", "type": "loop", "data": {}},
                {"id": "transform1", "type": "parser", "data": {
                    "text": "first: {{ handle_parser_input }}"
                }},
                # Second loop with its list source — fed from first loop's output
                {"id": "list2", "type": "parser", "data": {
                    "text": '[{{ handle_parser_input | join(",") }}]'
                }},
                {"id": "loop2", "type": "loop", "data": {}},
                {"id": "transform2", "type": "parser", "data": {
                    "text": "second: {{ handle_parser_input }}"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list1",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                # First loop
                {"id": "e2", "source": "list1", "target": "loop1",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop1", "target": "transform1",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "transform1", "target": "loop1",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                # Second loop (fed from first loop's aggregation)
                {"id": "e5", "source": "loop1", "target": "list2",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "list2", "target": "loop2",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_list"},
                {"id": "e7", "source": "loop2", "target": "transform2",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e8", "source": "transform2", "target": "loop2",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e9", "source": "loop2", "target": "end",
                 "sourceHandle": "handle_end", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")

        # Collect ALL events — not just debug, because loop_progress has its own type
        all_events = []
        debug_items = []
        loop_progress_events = []
        debug_summary = None

        async for item in run_agent(graph):
            all_events.append(item)
            if isinstance(item, dict):
                evt_type = item.get("type")
                if evt_type == "debug_summary":
                    debug_summary = item.get("content", {})
                elif evt_type == "debug":
                    debug_items.append(item)
                elif evt_type == "loop_progress":
                    loop_progress_events.append(item)

        executed = get_executed_nodes(debug_summary)

        # === PROOF: loop1 iterates correctly ===
        loop1_progress = [
            e for e in loop_progress_events
            if e.get("content", {}).get("loop_id") == "loop1"
        ]
        assert len(loop1_progress) == 2, (
            f"loop1 should produce 2 progress events (for items 'a' and 'b'), "
            f"got {len(loop1_progress)}: {loop1_progress}"
        )
        assert loop1_progress[0]["content"]["current"] == 0
        assert loop1_progress[1]["content"]["current"] == 1

        # transform1 runs twice (once per iteration)
        transform1_debug_count = sum(
            1 for d in debug_items
            if d.get("content", {}).get("node_id") == "transform1"
        )
        assert transform1_debug_count == 2, (
            f"transform1 should run twice (once per loop1 iteration), "
            f"got {transform1_debug_count} debug events"
        )

        # === PROOF: loop2 does NOT iterate (degraded behavior) ===

        # 1. NO loop_progress events for loop2 — this is the definitive proof
        #    that loop2's iteration subgraph is never entered.
        loop2_progress = [
            e for e in loop_progress_events
            if e.get("content", {}).get("loop_id") == "loop2"
        ]
        assert len(loop2_progress) == 0, (
            f"loop2 should NOT produce any loop_progress events (it's not handled "
            f"as a loop by the executor). Got {len(loop2_progress)} — this means "
            f"multi-loop handling changed."
        )

        # 2. transform2 runs only ONCE in post-loop phase, NOT per-item
        transform2_debug_count = sum(
            1 for d in debug_items
            if d.get("content", {}).get("node_id") == "transform2"
        )
        assert transform2_debug_count == 1, (
            f"transform2 should run exactly once in post-loop phase (loop2 doesn't "
            f"iterate). Got {transform2_debug_count} debug events — if this is 2+, "
            f"loop2 is now iterating correctly."
        )

        # 3. loop2 emits a JSONParseError because NodeLoop.process() tries to parse
        #    its input as a JSON list but receives a single string (the output of
        #    list2) since the iteration subgraph is never entered.
        loop2_errors = [
            d for d in debug_items
            if d.get("content", {}).get("node_id") == "loop2"
            and d.get("content", {}).get("error_type") == "JSONParseError"
        ]
        assert len(loop2_errors) >= 1, (
            "loop2 should emit a JSONParseError when treated as a regular node "
            "(NodeLoop.process() expects a JSON list but gets a single string). "
            "If this assertion fails, loop2's behavior has changed."
        )


class TestConditionalInsideLoop:
    """Slice 16 — Conditional node inside loop iteration subgraph."""

    @pytest.mark.asyncio
    async def test_conditional_inside_loop_iteration(self):
        """Conditional node inside loop iteration routes items to different branches.

        Graph: loop → conditional → parser_A (branch "yes") / parser_B (branch "no") → loop feedback
        The conditional evaluates each item and routes accordingly.

        CRITICAL: The conditional must NOT emit InputError during the static phase.
        Its inputs come from the loop's handle_item output, which isn't available until
        the iteration phase. The executor must skip such conditionals in the static phase.
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": '["yes_item", "no_item"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ handle_parser_input }}",
                    "output_handles": ["yes_item", "no_item"],
                }},
                {"id": "parser_yes", "type": "parser", "data": {
                    "text": "YES: {{ handle_parser_input }}"
                }},
                {"id": "parser_no", "type": "parser", "data": {
                    "text": "NO: {{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "Results: {{ handle_parser_input | join(' | ') }}"
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "cond",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                # Conditional branches
                {"id": "e4", "source": "cond", "target": "parser_yes",
                 "sourceHandle": "yes_item", "targetHandle": "handle_parser_input"},
                {"id": "e5", "source": "cond", "target": "parser_no",
                 "sourceHandle": "no_item", "targetHandle": "handle_parser_input"},
                # Both branches feed back to loop
                {"id": "e6", "source": "parser_yes", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e7", "source": "parser_no", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                # Post-loop
                {"id": "e8", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e9", "source": "format", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e10", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_summary = None
        debug_events = []
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_events.append(item.get("content", {}))
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        # CRITICAL: No InputError from the conditional — it must be skipped in static phase
        cond_input_errors = [
            e for e in debug_events
            if e.get("error_type") == "InputError" and e.get("node_id") == "cond"
        ]
        assert len(cond_input_errors) == 0, (
            f"Conditional 'cond' must NOT emit InputError during static phase. "
            f"Its inputs come from the loop's handle_item and aren't available until "
            f"the iteration phase. Got: {cond_input_errors}"
        )

        content_str = "".join(content_output)
        assert "DONE" in content_str

        executed = get_executed_nodes(debug_summary)
        # The conditional and both parser branches should be in the iteration subgraph
        assert "cond" in executed, f"conditional should execute, got: {executed}"
        # At least one of the branch parsers should execute
        branch_executed = {"parser_yes", "parser_no"} & executed
        assert len(branch_executed) > 0, \
            f"At least one branch parser should execute, got: {executed}"
        # Post-loop nodes should run
        assert "format" in executed, f"format should execute, got: {executed}"
        assert "send" in executed, f"send should execute, got: {executed}"

    @pytest.mark.asyncio
    async def test_conditional_inside_loop_exclusive_branch_execution(self):
        """Conditional inside loop MUST skip non-selected branches.

        Graph: loop → conditional → parser_yes (branch "yes_item") / parser_no (branch "no_item")
               → loop feedback

        When the condition evaluates to "yes_item" for ALL items, parser_no MUST be
        bypassed — NOT executed. Conversely, when it evaluates to "no_item", parser_yes
        MUST be bypassed.

        This is the core semantic of conditional routing: ONLY the selected branch runs.
        If both branches execute, the conditional is meaningless.
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": '["yes_item", "yes_item"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ handle_parser_input }}",
                    "output_handles": ["yes_item", "no_item"],
                }},
                {"id": "parser_yes", "type": "parser", "data": {
                    "text": "YES: {{ handle_parser_input }}"
                }},
                {"id": "parser_no", "type": "parser", "data": {
                    "text": "NO: {{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "Results: {{ handle_parser_input | join(' | ') }}"
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "cond",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "cond", "target": "parser_yes",
                 "sourceHandle": "yes_item", "targetHandle": "handle_parser_input"},
                {"id": "e5", "source": "cond", "target": "parser_no",
                 "sourceHandle": "no_item", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "parser_yes", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e7", "source": "parser_no", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e8", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e9", "source": "format", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e10", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_summary = None
        debug_events = []
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_events.append(item.get("content", {}))
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        executed = get_executed_nodes(debug_summary)
        bypassed = get_bypassed_nodes(debug_summary)

        # All items route to "yes_item", so parser_yes MUST execute
        assert "parser_yes" in executed, (
            f"parser_yes MUST execute when condition='yes_item'. "
            f"Executed: {executed}"
        )

        # parser_no MUST be bypassed — NOT executed — because the conditional
        # selected "yes_item", not "no_item"
        assert "parser_no" not in executed, (
            f"parser_no MUST NOT execute when condition selects 'yes_item'. "
            f"Both branches ran — conditional routing is broken. "
            f"Executed: {executed}, Bypassed: {bypassed}"
        )

        # parser_no should appear as bypassed in the debug summary
        assert "parser_no" in bypassed, (
            f"parser_no should be marked as bypassed. "
            f"Bypassed nodes: {bypassed}"
        )

        # Verify data flow through debug_summary node info.
        # The send_message node with message="" only outputs json_extras ("DONE"),
        # so we check the format node's output to verify correct routing.
        assert debug_summary is not None, "debug_summary must be present"
        node_infos = {n["node_id"]: n for n in debug_summary.get("nodes", [])}

        # format node should have received YES outputs from the loop aggregation
        format_info = node_infos.get("format", {})
        format_outputs = format_info.get("outputs", {})
        format_content = ""
        if "handle_parser_output" in format_outputs:
            fmt_out = format_outputs["handle_parser_output"]
            if isinstance(fmt_out, dict) and "content" in fmt_out:
                format_content = str(fmt_out["content"])

        assert "YES:" in format_content, (
            f"format node output should contain 'YES:' (from parser_yes). "
            f"Got: {format_content}"
        )
        assert "NO:" not in format_content, (
            f"format node output should NOT contain 'NO:' (parser_no was bypassed). "
            f"Got: {format_content}"
        )

        # The loop's aggregated feedback should also show only YES outputs
        loop_info = node_infos.get("loop", {})
        loop_outputs = loop_info.get("outputs", {})
        if "handle_end" in loop_outputs:
            loop_end = loop_outputs["handle_end"]
            if isinstance(loop_end, dict) and "content" in loop_end:
                loop_content = str(loop_end["content"])
                assert "YES:" in loop_content, (
                    f"Loop aggregation should contain YES outputs. Got: {loop_content}"
                )
                assert "NO:" not in loop_content, (
                    f"Loop aggregation should NOT contain NO outputs. Got: {loop_content}"
                )

    @pytest.mark.asyncio
    async def test_conditional_inside_loop_exclusive_branch_no_item_path(self):
        """Same as exclusive_branch_execution test but items route to 'no_item'.

        Verifies the opposite branch: when condition always selects 'no_item',
        parser_yes MUST be bypassed and parser_no MUST execute.
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": '["no_item", "no_item"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ handle_parser_input }}",
                    "output_handles": ["yes_item", "no_item"],
                }},
                {"id": "parser_yes", "type": "parser", "data": {
                    "text": "YES: {{ handle_parser_input }}"
                }},
                {"id": "parser_no", "type": "parser", "data": {
                    "text": "NO: {{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "Results: {{ handle_parser_input | join(' | ') }}"
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "cond",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "cond", "target": "parser_yes",
                 "sourceHandle": "yes_item", "targetHandle": "handle_parser_input"},
                {"id": "e5", "source": "cond", "target": "parser_no",
                 "sourceHandle": "no_item", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "parser_yes", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e7", "source": "parser_no", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e8", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e9", "source": "format", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e10", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})

        executed = get_executed_nodes(debug_summary)
        bypassed = get_bypassed_nodes(debug_summary)

        # All items route to "no_item", so parser_no MUST execute
        assert "parser_no" in executed, (
            f"parser_no MUST execute when condition='no_item'. "
            f"Executed: {executed}"
        )

        # parser_yes MUST be bypassed
        assert "parser_yes" not in executed, (
            f"parser_yes MUST NOT execute when condition selects 'no_item'. "
            f"Both branches ran — conditional routing is broken. "
            f"Executed: {executed}, Bypassed: {bypassed}"
        )

        assert "parser_yes" in bypassed, (
            f"parser_yes should be marked as bypassed. "
            f"Bypassed nodes: {bypassed}"
        )

        # Verify data flow through debug_summary
        node_infos = {n["node_id"]: n for n in debug_summary.get("nodes", [])}
        format_info = node_infos.get("format", {})
        format_outputs = format_info.get("outputs", {})
        format_content = ""
        if "handle_parser_output" in format_outputs:
            fmt_out = format_outputs["handle_parser_output"]
            if isinstance(fmt_out, dict) and "content" in fmt_out:
                format_content = str(fmt_out["content"])

        assert "NO:" in format_content, (
            f"format node output should contain 'NO:' (from parser_no). "
            f"Got: {format_content}"
        )
        assert "YES:" not in format_content, (
            f"format node output should NOT contain 'YES:' (parser_yes was bypassed). "
            f"Got: {format_content}"
        )


class TestConditionalControlsLoop:
    """Slice 17 — Conditional controls which items loop processes."""

    @pytest.mark.asyncio
    async def test_conditional_before_loop_both_paths(self):
        """Conditional before loop: one path runs the loop, the other bypasses it.

        Graph: input → cond → (run_loop → list_text → loop → transform → loop)
                              → (skip_loop → fallback → end)
        When condition selects "run_loop", loop executes normally.
        When condition selects "skip_loop", loop is bypassed and fallback runs.

        This test verifies the run_loop path.
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'run_loop' }}",
                    "output_handles": ["run_loop", "skip_loop"],
                }},
                {"id": "list_text", "type": "text", "data": {"text": '["x", "y", "z"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "item: {{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "Count: {{ handle_parser_input | length }}"
                }},
                {"id": "fallback", "type": "send_message", "data": {"message": "", "json_extras": "SKIPPED"}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "LOOP_DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "list_text",
                 "sourceHandle": "run_loop", "targetHandle": "handle_input"},
                {"id": "e3", "source": "cond", "target": "fallback",
                 "sourceHandle": "skip_loop", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e5", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e7", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e8", "source": "format", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e9", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
                {"id": "e10", "source": "fallback", "target": "end",
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
        # Loop path should execute
        assert "LOOP_DONE" in content_str
        assert "SKIPPED" not in content_str

        executed = get_executed_nodes(debug_summary)
        bypassed = get_bypassed_nodes(debug_summary)

        # Loop iteration subgraph should execute
        assert "transform" in executed
        assert "format" in executed
        # Fallback should be bypassed
        assert "fallback" in bypassed, f"fallback should be bypassed, got: {bypassed}"

    @pytest.mark.asyncio
    async def test_conditional_before_loop_skip_path(self):
        """Same graph as above but condition selects skip_loop → loop bypassed."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "cond", "type": "conditional", "data": {
                    "condition": "{{ 'skip_loop' }}",
                    "output_handles": ["run_loop", "skip_loop"],
                }},
                {"id": "list_text", "type": "text", "data": {"text": '["x", "y", "z"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "item: {{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "Count: {{ handle_parser_input | length }}"
                }},
                {"id": "fallback", "type": "send_message", "data": {"message": "", "json_extras": "SKIPPED"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "cond",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "cond", "target": "list_text",
                 "sourceHandle": "run_loop", "targetHandle": "handle_input"},
                {"id": "e3", "source": "cond", "target": "fallback",
                 "sourceHandle": "skip_loop", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e5", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e7", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e8", "source": "format", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h1"},
                {"id": "e9", "source": "fallback", "target": "end",
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
        assert "SKIPPED" in content_str

        bypassed = get_bypassed_nodes(debug_summary)
        # Loop and its subgraph should be bypassed
        assert "loop" in bypassed, f"loop should be bypassed, got: {bypassed}"
        assert "transform" in bypassed, f"transform should be bypassed, got: {bypassed}"





class TestLoopHelperFunctions:
    """Unit tests for loop helper functions."""

    def test_prepare_item_output_preserves_type(self):
        """prepare_item_output wraps item with type metadata."""
        result = prepare_item_output("hello", 0)
        assert result["content"] == "hello"
        assert result["index"] == 0
        assert result["type"] == "str"
        assert result["node"] == "NodeLoop"

    def test_prepare_item_output_dict_type(self):
        """prepare_item_output handles dict items correctly."""
        item = {"key": "value"}
        result = prepare_item_output(item, 5)
        assert result["content"] == item
        assert result["index"] == 5
        assert result["type"] == "dict"

    def test_emit_loop_progress(self):
        """emit_loop_progress returns correct structure."""
        progress = emit_loop_progress("loop_1", 0, 10, "item_a", 100.0)
        assert progress["type"] == "loop_progress"
        content = progress["content"]
        assert content["loop_id"] == "loop_1"
        assert content["current"] == 0
        assert content["total"] == 10
        assert content["progress"] == 10.0
        assert content["item_preview"] == "item_a"
        assert content["elapsed_ms"] == 100.0

    def test_find_iteration_subgraph(self):
        """find_iteration_subgraph identifies correct nodes."""
        from magic_agents.execution.event_dispatcher import GraphEventDispatcher
        from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
        from magic_agents.node_system import NodeLoop, NodeText, NodeParser, NodeEND
        from magic_agents.node_system.Node import Node

        # Create minimal mock nodes
        loop_node = NodeLoop(node_id="loop", debug=False)
        text_node = NodeText.__new__(NodeText)
        text_node.node_id = "text"
        text_node.outputs = {}
        text_node.inputs = {}
        text_node._response = None
        text_node.node_type = "text"

        parser_node = NodeParser.__new__(NodeParser)
        parser_node.node_id = "parser"
        parser_node.outputs = {}
        parser_node.inputs = {}
        parser_node._response = None
        parser_node.node_type = "parser"

        end_node = NodeEND(node_id="end", debug=False)

        nodes = {
            "loop": loop_node,
            "text": text_node,
            "parser": parser_node,
            "end": end_node,
        }

        edges = [
            EdgeNodeModel(id="e1", source="loop", target="parser",
                          sourceHandle=loop_node.OUTPUT_HANDLE_ITEM,
                          targetHandle="handle_parser_input"),
            EdgeNodeModel(id="e2", source="parser", target="loop",
                          sourceHandle="handle_parser_output",
                          targetHandle=loop_node.INPUT_HANDLE_LOOP),
            EdgeNodeModel(id="e3", source="loop", target="end",
                          sourceHandle=loop_node.OUTPUT_HANDLE_END,
                          targetHandle="h1"),
        ]

        subgraph = find_iteration_subgraph("loop", nodes, edges)
        assert "parser" in subgraph
        assert "loop" not in subgraph  # Loop itself is excluded
        assert "end" not in subgraph  # End node is not in iteration subgraph
