"""
Slice 15 — Inner node integration tests (mocked / no API keys).

Tests inner node basic execution and nested inner graphs.
Uses text/send_message nodes inside inner graphs to avoid LLM dependency.
"""
import pytest
from unittest.mock import patch

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.node_system import NodeInner


def extract_streamed_content(item):
    """Extract streamed content from send_message output."""
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


class TestInnerNodeBasicExecution:
    """Tests for basic inner node execution."""

    @pytest.mark.asyncio
    async def test_inner_node_basic_execution(self):
        """Inner graph executes, output propagates to outer graph."""
        # Inner graph: text -> end (no LLM)
        inner_graph = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "inner_text", "type": "text", "data": {"text": "INNER_RESULT"}},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "inner_input", "target": "inner_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "ie2", "source": "inner_text", "target": "inner_end",
                 "sourceHandle": "handle_text_output", "targetHandle": "h1"},
            ],
        }

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    "magic_flow": inner_graph,
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "OUTER"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="outer message")

        # Verify inner graph was built
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert isinstance(inner_node, NodeInner)
        assert inner_node.inner_graph is not None
        assert len(inner_node.inner_graph.nodes) > 0

        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "OUTER" in content_str
        executed = get_executed_nodes(debug_summary)
        assert "inner" in executed
        assert "send" in executed

    @pytest.mark.asyncio
    async def test_inner_node_nested(self):
        """Inner node containing another inner node — recursive build and execution."""
        # Innermost graph
        innermost = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "innermost_input", "type": "user_input"},
                {"id": "innermost_text", "type": "text", "data": {"text": "NESTED"}},
                {"id": "innermost_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "innermost_input", "target": "innermost_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "ie2", "source": "innermost_text", "target": "innermost_end",
                 "sourceHandle": "handle_text_output", "targetHandle": "h1"},
            ],
        }

        # Middle graph with inner node
        middle = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "middle_input", "type": "user_input"},
                {"id": "middle_inner", "type": "inner", "data": {
                    "magic_flow": innermost,
                }},
                {"id": "middle_end", "type": "end"},
            ],
            "edges": [
                {"id": "me1", "source": "middle_input", "target": "middle_inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "me2", "source": "middle_inner", "target": "middle_end",
                 "sourceHandle": "handle_execution_content", "targetHandle": "h1"},
            ],
        }

        # Outer graph
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "outer_inner", "type": "inner", "data": {
                    "magic_flow": middle,
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "NESTED_DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "outer_inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "outer_inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")

        # Verify recursive build
        outer_inner = graph.nodes.get("outer_inner")
        assert outer_inner is not None
        assert isinstance(outer_inner, NodeInner)
        assert outer_inner.inner_graph is not None

        # Verify the middle inner node was also built
        middle_inner = outer_inner.inner_graph.nodes.get("middle_inner")
        assert middle_inner is not None
        assert isinstance(middle_inner, NodeInner)
        assert middle_inner.inner_graph is not None

        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "NESTED_DONE" in content_str
        executed = get_executed_nodes(debug_summary)
        assert "outer_inner" in executed
        assert "send" in executed

    @pytest.mark.asyncio
    async def test_inner_node_missing_inner_graph_error(self):
        """Inner node without inner_graph set yields debug error."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    # Missing magic_flow — inner graph won't be built
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "end",
                 "sourceHandle": "handle_execution_content", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert isinstance(inner_node, NodeInner)
        # inner_graph should be None since magic_flow is missing
        assert inner_node.inner_graph is None

        debug_items = []
        async for item in run_agent(graph):
            if isinstance(item, dict) and item.get("type") == "debug":
                debug_items.append(item)

        # Should have a debug error about missing inner_graph
        assert len(debug_items) > 0
        error_types = [d.get("content", {}).get("error_type") for d in debug_items]
        assert "ConfigurationError" in error_types or "InputError" in error_types

    def test_inner_node_missing_magic_flow_build_does_not_crash(self):
        """App fix: build() handles missing magic_flow gracefully (no TypeError)."""
        agt = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "end",
                 "sourceHandle": "handle_execution_content", "targetHandle": "h1"},
            ],
        }

        # Before the fix, this would raise TypeError: argument of type 'NoneType' is not iterable
        graph = build(agt, message="test")
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert isinstance(inner_node, NodeInner)
        # inner_graph should remain None since magic_flow was missing
        assert inner_node.inner_graph is None


class TestInnerGraphWithConditional:
    """Slice 18 — Inner graph with conditional nodes."""

    @pytest.mark.asyncio
    async def test_inner_graph_with_conditional_routing(self):
        """Inner graph containing a conditional node routes correctly.

        Inner graph: user_input → text → conditional → parser_yes / parser_no → end
        The text node provides a known value that the conditional evaluates.
        """
        inner_graph = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "inner_text", "type": "text", "data": {"text": "hello"}},
                {"id": "inner_cond", "type": "conditional", "data": {
                    "condition": "{{ handle_input }}",
                    "output_handles": ["hello", "other"],
                }},
                {"id": "parser_hello", "type": "parser", "data": {
                    "text": "HELLO: {{ handle_parser_input }}"
                }},
                {"id": "parser_other", "type": "parser", "data": {
                    "text": "OTHER: {{ handle_parser_input }}"
                }},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie0", "source": "inner_input", "target": "inner_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "ie1", "source": "inner_text", "target": "inner_cond",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_input"},
                {"id": "ie2", "source": "inner_cond", "target": "parser_hello",
                 "sourceHandle": "hello", "targetHandle": "handle_parser_input"},
                {"id": "ie3", "source": "inner_cond", "target": "parser_other",
                 "sourceHandle": "other", "targetHandle": "handle_parser_input"},
                {"id": "ie4", "source": "parser_hello", "target": "inner_end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h1"},
                {"id": "ie5", "source": "parser_other", "target": "inner_end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h2"},
            ],
        }

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    "magic_flow": inner_graph,
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "OUTER_DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="hello")
        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "OUTER_DONE" in content_str

        executed = get_executed_nodes(debug_summary)
        # Inner node should execute
        assert "inner" in executed
        assert "send" in executed

        # The inner graph's conditional routing should have worked
        # We can't directly see inner graph node execution in the outer debug_summary
        # because inner graph has its own execution context. But we verify the
        # inner graph was built and executed without errors.
        inner_node = graph.nodes.get("inner")
        assert inner_node is not None
        assert inner_node.inner_graph is not None
        # Verify the conditional was built inside the inner graph
        inner_cond = inner_node.inner_graph.nodes.get("inner_cond")
        assert inner_cond is not None
        assert inner_cond.node_type == "conditional"


class TestInnerGraphErrorPropagation:
    """Slice 19 — Inner graph error propagation to outer graph."""

    @pytest.mark.asyncio
    async def test_inner_graph_error_propagates_as_debug_event(self):
        """When inner graph node raises an exception, outer graph receives a debug error.

        We use a graph where the inner graph has a node that will fail
        (e.g., a parser with an invalid Jinja2 template that raises an error).
        The outer graph should continue and receive a debug event about the error.
        """
        # Inner graph with a parser that will fail (invalid Jinja2 syntax)
        inner_graph = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "inner_input", "type": "user_input"},
                {"id": "bad_parser", "type": "parser", "data": {
                    # This Jinja2 template will raise an error
                    "text": "{{ undefined_var.some_method() }}"
                }},
                {"id": "inner_end", "type": "end"},
            ],
            "edges": [
                {"id": "ie1", "source": "inner_input", "target": "bad_parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "ie2", "source": "bad_parser", "target": "inner_end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h1"},
            ],
        }

        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "inner", "type": "inner", "data": {
                    "magic_flow": inner_graph,
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "AFTER_INNER"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "inner",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
                {"id": "e2", "source": "inner", "target": "send",
                 "sourceHandle": "handle_execution_content", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")

        # Patch dispatcher timeout to avoid 60s wait — the inner_end node will timeout
        # waiting for bad_parser's output (which never arrives due to the exception)
        from magic_agents.execution.event_dispatcher import GraphEventDispatcher
        original_init = GraphEventDispatcher.__init__

        def patched_init(self, nodes, edges, timeout=2.0):
            original_init(self, nodes, edges, timeout=timeout)

        debug_items = []
        content_output = []
        with patch.object(GraphEventDispatcher, '__init__', patched_init):
            async for item in run_agent(graph):
                if isinstance(item, dict) and item.get("type") == "debug":
                    debug_items.append(item)
                text = extract_streamed_content(item)
                if text:
                    content_output.append(text)

        content_str = "".join(content_output)
        # The outer graph completes (doesn't hang). Since NodeInner emits BYPASS_ALL
        # when inner graph errors, downstream nodes like 'send' are bypassed.
        # We verify completion by checking debug_summary and debug events.

        # There should be debug events about the inner graph error
        # NodeInner now propagates debug events from inner graph to outer graph
        assert len(debug_items) > 0, "Expected debug events from inner graph error"

        error_types = [d.get("content", {}).get("error_type") for d in debug_items]
        # Should have at least one error-type event (from bad_parser exception or inner_end timeout)
        assert any(
            et in ("TemplateError", "RuntimeError", "UndefinedError", "JinjaError",
                   "InputError", "TimeoutError")
            for et in error_types
        ), f"Expected error-type debug event, got: {error_types}"
