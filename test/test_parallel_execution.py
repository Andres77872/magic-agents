"""Parallel execution verification without external services.

The concurrency test runs coordinated Node instances through the production
reactive executor, so it proves graph scheduling rather than asyncio itself.
"""
import asyncio
import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.debug.events import DebugEventType
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from magic_agents.node_system.Node import Node


def get_executed_nodes(debug_summary: dict) -> set:
    """Extract set of executed node IDs from debug summary."""
    executed = set()
    if not debug_summary:
        return executed
    for node in debug_summary.get("nodes", []):
        if node.get("was_executed"):
            executed.add(node.get("node_id"))
    return executed


class DebugCapture:
    """Collect observer events delivered through run_agent's callback API."""

    def __init__(self):
        self.events = []

    async def __call__(self, event):
        self.events.append(event)

    @property
    def summary(self) -> dict:
        for event in reversed(self.events):
            if event.event_type == DebugEventType.GRAPH_END:
                return event.payload
        return {}


class CoordinatedNode(Node):
    """A real executor node that cannot finish until its peer has started."""

    def __init__(self, node_id, own_started, peer_started, order):
        super().__init__(node_id=node_id, node_type="parser", debug=True)
        self._own_started = own_started
        self._peer_started = peer_started
        self._order = order

    async def process(self, chat_log):
        self._order.append(f"{self.node_id}_start")
        self._own_started.set()
        await asyncio.wait_for(self._peer_started.wait(), timeout=1)
        self._order.append(f"{self.node_id}_end")
        yield self.yield_static(
            {"source": self.node_id},
            content_type="handle_parser_output",
        )


class TestParallelExecution:
    """Tests for parallel execution of independent nodes."""

    @pytest.mark.asyncio
    async def test_parallel_execution_concurrent_launch(self):
        """Independent graph nodes both start before either can complete."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "branch_a", "type": "parser", "data": {"text": "A"}},
                {"id": "branch_b", "type": "parser", "data": {"text": "B"}},
                {"id": "merge", "type": "parser", "data": {
                    "text": "{{ handle_parser_input_0.source }} + {{ handle_parser_input_1.source }}"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "branch_a",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "branch_b",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "branch_a", "target": "merge",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_parser_input_0"},
                {"id": "e4", "source": "branch_b", "target": "merge",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_parser_input_1"},
                {"id": "e5", "source": "merge", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        started_a = asyncio.Event()
        started_b = asyncio.Event()
        order = []
        graph.nodes["branch_a"] = CoordinatedNode(
            "branch_a", started_a, started_b, order
        )
        graph.nodes["branch_b"] = CoordinatedNode(
            "branch_b", started_b, started_a, order
        )

        debug_capture = DebugCapture()
        async for _item in run_agent(graph, debug_callback=debug_capture):
            pass

        assert set(order[:2]) == {"branch_a_start", "branch_b_start"}
        assert set(order[2:]) == {"branch_a_end", "branch_b_end"}

        executed = get_executed_nodes(debug_capture.summary)
        assert {"branch_a", "branch_b", "merge"} <= executed
        assert graph.nodes["merge"].inputs == {
            "handle_parser_input_0": {"source": "branch_a"},
            "handle_parser_input_1": {"source": "branch_b"},
        }

    @pytest.mark.asyncio
    async def test_parallel_execution_combined_output(self):
        """Both parser outputs combined correctly at downstream node."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser_a", "type": "parser", "data": {
                    "text": '{"source": "A"}'
                }},
                {"id": "parser_b", "type": "parser", "data": {
                    "text": '{"source": "B"}'
                }},
                {"id": "merge", "type": "parser", "data": {
                    "text": "Merged: {{ handle_parser_input_0.source }} + {{ handle_parser_input_1.source }}"
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "MERGED"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser_a",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser_b",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "parser_a", "target": "merge",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_parser_input_0"},
                {"id": "e4", "source": "parser_b", "target": "merge",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_parser_input_1"},
                {"id": "e5", "source": "merge", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e6", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_capture = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_capture):
            if item.get("type") == "content" and hasattr(item.get("content"), "choices"):
                choices = item["content"].choices
                if choices and choices[0].delta.content:
                    content_output.append(choices[0].delta.content)

        content_str = "".join(content_output)
        assert "MERGED" in content_str
        executed = get_executed_nodes(debug_capture.summary)
        assert "parser_a" in executed
        assert "parser_b" in executed
        assert "merge" in executed

    @pytest.mark.asyncio
    async def test_parallel_execution_dispatcher_ready_nodes(self):
        """Dispatcher identifies multiple ready nodes after source completes."""
        agt = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser_a", "type": "parser", "data": {"text": "A"}},
                {"id": "parser_b", "type": "parser", "data": {"text": "B"}},
                {"id": "parser_c", "type": "parser", "data": {"text": "C"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser_a",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser_b",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "input", "target": "parser_c",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "parser_a", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
                {"id": "e5", "source": "parser_b", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
                {"id": "e6", "source": "parser_c", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        dispatcher = GraphEventDispatcher(graph.nodes, graph.edges)

        # After user_input completes, all 3 parsers should be ready
        # Simulate input propagation
        await dispatcher.dispatch_input("parser_a", "handle_parser_input", "test")
        await dispatcher.dispatch_input("parser_b", "handle_parser_input", "test")
        await dispatcher.dispatch_input("parser_c", "handle_parser_input", "test")

        # All three should be ready
        assert dispatcher.get_tracker("parser_a").is_ready
        assert dispatcher.get_tracker("parser_b").is_ready
        assert dispatcher.get_tracker("parser_c").is_ready

    @pytest.mark.asyncio
    async def test_parallel_execution_three_independent_branches(self):
        """Three completely independent branches all execute from a single input."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "branch_a", "type": "parser", "data": {"text": "BRANCH_A"}},
                {"id": "branch_b", "type": "parser", "data": {"text": "BRANCH_B"}},
                {"id": "branch_c", "type": "parser", "data": {"text": "BRANCH_C"}},
                {"id": "send_a", "type": "send_message", "data": {"message": "", "json_extras": "A"}},
                {"id": "send_b", "type": "send_message", "data": {"message": "", "json_extras": "B"}},
                {"id": "send_c", "type": "send_message", "data": {"message": "", "json_extras": "C"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "branch_a",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "branch_b",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "input", "target": "branch_c",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "branch_a", "target": "send_a",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e5", "source": "branch_b", "target": "send_b",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e6", "source": "branch_c", "target": "send_c",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e7", "source": "send_a", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e8", "source": "send_b", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
                {"id": "e9", "source": "send_c", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_capture = DebugCapture()
        async for item in run_agent(graph, debug_callback=debug_capture):
            text = extract_streamed_content(item)
            if text:
                content_output.append(text)

        content_str = "".join(content_output)
        assert "A" in content_str
        assert "B" in content_str
        assert "C" in content_str
        executed = get_executed_nodes(debug_capture.summary)
        assert "branch_a" in executed
        assert "branch_b" in executed
        assert "branch_c" in executed
        assert "send_a" in executed
        assert "send_b" in executed
        assert "send_c" in executed


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
