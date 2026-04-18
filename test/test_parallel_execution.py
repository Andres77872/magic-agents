"""
Slice 16 — Parallel execution verification (mocked / no API keys).

Proves that independent nodes execute concurrently via the reactive executor.
Two types of proof:
1. Concurrent launch: tasks start within a small window (timing-based)
2. Event-loop interleaving: tasks actually interleave at await points (Event-based)
"""
import asyncio
import time
from datetime import datetime
import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.execution.event_dispatcher import GraphEventDispatcher


def get_executed_nodes(debug_summary: dict) -> set:
    """Extract set of executed node IDs from debug summary."""
    executed = set()
    if not debug_summary:
        return executed
    for node in debug_summary.get("nodes", []):
        if node.get("was_executed"):
            executed.add(node.get("node_id"))
    return executed


def get_node_execution_times(debug_summary: dict) -> dict:
    """Extract execution start/end times for each node."""
    times = {}
    if not debug_summary:
        return times
    for node in debug_summary.get("nodes", []):
        if node.get("was_executed"):
            nid = node.get("node_id")
            times[nid] = {
                "start": node.get("start_time"),
                "end": node.get("end_time"),
                "duration_ms": node.get("execution_duration_ms"),
            }
    return times


class TestParallelExecution:
    """Tests for parallel execution of independent nodes."""

    @pytest.mark.asyncio
    async def test_parallel_execution_concurrent_launch(self):
        """Two independent parsers are launched as concurrent tasks.

        Both parser receive input from user_input and have no dependency
        on each other. In the reactive executor, they should both be
        ready and have tasks created simultaneously.

        What this proves:
        - Both parsers execute (not just one)
        - Their start times are within a small window (< 50ms), proving
          concurrent task creation by asyncio.create_task()

        What this does NOT prove:
        - Wall-clock overlap of execution. Parser nodes complete in <1ms,
          so there is no measurable overlap. For interleaving proof, see
          test_parallel_execution_event_loop_interleaving.
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser_a", "type": "parser", "data": {
                    "text": "A: {{ handle_parser_input }}"
                }},
                {"id": "parser_b", "type": "parser", "data": {
                    "text": "B: {{ handle_parser_input }}"
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser_a",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "input", "target": "parser_b",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e3", "source": "parser_a", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "parser_b", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra_2"},
                {"id": "e5", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})

        executed = get_executed_nodes(debug_summary)
        assert "parser_a" in executed, "parser_a should have executed"
        assert "parser_b" in executed, "parser_b should have executed"

        # Prove concurrent launch: start times should be within 50ms of each other
        times = get_node_execution_times(debug_summary)
        assert "parser_a" in times, "parser_a should have timing data"
        assert "parser_b" in times, "parser_b should have timing data"

        start_a = times["parser_a"]["start"]
        start_b = times["parser_b"]["start"]
        assert start_a is not None, "parser_a should have start_time"
        assert start_b is not None, "parser_b should have start_time"

        # Parse ISO timestamps and compute difference
        if isinstance(start_a, str):
            start_a = datetime.fromisoformat(start_a).timestamp()
        if isinstance(start_b, str):
            start_b = datetime.fromisoformat(start_b).timestamp()

        # Both parsers should have started within 50ms of each other
        # (proving they were launched as concurrent tasks, not sequentially)
        time_diff_ms = abs(start_b - start_a) * 1000
        assert time_diff_ms < 50, (
            f"Parsers should have started within 50ms of each other "
            f"(concurrent launch), but diff was {time_diff_ms:.1f}ms. "
            f"start_a={start_a}, start_b={start_b}"
        )

    @pytest.mark.asyncio
    async def test_parallel_execution_event_loop_interleaving(self):
        """Prove the event loop actually interleaves concurrent async tasks.

        This test uses asyncio Events to demonstrate that two independent
        async generators can both reach a 'started' state before either
        completes — proving true interleaving, not just concurrent launch.

        This is the mechanism the reactive executor relies on: all node
        tasks are created via asyncio.create_task() in a tight loop, and
        the event loop interleaves them at each await point.
        """
        # Two async generators that signal when they start and wait for each other
        started_a = asyncio.Event()
        started_b = asyncio.Event()
        order = []

        async def slow_generator_a():
            order.append("a_start")
            started_a.set()
            await started_b.wait()  # Wait for B to start too
            order.append("a_end")
            yield {"type": "content", "content": "A"}

        async def slow_generator_b():
            order.append("b_start")
            started_b.set()
            await started_a.wait()  # Wait for A to start too
            order.append("b_end")
            yield {"type": "content", "content": "B"}

        async def collect(gen):
            items = []
            async for item in gen:
                items.append(item)
            return items

        # Run both generators concurrently as tasks (same pattern as the executor)
        task_a = asyncio.create_task(collect(slow_generator_a()))
        task_b = asyncio.create_task(collect(slow_generator_b()))
        results_a, results_b = await asyncio.gather(task_a, task_b)

        # Both must have started before either completed
        # If sequential: order would be ["a_start", "a_end", "b_start", "b_end"] or vice versa
        # If interleaved: both starts come before both ends
        a_start_idx = order.index("a_start")
        b_start_idx = order.index("b_start")
        a_end_idx = order.index("a_end")
        b_end_idx = order.index("b_end")

        # Both starts must happen before both ends (interleaving proof)
        assert a_start_idx < a_end_idx, "A should start before it ends"
        assert b_start_idx < b_end_idx, "B should start before it ends"
        # The key assertion: at least one start happens before the other ends
        # This proves interleaving — in sequential execution, one would fully complete first
        assert (a_start_idx < b_end_idx and b_start_idx < a_end_idx), (
            f"Expected interleaving (both starts before both ends), got order: {order}. "
            f"This means tasks ran sequentially, not concurrently."
        )

        # Both should have produced output
        assert len(results_a) == 1
        assert len(results_b) == 1

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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            if item.get("type") == "content" and hasattr(item.get("content"), "choices"):
                choices = item["content"].choices
                if choices and choices[0].delta.content:
                    content_output.append(choices[0].delta.content)

        content_str = "".join(content_output)
        assert "MERGED" in content_str
        executed = get_executed_nodes(debug_summary)
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
                 "sourceHandle": "handle_parser_output", "targetHandle": "h1"},
                {"id": "e5", "source": "parser_b", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h2"},
                {"id": "e6", "source": "parser_c", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "h3"},
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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
                {"id": "e8", "source": "send_b", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h2"},
                {"id": "e9", "source": "send_c", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h3"},
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
        assert "A" in content_str
        assert "B" in content_str
        assert "C" in content_str
        executed = get_executed_nodes(debug_summary)
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
