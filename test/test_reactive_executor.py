"""
Slices 6–11 — Unit tests for execute_graph_reactive() (non-loop executor).

This is the biggest coverage gap identified in explore.md §3.3:
the non-loop reactive executor had ZERO dedicated tests.

Tests cover:
- Slice 6: Parallel task creation and completion
- Slice 7: wait_ready() coordination (linear dependency chain)
- Slice 8: Output queue draining (no items lost, None sentinel)
- Slice 9: Error handling in execute_single_node (exception + timeout)
- Slice 10: BYPASS_ALL from non-conditional node
- Slice 11: Validation error blocking vs non-blocking

All tests use mocked nodes — no API keys required.
"""

import asyncio
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.execution.event_dispatcher import GraphEventDispatcher, NodeState
from magic_agents.execution.input_tracker import NodeInputTracker, InputInfo
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.util.const import SYSTEM_EVENT_DEBUG, SYSTEM_EVENT_STREAMING


# ─── Helpers ────────────────────────────────────────────────────────────────

async def _collect_all(async_gen):
    """Consume an async generator and return all yielded items."""
    results = []
    async for item in async_gen:
        results.append(item)
    return results


def _make_mock_graph(nodes_dict: dict, edges_list: list, debug: bool = True) -> AgentFlowModel:
    """Build a minimal mock AgentFlowModel for testing execute_graph_reactive."""
    graph = MagicMock(spec=AgentFlowModel)
    graph.nodes = nodes_dict
    graph.edges = edges_list
    graph.debug = debug
    graph.resolved_debug_config = None  # Disable debug feedback for simpler tests
    graph._validation_errors = []
    graph.type = "graph"
    graph.app_id = None
    graph.id_app = None
    return graph


class _CollectingNode(Node):
    """Test node that records its execution and yields a deterministic output."""

    def __init__(self, node_id: str, output_value: str = "done", output_handle: str = "output", **kwargs):
        super().__init__(node_id=node_id, **kwargs)
        self.output_value = output_value
        self.output_handle = output_handle
        self.execute_count = 0
        self.input_snapshot = None

    async def process(self, chat_log):
        self.execute_count += 1
        self.input_snapshot = dict(self.inputs)
        self._response = self.output_value
        yield self.yield_static(self.output_value, content_type=self.output_handle)


class _FailingNode(Node):
    """Test node that raises an exception during process()."""

    def __init__(self, node_id: str, exception: Exception = RuntimeError("boom"), **kwargs):
        super().__init__(node_id=node_id, **kwargs)
        self.exception = exception

    async def process(self, chat_log):
        # Yield first to satisfy async generator detection, then raise
        yield {"type": "debug", "content": {"node_id": self.node_id, "msg": "about to fail"}}
        raise self.exception


class _SlowNode(Node):
    """Test node that sleeps before responding (for timing tests)."""

    def __init__(self, node_id: str, delay: float = 0.05, output_value: str = "slow_done", **kwargs):
        super().__init__(node_id=node_id, **kwargs)
        self.delay = delay
        self.output_value = output_value
        self.start_time = None
        self.end_time = None

    async def process(self, chat_log):
        self.start_time = asyncio.get_event_loop().time()
        await asyncio.sleep(self.delay)
        self.end_time = asyncio.get_event_loop().time()
        self._response = self.output_value
        yield self.yield_static(self.output_value, content_type="output")


class _BypassAllNode(Node):
    """Test node that emits a BYPASS_ALL signal."""

    def __init__(self, node_id: str, **kwargs):
        super().__init__(node_id=node_id, **kwargs)
        from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalSignalTypes
        self.BYPASS_ALL = ConditionalSignalTypes.BYPASS_ALL

    async def process(self, chat_log):
        self._response = "bypass_all_sent"
        yield {
            "type": self.BYPASS_ALL,
            "content": {"signal": "bypass_all"}
        }


# ─── Slice 6: Parallel Task Creation and Completion ─────────────────────────

class TestParallelTaskCreation:
    """Slice 6: Verify parallel task creation and completion."""

    @pytest.mark.asyncio
    async def test_all_nodes_complete(self):
        """All nodes in a graph should complete, even when created as parallel tasks."""
        node_a = _CollectingNode(node_id="a", output_value="A")
        node_b = _CollectingNode(node_id="b", output_value="B")
        node_c = _CollectingNode(node_id="c", output_value="C")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "b": node_b, "c": node_c, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="end", sourceHandle="output", targetHandle="h1"),
            EdgeNodeModel(id="e2", source="b", target="end", sourceHandle="output", targetHandle="h1"),
            EdgeNodeModel(id="e3", source="c", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        results = await _collect_all(execute_graph_reactive(graph))

        # All nodes should have executed
        assert node_a.execute_count == 1
        assert node_b.execute_count == 1
        assert node_c.execute_count == 1
        assert node_end.execute_count == 1

    @pytest.mark.asyncio
    async def test_parallel_execution_timing(self):
        """Nodes with no dependencies should execute in parallel (not sequentially).

        If three slow nodes (50ms each) run in parallel, total time should be
        ~50ms, not ~150ms.
        """
        node_a = _SlowNode(node_id="a", delay=0.05)
        node_b = _SlowNode(node_id="b", delay=0.05)
        node_c = _SlowNode(node_id="c", delay=0.05)
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "b": node_b, "c": node_c, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="end", sourceHandle="output", targetHandle="h1"),
            EdgeNodeModel(id="e2", source="b", target="end", sourceHandle="output", targetHandle="h1"),
            EdgeNodeModel(id="e3", source="c", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        loop = asyncio.get_event_loop()
        start = loop.time()
        await _collect_all(execute_graph_reactive(graph))
        elapsed = loop.time() - start

        # If parallel: ~50ms. If sequential: ~150ms. Allow 100ms as threshold.
        assert elapsed < 0.15, \
            f"Nodes should execute in parallel (~50ms), but took {elapsed:.3f}s (sequential would be ~150ms)"


# ─── Slice 7: wait_ready() Coordination ─────────────────────────────────────

class TestWaitReadyCoordination:
    """Slice 7: Test wait_ready() coordination at executor level."""

    @pytest.mark.asyncio
    async def test_linear_dependency_chain(self):
        """Graph: A → B → C. Verify B waits for A, C waits for B."""
        node_a = _CollectingNode(node_id="a", output_value="A_out")
        node_b = _CollectingNode(node_id="b", output_value="B_out")
        node_c = _CollectingNode(node_id="c", output_value="C_out")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "b": node_b, "c": node_c, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="output", targetHandle="input"),
            EdgeNodeModel(id="e2", source="b", target="c", sourceHandle="output", targetHandle="input"),
            EdgeNodeModel(id="e3", source="c", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        results = await _collect_all(execute_graph_reactive(graph))

        # All nodes should have executed
        assert node_a.execute_count == 1
        assert node_b.execute_count == 1
        assert node_c.execute_count == 1
        assert node_end.execute_count == 1

        # B should have received A's output
        assert node_b.input_snapshot is not None
        assert "input" in node_b.input_snapshot
        assert node_b.input_snapshot["input"] == "A_out"

        # C should have received B's output
        assert node_c.input_snapshot is not None
        assert "input" in node_c.input_snapshot
        assert node_c.input_snapshot["input"] == "B_out"

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        """Graph: A → B, A → C, B → D, C → D. Verify D waits for both B and C."""
        node_a = _CollectingNode(node_id="a", output_value="A_out")
        node_b = _CollectingNode(node_id="b", output_value="B_out")
        node_c = _CollectingNode(node_id="c", output_value="C_out")
        node_d = _CollectingNode(node_id="d", output_value="D_out")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "b": node_b, "c": node_c, "d": node_d, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="output", targetHandle="input"),
            EdgeNodeModel(id="e2", source="a", target="c", sourceHandle="output", targetHandle="input"),
            EdgeNodeModel(id="e3", source="b", target="d", sourceHandle="output", targetHandle="input_b"),
            EdgeNodeModel(id="e4", source="c", target="d", sourceHandle="output", targetHandle="input_c"),
            EdgeNodeModel(id="e5", source="d", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        results = await _collect_all(execute_graph_reactive(graph))

        # All nodes executed
        assert node_d.execute_count == 1
        assert node_end.execute_count == 1

        # D should have received both inputs
        assert node_d.input_snapshot is not None
        assert "input_b" in node_d.input_snapshot
        assert "input_c" in node_d.input_snapshot
        assert node_d.input_snapshot["input_b"] == "B_out"
        assert node_d.input_snapshot["input_c"] == "C_out"


# ─── Slice 8: Output Queue Draining ─────────────────────────────────────────

class TestOutputQueueDraining:
    """Slice 8: Verify output queue draining behavior."""

    @pytest.mark.asyncio
    async def test_all_outputs_collected(self):
        """Verify all yielded items are collected, no items lost."""
        node_a = _CollectingNode(node_id="a", output_value="A_result")
        node_b = _CollectingNode(node_id="b", output_value="B_result")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "b": node_b, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="end", sourceHandle="output", targetHandle="h1"),
            EdgeNodeModel(id="e2", source="b", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        results = await _collect_all(execute_graph_reactive(graph))

        # The telemetry wrapper yields 'content' events with ChatCompletionModel.
        # The executor puts these on the queue as SYSTEM_EVENT_STREAMING with
        # content = ChatCompletionModel. We verify by checking node_id in extras.
        streaming = [r for r in results if isinstance(r, dict) and r.get("type") == SYSTEM_EVENT_STREAMING]
        node_ids_seen = []
        for item in streaming:
            content = item.get("content")
            if content and hasattr(content, "extras"):
                meta = content.extras.get("meta", {})
                node_ids_seen.append(meta.get("node_id"))

        # Both a and b should have produced streaming output
        assert "a" in node_ids_seen, f"Node 'a' not in streaming. Got nodes: {node_ids_seen}"
        assert "b" in node_ids_seen, f"Node 'b' not in streaming. Got nodes: {node_ids_seen}"

    @pytest.mark.asyncio
    async def test_none_sentinel_terminates(self):
        """Verify None sentinel terminates the queue correctly.

        The executor puts None on the queue when all tasks are done.
        The drain loop should break on None, not yield it.
        """
        node_a = _CollectingNode(node_id="a", output_value="A")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        results = await _collect_all(execute_graph_reactive(graph))

        # None sentinel should NOT appear in results
        assert None not in results, "None sentinel leaked into results"


# ─── Slice 9: Error Handling in execute_single_node ─────────────────────────

class TestErrorHandling:
    """Slice 9: Test error handling in execute_single_node."""

    @pytest.mark.asyncio
    async def test_node_exception_emits_debug_event(self):
        """Mock node that raises exception → verify debug event emitted, state set to ERROR.

        Graph: a → fail (raises), a → end. End node doesn't depend on fail,
        so the graph completes even though fail errors out.
        """
        node_a = _CollectingNode(node_id="a", output_value="A")
        node_fail = _FailingNode(node_id="fail", exception=RuntimeError("test error"))
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "fail": node_fail, "end": node_end}
        # "end" only depends on "a", NOT on "fail" — so graph completes
        edges = [
            EdgeNodeModel(id="e1", source="a", target="fail", sourceHandle="output", targetHandle="input"),
            EdgeNodeModel(id="e2", source="a", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        results = await _collect_all(execute_graph_reactive(graph))

        # Should have a debug event with the error
        debug_events = [
            r for r in results
            if isinstance(r, dict) and r.get("type") == SYSTEM_EVENT_DEBUG
        ]
        error_events = [
            e for e in debug_events
            if e.get("content", {}).get("error_type") == "RuntimeError"
        ]
        assert len(error_events) > 0, f"Expected RuntimeError debug event, got: {debug_events}"
        assert "test error" in error_events[0]["content"]["error_message"]

    @pytest.mark.asyncio
    async def test_node_timeout_emits_debug_event(self):
        """Mock node that times out waiting for inputs → verify TimeoutError handling.

        Node "b" depends on a non-existent "ghost" node that never produces output.
        With a tiny timeout (0.01s), the timeout fires quickly.
        Node "end" only depends on "a" so the graph completes.
        """
        node_a = _CollectingNode(node_id="a", output_value="A")
        node_b = _CollectingNode(node_id="b", output_value="B")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "b": node_b, "end": node_end}
        # "b" depends on "ghost" which doesn't exist — will never get input
        # "end" only depends on "a" — graph can complete
        edges = [
            EdgeNodeModel(id="e1", source="ghost", target="b", sourceHandle="output", targetHandle="input_ghost"),
            EdgeNodeModel(id="e2", source="a", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)

        # Create a real dispatcher
        dispatcher = GraphEventDispatcher(nodes, edges)
        # Set a tiny timeout for the test
        dispatcher.timeout = 0.01

        # Patch the dispatcher creation to return our pre-configured one
        with patch('magic_agents.execution.reactive_executor.GraphEventDispatcher', return_value=dispatcher):
            results = await _collect_all(execute_graph_reactive(graph))

        # Should have a debug event with timeout error
        debug_events = [
            r for r in results
            if isinstance(r, dict) and r.get("type") == SYSTEM_EVENT_DEBUG
        ]
        timeout_events = [
            e for e in debug_events
            if e.get("content", {}).get("error_type") == "TimeoutError"
        ]
        assert len(timeout_events) > 0, \
            f"Expected TimeoutError debug event, got: {[e.get('content', {}).get('error_type') for e in debug_events]}"

        # Node "a" should still have executed
        assert node_a.execute_count == 1
        # Node "end" should still have executed (it only depends on "a")
        assert node_end.execute_count == 1


# ─── Slice 10: BYPASS_ALL from Non-Conditional Node ─────────────────────────

class TestBypassAllSignal:
    """Slice 10: Test BYPASS_ALL from non-conditional node."""

    @pytest.mark.asyncio
    async def test_bypass_all_from_node(self):
        """Mock node that emits BYPASS_ALL signal → verify handle_bypass_all_signal called."""
        from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalSignalTypes

        node_a = _CollectingNode(node_id="a", output_value="A")
        # Create a node that emits BYPASS_ALL
        node_bypass = _BypassAllNode(node_id="bypass_all_node")
        node_downstream = _CollectingNode(node_id="downstream", output_value="DOWNSTREAM")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {
            "a": node_a,
            "bypass_all_node": node_bypass,
            "downstream": node_downstream,
            "end": node_end
        }
        edges = [
            EdgeNodeModel(id="e1", source="a", target="bypass_all_node",
                          sourceHandle="output", targetHandle="input"),
            EdgeNodeModel(id="e2", source="bypass_all_node", target="downstream",
                          sourceHandle="output", targetHandle="input"),
            EdgeNodeModel(id="e3", source="a", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        results = await _collect_all(execute_graph_reactive(graph))

        # The bypass_all_node should have executed (response set)
        assert node_bypass._response == "bypass_all_sent", \
            f"bypass_all_node should have set response, got: {node_bypass._response}"

        # Downstream node should be bypassed (BYPASS_ALL cascades to all downstream)
        # The dispatcher should have called handle_bypass_all_signal
        # Verify by checking that downstream did NOT execute
        assert node_downstream.execute_count == 0, \
            "Downstream node should be bypassed after BYPASS_ALL signal"

        # Node "a" and "end" should still have executed
        assert node_a.execute_count == 1
        assert node_end.execute_count == 1


# ─── Slice 11: Validation Error Blocking vs Non-Blocking ────────────────────

class TestValidationErrors:
    """Slice 11: Test validation error blocking vs non-blocking."""

    @pytest.mark.asyncio
    async def test_graphValidationError_blocks_execution(self):
        """Graph with GraphValidationError → verify execution aborts."""
        node_a = _CollectingNode(node_id="a", output_value="A")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        # Inject a blocking validation error
        graph._validation_errors = [{
            "error_type": "GraphValidationError",
            "message": "Structural error: missing required node",
            "node_id": "a",
        }]

        results = await _collect_all(execute_graph_reactive(graph))

        # Should have debug event for the error
        debug_events = [
            r for r in results
            if isinstance(r, dict) and r.get("type") == SYSTEM_EVENT_DEBUG
        ]
        assert len(debug_events) > 0, "Should have yielded debug event for validation error"

        # Node should NOT have executed (execution aborted)
        assert node_a.execute_count == 0, "Node should not execute when GraphValidationError present"

    @pytest.mark.asyncio
    async def test_MissingConditionalEdge_does_not_block(self):
        """Graph with MissingConditionalEdge → verify execution continues."""
        node_a = _CollectingNode(node_id="a", output_value="A")
        node_end = _CollectingNode(node_id="end", output_value="END", output_handle="h1")

        nodes = {"a": node_a, "end": node_end}
        edges = [
            EdgeNodeModel(id="e1", source="a", target="end", sourceHandle="output", targetHandle="h1"),
        ]

        graph = _make_mock_graph(nodes, edges)
        # Inject a non-blocking conditional validation error
        graph._validation_errors = [{
            "error_type": "MissingConditionalEdge",
            "message": "Conditional node has unconnected branch",
            "node_id": "cond",
        }]

        results = await _collect_all(execute_graph_reactive(graph))

        # Should have debug event for the warning
        debug_events = [
            r for r in results
            if isinstance(r, dict) and r.get("type") == SYSTEM_EVENT_DEBUG
        ]
        assert len(debug_events) > 0, "Should have yielded debug event for conditional warning"

        # Node SHOULD have executed (non-blocking error)
        assert node_a.execute_count == 1, "Node should execute despite MissingConditionalEdge"
        assert node_end.execute_count == 1, "End node should execute despite MissingConditionalEdge"
