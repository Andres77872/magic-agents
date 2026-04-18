"""
Unit tests for GraphEventDispatcher.

Tests cover:
- Edge map building
- Output routing
- Bypass propagation
- Recursive bypass cascade
- State machine transitions
"""
import asyncio
import pytest

from magic_agents.execution.event_dispatcher import (
    GraphEventDispatcher, NodeState, NodeExecution
)
from magic_agents.execution.input_tracker import NodeInputTracker
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel


def make_mock_node(node_id: str):
    """Create a minimal mock node with required attributes."""
    class MockNode:
        def __init__(self):
            self.node_id = node_id
            self.inputs = {}
            self.outputs = {}
            self._response = None
            self.generated = ""
        def mark_bypassed(self):
            pass
    return MockNode()


def make_mock_graph(node_ids: list, edges: list):
    """Create a dict of mock nodes."""
    return {nid: make_mock_node(nid) for nid in node_ids}


class TestDispatcherEdgeMaps:
    """Test edge map building."""

    def test_dispatcher_builds_edge_maps(self):
        """Correct incoming/outgoing maps."""
        nodes = make_mock_graph(["a", "b", "c"], [])
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="out1", targetHandle="in1"),
            EdgeNodeModel(id="e2", source="a", target="c", sourceHandle="out2", targetHandle="in1"),
            EdgeNodeModel(id="e3", source="b", target="c", sourceHandle="out1", targetHandle="in2"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        # Node "a" has 2 outgoing edges
        assert len(dispatcher._outgoing.get("a", [])) == 2
        # Node "c" has 2 incoming edges
        assert len(dispatcher._incoming.get("c", [])) == 2
        # Node "a" has no incoming edges
        assert dispatcher._incoming.get("a", []) == []

    def test_dispatcher_creates_trackers_for_all_nodes(self):
        """One tracker per node."""
        nodes = make_mock_graph(["a", "b", "c"], [])
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="out", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        assert len(dispatcher._trackers) == 3
        assert "a" in dispatcher._trackers
        assert "b" in dispatcher._trackers
        assert "c" in dispatcher._trackers

    def test_dispatcher_get_source_nodes(self):
        """Nodes with no incoming edges are source nodes."""
        nodes = make_mock_graph(["a", "b", "c"], [])
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="out", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        sources = dispatcher.get_source_nodes()
        assert "a" in sources  # No incoming edges
        assert "c" in sources  # No incoming edges
        assert "b" not in sources  # Has incoming edge from "a"


class TestDispatcherOutputPropagation:
    """Test output routing to downstream nodes."""

    def test_dispatcher_propagate_output_to_targets(self):
        """Output routed to correct downstream nodes."""
        nodes = make_mock_graph(["a", "b"], [])
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="out1", targetHandle="input_x"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            await dispatcher.propagate_outputs("a", {"out1": {"content": "hello"}})
            # The input should be stored in node b's inputs
            assert nodes["b"].inputs.get("input_x") == "hello"

        asyncio.get_event_loop().run_until_complete(_test())

    def test_dispatcher_propagate_output_unwraps_prep(self):
        """Output unwraps prep() wrapper if present."""
        nodes = make_mock_graph(["a", "b"], [])
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="out1", targetHandle="input_x"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            # Simulate prep() wrapper format
            await dispatcher.propagate_outputs("a", {
                "out1": {"content": {"node": "NodeA", "content": "wrapped"}}
            })
            assert nodes["b"].inputs.get("input_x") == {"node": "NodeA", "content": "wrapped"}

        asyncio.get_event_loop().run_until_complete(_test())


class TestDispatcherBypassPropagation:
    """Test bypass propagation."""

    def test_dispatcher_propagate_conditional_bypass_active(self):
        """Active targets get input, not bypass."""
        nodes = make_mock_graph(["cond", "target_a", "target_b"], [])
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="target_a", sourceHandle="handle_yes", targetHandle="in"),
            EdgeNodeModel(id="e2", source="cond", target="target_b", sourceHandle="handle_no", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            result = await dispatcher.propagate_conditional_bypass("cond", "handle_yes")
            assert result["selected_handle"] == "handle_yes"
            assert "target_a" in result["active_targets"]
            assert "target_b" in result["bypassed_targets"]

        asyncio.get_event_loop().run_until_complete(_test())

    def test_dispatcher_propagate_conditional_bypass_inactive(self):
        """Inactive targets get bypass signal."""
        nodes = make_mock_graph(["cond", "target_a", "target_b"], [])
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="target_a", sourceHandle="handle_yes", targetHandle="in"),
            EdgeNodeModel(id="e2", source="cond", target="target_b", sourceHandle="handle_no", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            await dispatcher.propagate_conditional_bypass("cond", "handle_yes")
            # target_b should be bypassed
            tracker_b = dispatcher.get_tracker("target_b")
            assert tracker_b.is_bypassed is True
            # target_a should NOT be bypassed
            tracker_a = dispatcher.get_tracker("target_a")
            assert tracker_a.is_bypassed is False

        asyncio.get_event_loop().run_until_complete(_test())

    def test_dispatcher_recursive_bypass_cascade(self):
        """Bypass cascades when ALL inputs bypassed."""
        nodes = make_mock_graph(["cond", "mid", "end"], [])
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="mid", sourceHandle="handle_no", targetHandle="in"),
            EdgeNodeModel(id="e2", source="mid", target="end", sourceHandle="out", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            # Bypass the conditional's "no" path
            await dispatcher.propagate_conditional_bypass("cond", "handle_yes")
            # "mid" should be bypassed (all its inputs bypassed)
            assert dispatcher.get_state("mid") == NodeState.BYPASSED
            # "end" should also be bypassed (cascade)
            assert dispatcher.get_state("end") == NodeState.BYPASSED

        asyncio.get_event_loop().run_until_complete(_test())

    def test_dispatcher_recursive_bypass_stops_at_completed_node(self):
        """Bypass doesn't re-bypass already completed nodes and stops cascade."""
        nodes = make_mock_graph(["cond", "mid", "end"], [])
        edges = [
            EdgeNodeModel(id="e1", source="cond", target="mid", sourceHandle="handle_no", targetHandle="in"),
            EdgeNodeModel(id="e2", source="mid", target="end", sourceHandle="out", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            # Mark mid as already completed
            dispatcher.set_state("mid", NodeState.COMPLETED)
            # Bypass should skip completed nodes and stop cascade
            await dispatcher.propagate_conditional_bypass("cond", "handle_yes")
            # mid should remain COMPLETED
            assert dispatcher.get_state("mid") == NodeState.COMPLETED
            # end should remain PENDING (cascade stopped at mid)
            assert dispatcher.get_state("end") == NodeState.PENDING

        asyncio.get_event_loop().run_until_complete(_test())


class TestDispatcherStateTransitions:
    """Test state machine transitions."""

    def test_dispatcher_state_transitions(self):
        """PENDING → READY → EXECUTING → COMPLETED."""
        nodes = make_mock_graph(["a"], [])
        edges = []
        dispatcher = GraphEventDispatcher(nodes, edges)

        assert dispatcher.get_state("a") == NodeState.PENDING
        dispatcher.set_state("a", NodeState.EXECUTING)
        assert dispatcher.get_state("a") == NodeState.EXECUTING
        dispatcher.set_state("a", NodeState.COMPLETED)
        assert dispatcher.get_state("a") == NodeState.COMPLETED

    def test_dispatcher_reset_for_iteration(self):
        """Trackers reset, states cleared."""
        nodes = make_mock_graph(["a", "b"], [])
        edges = [
            EdgeNodeModel(id="e1", source="a", target="b", sourceHandle="out", targetHandle="in"),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            # Set some state
            await dispatcher.propagate_outputs("a", {"out": {"content": "data"}})
            dispatcher.set_state("a", NodeState.COMPLETED)

            # Reset
            dispatcher.reset_for_iteration()

            assert dispatcher.get_state("a") == NodeState.PENDING
            assert dispatcher.get_state("b") == NodeState.PENDING
            # Tracker should be reset
            tracker_a = dispatcher.get_tracker("a")
            assert tracker_a.is_ready is True  # No inputs, immediately ready

        asyncio.get_event_loop().run_until_complete(_test())

    def test_dispatcher_all_completed(self):
        """all_completed returns True when all nodes done."""
        nodes = make_mock_graph(["a", "b"], [])
        edges = []
        dispatcher = GraphEventDispatcher(nodes, edges)

        dispatcher.set_state("a", NodeState.COMPLETED)
        dispatcher.set_state("b", NodeState.BYPASSED)
        assert dispatcher.all_completed() is True

        dispatcher.set_state("a", NodeState.PENDING)
        assert dispatcher.all_completed() is False

    def test_dispatcher_get_execution_summary(self):
        """Summary reflects current state distribution."""
        nodes = make_mock_graph(["a", "b", "c"], [])
        edges = []
        dispatcher = GraphEventDispatcher(nodes, edges)

        dispatcher.set_state("a", NodeState.COMPLETED)
        dispatcher.set_state("b", NodeState.BYPASSED)
        # "c" stays PENDING

        summary = dispatcher.get_execution_summary()
        assert summary["total"] == 3
        assert summary["completed"] == 1
        assert summary["bypassed"] == 1
        assert summary["pending"] == 1

    def test_dispatcher_dispatch_input_unknown_node(self):
        """Dispatching to unknown node doesn't crash."""
        nodes = make_mock_graph(["a"], [])
        edges = []
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            # Should not raise
            await dispatcher.dispatch_input("unknown", "handle", "data")

        asyncio.get_event_loop().run_until_complete(_test())

    def test_dispatcher_dispatch_bypass_unknown_node(self):
        """Dispatching bypass to unknown node doesn't crash."""
        nodes = make_mock_graph(["a"], [])
        edges = []
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            # Should not raise
            await dispatcher.dispatch_bypass("unknown", "handle")

        asyncio.get_event_loop().run_until_complete(_test())


class TestDispatcherSourceHandleNone:
    """Regression guard: sourceHandle=None causes silent skip in propagation."""

    def test_propagate_skips_when_sourceHandle_is_none(self):
        """When edge.sourceHandle is None, output is NOT propagated (documents defect)."""
        nodes = make_mock_graph(["fetch-1", "llm-1"], [])
        edges = [
            EdgeNodeModel(
                id="e1", source="fetch-1", target="llm-1",
                sourceHandle=None,  # The defect: no sourceHandle
                targetHandle="handle-tool-definition-0"
            ),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            # Simulate fetch node producing output on its handle
            await dispatcher.propagate_outputs("fetch-1", {
                "handle_fetch_output": {"content": "FetchToolCallable_instance"}
            })
            # The target node's input should NOT be populated because sourceHandle=None
            assert nodes["llm-1"].inputs.get("handle-tool-definition-0") is None

        asyncio.get_event_loop().run_until_complete(_test())

    def test_propagate_succeeds_when_sourceHandle_is_set(self):
        """When edge.sourceHandle is correctly set, output IS propagated."""
        nodes = make_mock_graph(["fetch-1", "llm-1"], [])
        edges = [
            EdgeNodeModel(
                id="e1", source="fetch-1", target="llm-1",
                sourceHandle="handle_fetch_output",  # Correctly set
                targetHandle="handle-tool-definition-0"
            ),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            await dispatcher.propagate_outputs("fetch-1", {
                "handle_fetch_output": {"content": "FetchToolCallable_instance"}
            })
            # The target node's input SHOULD be populated
            assert nodes["llm-1"].inputs.get("handle-tool-definition-0") == "FetchToolCallable_instance"

        asyncio.get_event_loop().run_until_complete(_test())

    def test_propagate_python_exec_sourceHandle(self):
        """python_exec edge with correct sourceHandle propagates output."""
        nodes = make_mock_graph(["py-1", "llm-1"], [])
        edges = [
            EdgeNodeModel(
                id="e1", source="py-1", target="llm-1",
                sourceHandle="handle-tool-definition",
                targetHandle="handle-tool-definition-0"
            ),
        ]
        dispatcher = GraphEventDispatcher(nodes, edges)

        async def _test():
            await dispatcher.propagate_outputs("py-1", {
                "handle-tool-definition": {"content": "PythonExecutor_instance"}
            })
            assert nodes["llm-1"].inputs.get("handle-tool-definition-0") == "PythonExecutor_instance"

        asyncio.get_event_loop().run_until_complete(_test())
