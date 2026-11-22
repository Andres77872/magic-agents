"""
Test suite for NodeLoop refactored implementation.

These tests verify the fixes for critical issues:
1. Aggregation contains actual results (not nulls)
2. All iteration subgraph nodes properly reset
3. Item types preserved through iteration
4. Iteration limits enforced
5. Progress events emitted
"""

import pytest
import json

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.debug.events import DebugEventType
from magic_agents.execution.reactive_executor import (
    find_iteration_subgraph,
    topological_sort_iteration,
    emit_loop_progress,
)
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.node_system.NodeLoop import NodeLoop


# ─── Helpers ────────────────────────────────────────────────────────────────

def _get_executed_nodes(debug_summary: dict) -> set:
    """Extract set of executed node IDs from debug summary."""
    if not debug_summary:
        return set()
    return {n["node_id"] for n in debug_summary.get("nodes", []) if n.get("was_executed")}


def _collect_all(async_gen):
    """Consume an async generator and return all yielded items."""
    async def _collect():
        results = []
        async for item in async_gen:
            results.append(item)
        return results
    return _collect()


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


class TestFindIterationSubgraph:
    """Tests for the iteration subgraph detection."""

    def test_simple_linear_subgraph(self):
        """Test finding nodes in a simple linear iteration."""
        # Loop -> Node A -> Loop (feedback)
        loop = NodeLoop(node_id="loop", node_type="loop")
        nodes = {
            "loop": loop,
            "node_a": object(),
        }
        edges = [
            EdgeNodeModel(id="e1", source="loop", target="node_a",
                          sourceHandle=loop.OUTPUT_HANDLE_ITEM, targetHandle="input"),
            EdgeNodeModel(id="e2", source="node_a", target="loop",
                          sourceHandle="output", targetHandle=loop.INPUT_HANDLE_LOOP),
        ]

        result = find_iteration_subgraph('loop', nodes, edges)
        assert 'node_a' in result
        assert 'loop' not in result

    def test_multi_node_subgraph(self):
        """Test finding nodes in a multi-node iteration chain."""
        # Loop -> Node A -> Node B -> Loop
        loop = NodeLoop(node_id="loop", node_type="loop")
        nodes = {
            "loop": loop,
            "node_a": object(),
            "node_b": object(),
        }
        edges = [
            EdgeNodeModel(id="e1", source="loop", target="node_a",
                          sourceHandle=loop.OUTPUT_HANDLE_ITEM, targetHandle="input"),
            EdgeNodeModel(id="e2", source="node_a", target="node_b",
                          sourceHandle="output", targetHandle="input"),
            EdgeNodeModel(id="e3", source="node_b", target="loop",
                          sourceHandle="output", targetHandle=loop.INPUT_HANDLE_LOOP),
        ]

        result = find_iteration_subgraph('loop', nodes, edges)
        assert 'node_a' in result
        assert 'node_b' in result
        assert 'loop' not in result

    def test_excludes_end_nodes(self):
        """Test that nodes after handle_end are not included."""
        loop = NodeLoop(node_id="loop", node_type="loop")
        nodes = {
            "loop": loop,
            "node_a": object(),
            "node_end": object(),
        }
        edges = [
            EdgeNodeModel(id="e1", source="loop", target="node_a",
                          sourceHandle=loop.OUTPUT_HANDLE_ITEM, targetHandle="input"),
            EdgeNodeModel(id="e2", source="node_a", target="loop",
                          sourceHandle="output", targetHandle=loop.INPUT_HANDLE_LOOP),
            EdgeNodeModel(id="e3", source="loop", target="node_end",
                          sourceHandle=loop.OUTPUT_HANDLE_END, targetHandle="input"),
        ]

        result = find_iteration_subgraph('loop', nodes, edges)
        assert 'node_a' in result
        assert 'node_end' not in result


class TestTopologicalSortIteration:
    """Tests for iteration node sorting."""

    def test_linear_order(self):
        """Test sorting a linear chain of nodes."""
        iteration_nodes = {'a', 'b', 'c'}
        item_edges = [EdgeNodeModel(id="item", source="loop", target="a",
                                    sourceHandle="out", targetHandle="in")]
        loop_back_edges = [
            EdgeNodeModel(id="ab", source="a", target="b", sourceHandle="out", targetHandle="in"),
            EdgeNodeModel(id="bc", source="b", target="c", sourceHandle="out", targetHandle="in"),
            EdgeNodeModel(id="back", source="c", target="loop", sourceHandle="out", targetHandle="in"),
        ]

        result = topological_sort_iteration(iteration_nodes, item_edges, loop_back_edges)

        # Verify order: a before b before c
        assert result.index('a') < result.index('b')
        assert result.index('b') < result.index('c')

    def test_parallel_nodes(self):
        """Test nodes with no dependencies between them."""
        # Both a and b receive from loop, both feed to c
        iteration_nodes = {'a', 'b', 'c'}
        item_edges = [
            EdgeNodeModel(id="item-a", source="loop", target="a", sourceHandle="out", targetHandle="in"),
            EdgeNodeModel(id="item-b", source="loop", target="b", sourceHandle="out", targetHandle="in"),
        ]
        loop_back_edges = [
            EdgeNodeModel(id="ac", source="a", target="c", sourceHandle="out", targetHandle="in"),
            EdgeNodeModel(id="bc", source="b", target="c", sourceHandle="out", targetHandle="in"),
            EdgeNodeModel(id="back", source="c", target="loop", sourceHandle="out", targetHandle="in"),
        ]

        result = topological_sort_iteration(iteration_nodes, item_edges, loop_back_edges)

        # a and b can be in any order, but both before c
        assert result.index('c') > result.index('a')
        assert result.index('c') > result.index('b')


class TestProgressEvents:
    """Tests for loop progress events."""
    
    def test_progress_event_structure(self):
        """Verify progress event has expected fields."""
        event = emit_loop_progress(
            loop_id='loop1',
            current_index=5,
            total_items=10,
            item='test_item',
            elapsed_ms=500.0
        )
        
        assert event['type'] == 'loop_progress'
        content = event['content']
        assert content['loop_id'] == 'loop1'
        assert content['current'] == 5
        assert content['total'] == 10
        assert content['progress'] == 60.0  # (5+1)/10 * 100
        assert 'estimated_remaining_ms' in content


class TestAggregationResults:
    """Integration-like tests for aggregation behavior."""

    @pytest.mark.asyncio
    async def test_aggregation_order_preserved(self):
        """Slice 4: Verify results are in same order as input items.

        Same graph as Slice 3, but verify order matches input.
        """
        input_items = ["first", "second", "third", "fourth"]
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": json.dumps(input_items)}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "{{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "{{ handle_parser_input | join(' | ') }}"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e5", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "format", "target": "end",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        await _collect_all(run_agent(graph))

        # Get aggregation from format node
        format_node = graph.nodes.get("format")
        agg_input = format_node.inputs.get("handle_parser_input")
        assert agg_input is not None, "Format node should have received aggregation"
        assert len(agg_input) == len(input_items), \
            f"Should have {len(input_items)} items, got {len(agg_input)}"

        # Order must match input
        for i, (expected, actual) in enumerate(zip(input_items, agg_input)):
            assert actual == expected, \
                f"Order mismatch at index {i}: expected '{expected}', got '{actual}'"


class TestNodeReset:
    """Tests for proper node reset between iterations."""

    @pytest.mark.asyncio
    async def test_all_iteration_nodes_reset(self):
        """Slice 5: Verify all nodes in iteration subgraph are reset between iterations.

        Build a multi-node iteration subgraph (transform → enrich → loop feedback).
        Verify that stale outputs from iteration N don't leak into iteration N+1.
        Specifically test that each iteration starts fresh (explore.md §4.2 risk).
        """
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": '["A", "B", "C"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "T:{{ handle_parser_input }}"
                }},
                {"id": "enrich", "type": "parser", "data": {
                    "text": "E:{{ handle_parser_input }}"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "transform", "target": "enrich",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_parser_input"},
                {"id": "e5", "source": "enrich", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e6", "source": "loop", "target": "end",
                 "sourceHandle": "handle_end", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        debug_capture = DebugCapture()
        await _collect_all(run_agent(graph, debug_callback=debug_capture))

        # Both iteration nodes should have executed
        executed = _get_executed_nodes(debug_capture.summary)
        assert "transform" in executed, f"transform should have executed, got: {executed}"
        assert "enrich" in executed, f"enrich should have executed, got: {executed}"

        # One fresh enrich(transform(item)) result per iteration, never a stale value.
        assert graph.nodes["end"].inputs["handle_flow_input"] == ["E:T:A", "E:T:B", "E:T:C"]
