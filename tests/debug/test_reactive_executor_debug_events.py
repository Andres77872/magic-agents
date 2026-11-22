"""Behavior tests for executor-owned debug observer lifecycle events."""

from datetime import datetime

import pytest

from magic_agents.debug.events import DebugEventType
from magic_agents.debug.registry import ObserverRegistry
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.Node import Node


class _ExecutionNode(Node):
    async def process(self, chat_log):
        yield self.yield_static("test output", content_type="end")


async def _execute_graph(*, debug: bool):
    callback_events = []
    streamed_events = []

    async def record(event):
        callback_events.append(event)

    graph = AgentFlowModel(
        nodes={
            "test-1": _ExecutionNode(
                node_id="test-1",
                node_type="test",
                debug=debug,
            )
        },
        edges=[],
        debug=debug,
    )
    async for event in execute_graph_reactive(graph, debug_callback=record):
        streamed_events.append(event)

    return graph, streamed_events, callback_events


def _events_for_node(events, node_id="test-1"):
    return [event for event in events if event.node_id == node_id]


@pytest.mark.asyncio
async def test_executor_reports_exact_graph_and_node_lifecycle_order():
    graph, streamed, callback_events = await _execute_graph(debug=True)

    assert [event.event_type for event in callback_events] == [
        DebugEventType.GRAPH_START,
        DebugEventType.NODE_START,
        DebugEventType.NODE_END,
        DebugEventType.GRAPH_END,
    ]
    node_events = _events_for_node(callback_events)
    assert [event.event_type for event in node_events] == [
        DebugEventType.NODE_START,
        DebugEventType.NODE_END,
    ]
    assert graph.nodes["test-1"].outputs == {
        "end": {"node": "_ExecutionNode", "content": "test output"}
    }
    assert streamed[0]["content"]["event_type"] == "GRAPH_START"


@pytest.mark.asyncio
async def test_node_lifecycle_payloads_have_real_inputs_outputs_and_timestamps():
    _, _, callback_events = await _execute_graph(debug=True)
    start, end = _events_for_node(callback_events)

    assert start.payload["inputs"] == {}
    assert datetime.fromisoformat(start.payload["start_time"])
    assert end.payload["outputs"] == {
        "end": {"node": "_ExecutionNode", "content": "test output"}
    }
    assert end.payload["duration_ms"] >= 0
    assert datetime.fromisoformat(end.payload["start_time"])
    assert datetime.fromisoformat(end.payload["end_time"])


@pytest.mark.asyncio
async def test_graph_end_callback_contains_the_completed_node_summary():
    _, _, callback_events = await _execute_graph(debug=True)
    summary = callback_events[-1]

    assert summary.event_type == DebugEventType.GRAPH_END
    assert summary.payload["total_nodes"] == 1
    assert summary.payload["executed_nodes"] == 1
    assert summary.payload["bypassed_nodes"] == 0
    assert summary.payload["failed_nodes"] == 0
    assert [node["node_id"] for node in summary.payload["nodes"]] == ["test-1"]
    assert summary.payload["nodes"][0]["was_executed"] is True


@pytest.mark.asyncio
async def test_debug_disabled_uses_null_observer_without_callback_events():
    _, streamed, callback_events = await _execute_graph(debug=False)

    assert callback_events == []
    # Graph persistence lifecycle remains part of the runtime stream and is
    # intentionally independent from the user-facing debug observer gate.
    assert streamed[0]["content"]["event_type"] == "GRAPH_START"


@pytest.mark.asyncio
async def test_node_call_drives_a_real_observer_around_process():
    callback_events = []

    async def record(event):
        callback_events.append(event)

    registry = ObserverRegistry.create(
        debug_enabled_global=True,
        graph_debug=True,
        graph_debug_config=None,
        execution_id="exec-1",
        graph_type="test",
        total_nodes=1,
        total_edges=0,
        callback=record,
    )
    node = _ExecutionNode(node_id="test-1", node_type="test", debug=True)

    yielded = [
        event
        async for event in node(
            ModelAgentRunLog(),
            observer=registry.graph_observer,
        )
    ]

    # The telemetry decorator brackets the node's routed value with content
    # lifecycle chunks; the middle event is the node's actual output.
    assert [event["type"] for event in yielded] == ["content", "end", "content"]
    assert yielded[1] == {
        "type": "end",
        "content": {"node": "_ExecutionNode", "content": "test output"},
    }
    assert [event.event_type for event in callback_events] == [
        DebugEventType.NODE_START,
        DebugEventType.NODE_END,
    ]
