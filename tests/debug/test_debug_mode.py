"""
Test debug mode functionality.
Demonstrates the direct runtime's typed ``debug_callback`` contract.
"""
from copy import deepcopy

import pytest
from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.debug.events import DebugEventType


class DebugCapture(list):
    """Async callback collector for typed observer events."""

    async def __call__(self, event):
        self.append(event)


def nodes_with_event(debug_events, event_type: DebugEventType) -> set[str]:
    return {
        event.node_id
        for event in debug_events
        if event.event_type == event_type
    }


def graph_summary(debug_events) -> dict:
    summaries = [
        event.payload
        for event in debug_events
        if event.event_type == DebugEventType.GRAPH_END
    ]
    assert len(summaries) == 1
    return summaries[0]


# Simple test graph with debug enabled
simple_debug_graph = {
    "type": "chat",
    "debug": True,  # Enable debug mode
    "nodes": [
        {
            "id": "user-input-1",
            "type": "user_input"
        },
        {
            "id": "text-1",
            "type": "text",
            "data": {
                "text": "Hello from text node"
            }
        },
        {
            "id": "end-1",
            "type": "end"
        }
    ],
    "edges": [
        {
            "id": "edge-1",
            "source": "user-input-1",
            "target": "text-1",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle_flow_input"
        },
        {
            "id": "edge-2",
            "source": "text-1",
            "target": "end-1",
            "sourceHandle": "handle_text_output",
            "targetHandle": "handle_flow_input"
        }
    ]
}


@pytest.mark.asyncio
async def test_debug_mode_basic():
    """Debug mode emits typed lifecycle events through debug_callback."""
    graph = build(
        agt_data=simple_debug_graph,
        message="World",
        load_chat=None,
    )
    assert not graph._validation_errors

    debug_events = DebugCapture()
    runtime_events = []
    async for result in run_agent(graph=graph, debug_callback=debug_events):
        runtime_events.append(result)

    summary = graph_summary(debug_events)
    assert summary["execution_id"]
    assert summary["total_nodes"] == len(graph.nodes)
    assert summary["executed_nodes"] == len(graph.nodes)
    assert summary["bypassed_nodes"] == 0
    assert summary["failed_nodes"] == 0
    assert {"user-input-1", "text-1", "end-1"} <= nodes_with_event(
        debug_events, DebugEventType.NODE_END
    )
    assert graph.nodes["text-1"].inputs["handle_flow_input"] == "World"
    assert graph.nodes["text-1"].response == "Hello from text node"
    assert graph.nodes["end-1"].response is not None
    assert not [
        event for event in runtime_events
        if event.get("type") == "debug"
        and event.get("content", {}).get("error_type")
    ]


# Conditional flow graph with debug
conditional_debug_graph = {
    "type": "chat",
    "debug": True,
    "nodes": [
        {
            "id": "user-input-1",
            "type": "user_input"
        },
        {
            "id": "conditional-1",
            "type": "conditional",
            "data": {
                "condition": "{{ 'handle_true' if (value|string|length) > 5 else 'handle_false' }}",
                "output_handles": ["handle_true", "handle_false"]
            }
        },
        {
            "id": "text-long",
            "type": "text",
            "data": {
                "text": "Long input received"
            }
        },
        {
            "id": "text-short",
            "type": "text",
            "data": {
                "text": "Short input received"
            }
        },
        {
            "id": "end-1",
            "type": "end"
        }
    ],
    "edges": [
        {
            "source": "user-input-1",
            "target": "conditional-1",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle_input"
        },
        {
            "source": "conditional-1",
            "target": "text-long",
            "sourceHandle": "handle_true",
            "targetHandle": "handle_flow_input"
        },
        {
            "source": "conditional-1",
            "target": "text-short",
            "sourceHandle": "handle_false",
            "targetHandle": "handle_flow_input"
        },
        {
            "source": "text-long",
            "target": "end-1",
            "sourceHandle": "handle_text_output",
            "targetHandle": "handle_flow_input"
        },
        {
            "source": "text-short",
            "target": "end-1",
            "sourceHandle": "handle_text_output",
            "targetHandle": "handle_flow_input"
        }
    ]
}


@pytest.mark.asyncio
async def test_debug_mode_conditional():
    """Test debug mode with conditional flow to verify bypassed nodes."""
    # Test with long input - use deepcopy to prevent mutation
    graph = build(
        agt_data=deepcopy(conditional_debug_graph),
        message="Long message here",
        load_chat=None,
    )
    assert not graph._validation_errors

    debug_events = DebugCapture()
    async for _ in run_agent(graph=graph, debug_callback=debug_events):
        pass

    completed = nodes_with_event(debug_events, DebugEventType.NODE_END)
    bypassed = nodes_with_event(debug_events, DebugEventType.NODE_BYPASS)
    assert "text-long" in completed
    assert "end-1" in completed
    assert "text-short" in bypassed

    summary = graph_summary(debug_events)
    assert summary["bypassed_nodes"] == 1
    bypassed_summaries = {
        node["node_id"]: node
        for node in summary["nodes"]
        if node["was_bypassed"]
    }
    assert set(bypassed_summaries) == {"text-short"}
    assert bypassed_summaries["text-short"]["outputs"] == {}
    assert bypassed_summaries["text-short"]["internal_variables"] == {}


@pytest.mark.asyncio
async def test_proper_result_handling():
    """Runtime output and observer diagnostics remain separate channels."""
    graph = build(
        agt_data=deepcopy(conditional_debug_graph),
        message="tiny",
        load_chat=None,
    )
    assert not graph._validation_errors

    debug_events = DebugCapture()
    runtime_events = []
    async for result in run_agent(graph=graph, debug_callback=debug_events):
        runtime_events.append(result)

    assert runtime_events
    assert all(isinstance(event, dict) for event in runtime_events)
    assert "text-short" in nodes_with_event(debug_events, DebugEventType.NODE_END)
    assert "text-long" in nodes_with_event(debug_events, DebugEventType.NODE_BYPASS)
    assert graph_summary(debug_events)["failed_nodes"] == 0
    assert not any(event.get("type") == "debug_summary" for event in runtime_events)
