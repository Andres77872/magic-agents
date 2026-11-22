"""Integration tests for Node -> DefaultObserver -> SSE queue envelopes."""

import asyncio
import json
from datetime import datetime

import pytest

from magic_agents.debug.config import DebugConfig
from magic_agents.debug.default_observer import DefaultObserver
from magic_agents.debug.emitter import EmitterRegistry, QueueEmitter
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.Node import Node


class _SSEFlowNode(Node):
    async def process(self, chat_log):
        yield self.yield_static("partial", content_type="streaming")
        yield self.yield_static("final output", content_type="end")


async def _run_node_through_sse_queue():
    queue = asyncio.Queue()
    emitters = EmitterRegistry().register(QueueEmitter(queue))
    observer = DefaultObserver(
        execution_id="sse-exec-1",
        graph_type="test",
        config=DebugConfig(enabled=True),
        emitter_registry=emitters,
        total_nodes=1,
    )
    node = _SSEFlowNode(node_id="sse-test-1", node_type="test", debug=True)

    yielded = [
        event
        async for event in node(ModelAgentRunLog(), observer=observer)
    ]
    envelopes = []
    while not queue.empty():
        envelopes.append(queue.get_nowait())
    return yielded, envelopes


@pytest.mark.asyncio
async def test_full_node_observer_chain_emits_start_then_end():
    yielded, envelopes = await _run_node_through_sse_queue()

    assert [event["type"] for event in yielded] == [
        "content",
        "streaming",
        "end",
        "content",
    ]
    assert [event["event_type"] for event in envelopes] == [
        "node_start",
        "node_end",
    ]
    assert [event["content"]["node_id"] for event in envelopes] == [
        "sse-test-1",
        "sse-test-1",
    ]
    assert envelopes[0]["content"]["was_executed"] is False
    assert envelopes[1]["content"]["was_executed"] is True


@pytest.mark.asyncio
async def test_sse_envelopes_are_json_round_trip_safe():
    _, envelopes = await _run_node_through_sse_queue()

    encoded = [json.dumps(event) for event in envelopes]
    decoded = [json.loads(event) for event in encoded]

    assert decoded == envelopes
    assert all(event.startswith('{"type": "debug"') for event in encoded)


@pytest.mark.asyncio
async def test_sse_lifecycle_timestamps_and_outputs_match_execution():
    _, envelopes = await _run_node_through_sse_queue()
    start, end = [event["content"] for event in envelopes]

    assert datetime.fromisoformat(start["start_time"])
    assert datetime.fromisoformat(end["start_time"])
    assert datetime.fromisoformat(end["end_time"])
    assert end["outputs"] == {}
    assert end["execution_duration_ms"] >= 0
