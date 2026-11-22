"""
Tests: DefaultObserver collector integration.

Verifies that all hooks route through DebugContext.emit() so the
DebugCollector receives events and populates the execution summary
with node-level data.  Without this path, get_summary().nodes is
empty and the external ``debug_summary`` event carries ``nodes=[]``.
"""

import pytest
from datetime import datetime, UTC

from magic_agents.debug.default_observer import DefaultObserver
from magic_agents.debug.config import DebugConfig
from magic_agents.debug.emitter import CallbackEmitter, EmitterRegistry
from magic_agents.debug.events import DebugEventType


@pytest.fixture
def observer():
    """Create a DefaultObserver with a mock emitter registry."""
    emitter_registry = EmitterRegistry()
    config = DebugConfig(enabled=True)
    obs = DefaultObserver(
        execution_id="test-exec-1",
        graph_type="agent",
        config=config,
        emitter_registry=emitter_registry,
        total_nodes=3,
    )
    return obs


@pytest.mark.asyncio
async def test_node_start_end_populates_summary_nodes(observer):
    """After on_node_start + on_node_end, the collector summary contains the node."""
    await observer.on_node_start(
        node_id="node-1",
        node_type="text",
        node_class="TextNode",
        inputs={"prompt": "hello"},
    )
    await observer.on_node_end(
        node_id="node-1",
        node_type="text",
        node_class="TextNode",
        outputs={"result": "Hello world"},
        internal_state={},
        duration_ms=150.0,
        start_time=datetime.now(UTC).isoformat(),
    )

    summary = observer._ctx.collector.get_summary()
    assert "node-1" in summary.nodes, (
        f"Collector should have 'node-1', got keys: {list(summary.nodes.keys())}"
    )
    node_summary = summary.nodes["node-1"]
    assert node_summary.was_executed is True
    assert node_summary.duration_ms == 150.0


@pytest.mark.asyncio
async def test_node_bypass_populates_summary_nodes(observer):
    """After on_node_bypass, the collector summary contains the bypassed node."""
    await observer.on_node_bypass(
        node_id="node-bypass-1",
        node_type="text",
        node_class="TextNode",
        reason="conditional_not_taken",
    )

    summary = observer._ctx.collector.get_summary()
    assert "node-bypass-1" in summary.nodes, (
        f"Bypassed node missing from summary: {list(summary.nodes.keys())}"
    )
    assert summary.nodes["node-bypass-1"].was_bypassed is True


@pytest.mark.asyncio
async def test_multiple_nodes_in_summary(observer):
    """Multiple node events all appear in the collector summary dict."""
    node_ids = ["parser-1", "llm-2", "send-3", "end-4"]
    for i, nid in enumerate(node_ids):
        await observer.on_node_start(
            node_id=nid,
            node_type="text",
            node_class="GenericNode",
            inputs={"idx": i},
        )
        await observer.on_node_end(
            node_id=nid,
            node_type="text",
            node_class="GenericNode",
            outputs={"idx": i},
            internal_state={},
            duration_ms=float(i * 10),
            start_time=datetime.now(UTC).isoformat(),
        )

    summary = observer._ctx.collector.get_summary()
    assert set(summary.nodes.keys()) == set(node_ids), (
        f"Expected nodes {set(node_ids)}, got {set(summary.nodes.keys())}"
    )
    assert summary.executed_nodes == 4


@pytest.mark.asyncio
async def test_mixed_execute_and_bypass_in_summary(observer):
    """Mixed executed and bypassed nodes all appear in summary."""
    # Execute node-1
    await observer.on_node_start(
        node_id="node-1", node_type="text", node_class="TextNode", inputs={}
    )
    await observer.on_node_end(
        node_id="node-1", node_type="text", node_class="TextNode",
        outputs={}, internal_state={}, duration_ms=10.0,
        start_time=datetime.now(UTC).isoformat(),
    )
    # Bypass node-2
    await observer.on_node_bypass(
        node_id="node-2", node_type="conditional", node_class="CondNode",
        reason="not_taken",
    )

    summary = observer._ctx.collector.get_summary()
    assert set(summary.nodes.keys()) == {"node-1", "node-2"}
    assert summary.nodes["node-1"].was_executed is True
    assert summary.nodes["node-1"].was_bypassed is False
    assert summary.nodes["node-2"].was_bypassed is True
    assert summary.nodes["node-2"].was_executed is False
    assert summary.executed_nodes == 1
    assert summary.bypassed_nodes == 1


@pytest.mark.asyncio
async def test_on_graph_end_summary_legacy_format_has_nodes(observer):
    """to_legacy_format() on the graph-end summary produces a non-empty nodes list."""
    # Simulate a simple graph with 2 executed nodes and 1 bypassed
    await observer.on_node_start(
        node_id="n1", node_type="text", node_class="TextNode", inputs={}
    )
    await observer.on_node_end(
        node_id="n1", node_type="text", node_class="TextNode",
        outputs={}, internal_state={}, duration_ms=10.0,
        start_time=datetime.now(UTC).isoformat(),
    )
    await observer.on_node_start(
        node_id="n2", node_type="text", node_class="TextNode", inputs={}
    )
    await observer.on_node_end(
        node_id="n2", node_type="text", node_class="TextNode",
        outputs={}, internal_state={}, duration_ms=20.0,
        start_time=datetime.now(UTC).isoformat(),
    )
    await observer.on_node_bypass(
        node_id="n3", node_type="conditional", node_class="CondNode",
        reason="not_taken",
    )

    emitted = []
    callback = CallbackEmitter()
    callback.add_sync_callback(emitted.append)
    observer.emitter_registry.register(callback)
    await observer.on_graph_end(
        graph_type="agent", execution_id="test-exec-1", total_duration_ms=42.0,
        node_count=3, executed_count=2, bypassed_count=1, failed_count=0,
    )

    graph_ends = [event for event in emitted if event.event_type == DebugEventType.GRAPH_END]
    assert len(graph_ends) == 1
    legacy = graph_ends[0].payload
    assert {n["node_id"] for n in legacy["nodes"]} == {"n1", "n2", "n3"}


@pytest.mark.asyncio
async def test_graph_start_sets_collector_start_time(observer):
    """GRAPH_START event is collected, setting the collector's start_time."""
    await observer.on_graph_start(
        graph_type="agent",
        execution_id="test-exec-1",
        node_count=3,
        edge_count=2,
    )
    assert observer._ctx.collector._start_time is not None, (
        "collector._start_time should be set after graph_start"
    )
