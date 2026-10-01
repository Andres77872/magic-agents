"""Loop iterations never reuse another iteration's values.

Inputs produced inside an iteration (by the loop item or another iteration
node) are cleared before each iteration; pre-loop inputs are kept. A node runs
with what arrived for the item, and is skipped when no in-iteration value
arrived (its producers failed, were bypassed or stayed silent): pre-loop
inputs such as a Client or a fixed Text never make it run on their own.
"""
import asyncio

import pytest

from magic_agents.agt_flow import build
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.hooks.flow_hooks import FlowHooks
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.node_system.NodeParser import NodeParser

pytestmark = pytest.mark.asyncio

PRODUCE = "{% if value == 'bad' %}{{ missing.invalid() }}{% endif %}r-{{ value }}"


def edge(identifier, source, target, source_handle, target_handle):
    return {"id": identifier, "source": source, "target": target,
            "sourceHandle": source_handle, "targetHandle": target_handle}


def loop_graph(items, *, consume="got:{{ r }}", item_into_consumer=False, static_into_consumer=False):
    """input -> loop -> produce -> consume -> loop; optional fan-in into consume."""
    nodes = [
        {"id": "input", "type": "user_input"},
        {"id": "loop", "type": "loop"},
        {"id": "produce", "type": "parser", "data": {"text": PRODUCE}},
        {"id": "consume", "type": "parser", "data": {"text": consume}},
        {"id": "end", "type": "end"},
    ]
    edges = [
        edge("e1", "input", "loop", "handle_user_message", "handle_list"),
        edge("e2", "loop", "produce", "handle_item", "value"),
        edge("e3", "produce", "consume", "handle_parser_output", "r"),
        edge("e4", "consume", "loop", "handle_parser_output", "handle_loop"),
        edge("e5", "loop", "end", "handle_end", "handle_flow_input"),
    ]
    if item_into_consumer:
        edges.append(edge("e6", "loop", "consume", "handle_item", "item"))
    if static_into_consumer:
        nodes.append({"id": "label", "type": "text", "data": {"text": "LBL"}})
        edges.append(edge("e7", "label", "consume", "handle_text_output", "label"))
    return build({"type": "graph", "timeout": 3, "nodes": nodes, "edges": edges}, message=items)


class GraphStatus(FlowHooks):
    def __init__(self):
        self.status = None
        self.bypasses = []

    async def on_node_bypass(self, context, reason):
        self.bypasses.append((context.node_id, reason, dict(context.metadata or {})))

    async def on_graph_error(self, context, error):
        self.status = "error"

    async def on_graph_end(self, context):
        self.status = "success"


async def run(graph, status=None):
    status = status or GraphStatus()
    registry = HookRegistry()
    registry.register_graph(status)
    await asyncio.wait_for(_drain(execute_graph_reactive(graph, hooks=registry)), 5)
    return graph.nodes["loop"].outputs["handle_end"]["content"], status.status


async def _drain(generator):
    return [event async for event in generator]


@pytest.fixture
def quiet_on(monkeypatch):
    """Make ``produce`` complete silently (no output) for the item 'quiet'."""
    original = NodeParser.process

    async def process(self, chat_log):
        if self.node_id == "produce" and self.inputs.get("value") == "quiet":
            return
            yield  # pragma: no cover - generator marker
        async for item in original(self, chat_log):
            yield item

    monkeypatch.setattr(NodeParser, "process", process)


@pytest.mark.parametrize("items, expected", [
    ('["ok", "bad", "ok3"]', ["got:r-ok", None, "got:r-ok3"]),
    ('["ok", "bad"]', ["got:r-ok", None]),
    ('["bad", "ok"]', [None, "got:r-ok"]),
], ids=["fail-in-2nd", "ok-then-fail", "fail-then-ok"])
async def test_failed_producer_never_feeds_the_previous_value(items, expected):
    values, status = await run(loop_graph(items))
    assert values == expected
    assert status == "error"


async def test_fan_in_consumer_runs_with_what_arrived_not_with_stale_values():
    graph = loop_graph('["ok", "bad", "ok3"]', consume="{{ item }}:{{ r }}", item_into_consumer=True)
    values, status = await run(graph)
    assert values == ["ok:r-ok", "bad:", "ok3:r-ok3"]
    assert status == "error"


async def test_pre_loop_inputs_are_kept_but_never_run_a_node_on_their_own():
    graph = loop_graph('["ok", "bad", "ok3"]', consume="{{ label }}:{{ r }}", static_into_consumer=True)
    values, status = await run(graph)
    # The pre-loop label reaches every item that ran; for the failed item the
    # label alone does not make ``consume`` run, so the slot is null.
    assert values == ["LBL:r-ok", None, "LBL:r-ok3"]
    assert status == "error"


async def test_llm_with_a_pre_loop_client_is_skipped_when_its_message_producer_fails(monkeypatch):
    """The common body shape: item -> Parser -> LLM (Client wired from before the loop)."""
    calls = []

    async def answer(self, chat_log):
        calls.append(sorted(self.inputs))
        message = self.inputs.get(self.INPUT_HANDLER_USER_MESSAGE)
        yield self.yield_static(f"A:{message}", content_type="handle_generated_content")

    monkeypatch.setattr(NodeLLM, "process", answer)
    graph = build({"type": "graph", "timeout": 3, "nodes": [
        {"id": "input", "type": "user_input"},
        {"id": "loop", "type": "loop"},
        {"id": "client", "type": "client", "data": {
            "engine": "openai", "model": "fake", "api_info": {"api_key": "fake", "base_url": "http://127.0.0.1:9"}}},
        {"id": "produce", "type": "parser", "data": {"text": PRODUCE}},
        {"id": "llm", "type": "llm", "data": {"stream": False, "iterate": True}},
        {"id": "end", "type": "end"},
    ], "edges": [
        edge("e1", "input", "loop", "handle_user_message", "handle_list"),
        edge("e2", "loop", "produce", "handle_item", "value"),
        edge("e3", "produce", "llm", "handle_parser_output", "handle_user_message"),
        edge("e4", "client", "llm", "handle-client-provider", "handle-client-provider"),
        edge("e5", "llm", "loop", "handle_generated_content", "handle_loop"),
        edge("e6", "loop", "end", "handle_end", "handle_flow_input"),
    ]}, message='["ok", "bad", "ok3"]')
    status = GraphStatus()
    values, outcome = await run(graph, status)
    assert values == ["A:r-ok", None, "A:r-ok3"]
    assert len(calls) == 2 and all("handle_user_message" in inputs for inputs in calls)
    assert outcome == "error"
    # Skipped because its producer failed: reported like the reactive executor.
    assert status.bypasses == [("llm", "upstream_error",
                                {"phase": "iteration", "upstream_error_node": "produce", "iteration": 1})]


@pytest.mark.parametrize("items, expected", [
    ('["ok", "quiet", "ok3"]', ["got:r-ok", None, "got:r-ok3"]),
    ('["quiet", "ok"]', [None, "got:r-ok"]),
], ids=["silent-in-2nd", "silent-first"])
async def test_silent_producer_skips_its_consumer_for_that_iteration(quiet_on, items, expected):
    graph = loop_graph(items)
    status = GraphStatus()
    values, outcome = await run(graph, status)
    assert values == expected
    # Nothing failed: a silent handle is skipped, not an error.
    assert outcome == "success"
    assert [(node, reason) for node, reason, _ in status.bypasses] == [("consume", "not_ready")]


@pytest.mark.parametrize("phase", ["static", "post_loop"])
async def test_failure_outside_the_iterations_is_reported_as_upstream_error(phase):
    """A Loop graph reports failure-caused bypasses like the reactive executor."""
    nodes = [
        {"id": "input", "type": "user_input"},
        {"id": "loop", "type": "loop"},
        {"id": "body", "type": "parser", "data": {"text": "b-{{ value }}"}},
        {"id": "broken", "type": "parser", "data": {"text": "{{ missing.invalid() }}"}},
        {"id": "after", "type": "parser", "data": {"text": "{{ v }}"}},
        {"id": "end", "type": "end"},
    ]
    edges = [
        edge("e1", "input", "loop", "handle_user_message", "handle_list"),
        edge("e2", "loop", "body", "handle_item", "value"),
        edge("e3", "body", "loop", "handle_parser_output", "handle_loop"),
        edge("e5", "broken", "after", "handle_parser_output", "v"),
        edge("e6", "after", "end", "handle_parser_output", "handle_flow_input"),
    ]
    if phase == "static":
        edges.append(edge("e4", "input", "broken", "handle_user_message", "x"))
    else:
        edges.append(edge("e4", "loop", "broken", "handle_end", "x"))
    graph = build({"type": "graph", "timeout": 3, "nodes": nodes, "edges": edges}, message='["a"]')
    status = GraphStatus()
    values, outcome = await run(graph, status)
    assert values == ["b-a"] and outcome == "error"
    assert [(node, reason, metadata) for node, reason, metadata in status.bypasses if node == "after"] == [
        ("after", "upstream_error", {"phase": phase, "upstream_error_node": "broken"})]
