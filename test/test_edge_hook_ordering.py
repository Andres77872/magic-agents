"""Edge Hooks finish and emit before the original input reaches its target."""
import asyncio

import pytest

from magic_agents.agt_flow import build, run_agent
from magic_agents.hooks.emit_context import EmitInterface
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes import ParserNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.node_system.NodeParser import NodeParser


def make_edge(source, target, source_handle, target_handle, hooked=False):
    value = {
        "id": f"{source}-{source_handle}-{target}-{target_handle}",
        "source": source,
        "target": target,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
    }
    if hooked:
        value["hooks"] = {"enabled": True, "hook_node_id": "hook"}
    return value


def make_graph(position, template=None, enabled=True):
    """Exercise normal dispatch and every boundary of the phase loop executor."""
    if template is None:
        template = """async def hook(context, chat_log):
    import asyncio
    await asyncio.sleep(0)
    return emit.user('hook:' + str(context.inputs['content']))
"""
    nodes = [
        {"id": "hook", "type": "hook", "data": {"function_template": template}},
        {"id": "input", "type": "user_input"},
        {"id": "target", "type": "parser", "data": {"text": "{{ handle_parser_input }}"}},
        {"id": "end", "type": "end"},
    ]
    if position == "normal":
        message = "payload"
        edges = [
            make_edge("input", "target", "handle_user_message", "handle_parser_input", True),
            make_edge("target", "end", "handle_parser_output", "handle_flow_input"),
        ]
    else:
        message = '["alpha", "beta"]'
        nodes.append({"id": "loop", "type": "loop"})
        edges = [
            make_edge("input", "loop", "handle_user_message", "handle_list", position == "static"),
            make_edge("loop", "target", "handle_item", "handle_parser_input", position == "item"),
            make_edge("target", "loop", "handle_parser_output", "handle_loop", position == "feedback"),
            make_edge("loop", "end", "handle_end", "handle_flow_input", position == "end"),
        ]
    hooked_edge = next(edge for edge in edges if "hooks" in edge)
    hooked_edge["hooks"]["enabled"] = enabled
    graph = build({"type": "graph", "debug": False, "timeout": 2, "nodes": nodes, "edges": edges}, message=message)
    return graph, hooked_edge


def record_order(graph, hooked_edge):
    """Capture input assignment separately from target processing.

    Target processing alone misses the loop executor's early add_parent call.
    """
    timeline = []
    target = graph.nodes[hooked_edge["target"]]
    target_handle = hooked_edge["targetHandle"]

    class RecordingInputs(dict):
        def __setitem__(self, key, value):
            if key == target_handle:
                timeline.append(("deliver", value))
            super().__setitem__(key, value)

    target.inputs = RecordingInputs(target.inputs)
    hook = graph.nodes["hook"]
    original_hook_process = hook.process

    async def hook_process(chat_log):
        payload = hook.inputs[hook.INPUT_HANDLE_HOOK_CONTEXT].inputs["content"]
        timeline.append(("hook-start", payload))
        async for event in original_hook_process(chat_log):
            timeline.append(("hook-emits", event["type"]))
            yield event
        timeline.append(("hook-finish", payload))

    hook.process = hook_process
    # Instrument parser target to create a user-visible output whose position
    # can be compared with the hook's output. Loop and end targets retain their
    # original behavior, since they own phase scheduling and terminal output.
    parser = graph.nodes["target"]
    original_parser_process = parser.process

    async def parser_process(chat_log):
        payload = parser.inputs["handle_parser_input"]
        timeline.append(("target-start", payload))
        yield EmitInterface(parser, "target").user("target:" + str(payload))
        async for event in original_parser_process(chat_log):
            yield event

    parser.process = parser_process
    return timeline


def content_messages(events):
    return [
        event["content"].choices[0].delta.content
        for event in events
        if event.get("type") == "content" and event["content"].choices[0].delta.content
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["normal", "static", "item", "feedback", "end"])
async def test_edge_hook_finishes_before_input_delivery(position):
    graph, hooked_edge = make_graph(position)
    timeline = record_order(graph, hooked_edge)
    events = [event async for event in run_agent(graph)]

    delivered = [(idx, payload) for idx, (kind, payload) in enumerate(timeline) if kind == "deliver"]
    assert delivered, timeline
    expected_count = 2 if position in ("item", "feedback") else 1
    assert sum(kind == "hook-start" for kind, _payload in timeline) == expected_count, timeline
    assert sum(kind == "hook-finish" for kind, _payload in timeline) == expected_count, timeline
    for idx, payload in delivered:
        assert ("hook-finish", payload) in timeline[:idx], timeline

    if position in ("normal", "item"):
        messages = content_messages(events)
        payloads = ["payload"] if position == "normal" else ["alpha", "beta"]
        for payload in payloads:
            assert messages.index("hook:" + payload) < messages.index("target:" + payload), messages
    if position != "normal":
        assert graph.nodes["loop"].outputs["handle_end"]["content"] == ["alpha", "beta"]
    assert "handle_end_output" in graph.nodes["end"].outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["normal", "item"])
@pytest.mark.parametrize("error_type", ["HookError", "HookTimeout"])
async def test_edge_hook_failure_is_emitted_before_fail_open_delivery(position, error_type):
    template = "async def hook(context, chat_log):\n    raise RuntimeError('hook failed')"
    graph, hooked_edge = make_graph(position, template=template)
    if error_type == "HookTimeout":
        # Explicit zero is deterministic and verifies the configured timeout
        # without incurring a real timeout delay.
        hooked = next(edge for edge in graph.edges if edge.hooks)
        hooked.hooks.timeout_override = 0
    timeline = record_order(graph, hooked_edge)
    events = [event async for event in run_agent(graph)]
    errors = [
        idx for idx, event in enumerate(events)
        if event.get("type") == "debug" and event.get("content", {}).get("error_type") == error_type
    ]
    target_events = [
        idx for idx, event in enumerate(events)
        if event.get("type") == "content" and event["content"].choices[0].delta.content.startswith("target:")
    ]
    assert len(errors) == len(target_events) == (1 if position == "normal" else 2), events
    assert all(error_idx < target_idx for error_idx, target_idx in zip(errors, target_events)), events
    for idx, (kind, payload) in enumerate(timeline):
        if kind == "deliver":
            assert ("hook-finish", payload) in timeline[:idx], timeline
    assert "handle_end_output" in graph.nodes["end"].outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["normal", "item"])
async def test_disabled_edge_hook_delivers_original_input_without_running_hook(position):
    graph, hooked_edge = make_graph(position, enabled=False)
    timeline = record_order(graph, hooked_edge)
    events = [event async for event in run_agent(graph)]
    assert not any(kind.startswith("hook-") for kind, _payload in timeline), timeline
    assert any(kind == "deliver" for kind, _payload in timeline), timeline
    assert not any(message.startswith("hook:") for message in content_messages(events))
    assert "handle_end_output" in graph.nodes["end"].outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["normal", "item"])
async def test_target_has_no_input_while_hook_is_still_running(position):
    graph, hooked_edge = make_graph(position)
    timeline = record_order(graph, hooked_edge)
    hook = graph.nodes["hook"]
    original_process = hook.process
    started = asyncio.Event()
    release = asyncio.Event()

    async def gated_process(chat_log):
        started.set()
        await release.wait()
        async for event in original_process(chat_log):
            yield event

    hook.process = gated_process

    async def collect():
        return [event async for event in run_agent(graph)]

    task = asyncio.create_task(collect())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not any(kind in ("deliver", "target-start") for kind, _payload in timeline), timeline
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=2)
    assert "handle_end_output" in graph.nodes["end"].outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["normal", "item"])
async def test_cancelling_a_running_hook_never_delivers_the_original_event(position):
    graph, hooked_edge = make_graph(position)
    timeline = record_order(graph, hooked_edge)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_process(chat_log):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        if False:
            yield None

    graph.nodes["hook"].process = blocked_process

    async def collect():
        return [event async for event in run_agent(graph)]

    task = asyncio.create_task(collect())
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert cancelled.is_set()
    assert not any(kind in ("deliver", "target-start") for kind, _payload in timeline)


@pytest.mark.asyncio
async def test_hook_processing_does_not_consume_downstream_input_wait_budget():
    template = """async def hook(context, chat_log):
    import asyncio
    await asyncio.sleep(0.10)
    return emit.user('hook:' + str(context.inputs['content']))
"""
    graph, hooked_edge = make_graph("normal", template=template)
    # Runtime input trackers accept subsecond budgets. Setting the built
    # model's value avoids a one-second production configuration minimum and
    # keeps this timeout regression fast.
    graph.timeout = 0.02
    graph.nodes["hook"]._timeout_seconds = 1
    timeline = record_order(graph, hooked_edge)
    events = [event async for event in run_agent(graph)]

    assert ("hook-finish", "payload") in timeline, timeline
    assert ("target-start", "payload") in timeline, timeline
    assert "handle_end_output" in graph.nodes["end"].outputs
    assert not any(
        event.get("type") == "debug"
        and event.get("content", {}).get("error_type") == "TimeoutError"
        for event in events
    ), events


DATA_HOOK_TEMPLATE = """async def hook(context, chat_log):
    calls = chat_log.flow_state.get('data_hook_calls', 0) + 1
    chat_log.flow_state['data_hook_calls'] = calls
    return emit._node.yield_static(
        {'value': context.inputs['content'], 'calls': calls},
        content_type='handle-user-output',
    )
"""


@pytest.mark.asyncio
async def test_loop_hook_data_consumer_runs_once_after_hook_for_every_item():
    graph, hooked_edge = make_graph("item", template=DATA_HOOK_TEMPLATE)
    side = NodeParser(
        data=ParserNodeModel(text="{{ handle_parser_input }}"),
        node_id="side", node_type="parser", debug=False,
    )
    graph.nodes["side"] = side
    graph.edges.append(EdgeNodeModel(**make_edge(
        "hook", "side", "handle-user-output", "handle_parser_input",
    )))
    timeline = record_order(graph, hooked_edge)
    seen = []

    async def side_process(chat_log):
        payload = side.inputs.get("handle_parser_input")
        seen.append(payload)
        timeline.append(("side-start", payload))
        yield side.yield_static(payload, content_type="handle_parser_output")

    side.process = side_process
    events = [event async for event in run_agent(graph)]
    assert seen == [
        {"value": "alpha", "calls": 1},
        {"value": "beta", "calls": 2},
    ], (seen, timeline, events)
    for idx, (kind, payload) in enumerate(timeline):
        if kind == "side-start":
            assert ("hook-finish", payload["value"]) in timeline[:idx], timeline
        elif kind == "target-start":
            assert ("hook-finish", payload) in timeline[:idx], timeline
    assert graph.nodes["loop"].outputs["handle_end"]["content"] == ["alpha", "beta"]
    assert "handle_end_output" in graph.nodes["end"].outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_source_to_hook", [False, True])
async def test_hook_data_and_original_input_converge_without_duplicate_hook(explicit_source_to_hook):
    graph, hooked_edge = make_graph("normal", template=DATA_HOOK_TEMPLATE)
    # Keep the original edge before the optional explicit source -> Hook edge.
    # The traversal-owned Hook must not wait for that later edge and deadlock
    # its source while the source is still delivering its outgoing edges.
    if explicit_source_to_hook:
        graph.edges.append(EdgeNodeModel(**make_edge(
            "input", "hook", "handle_user_message", "legacy-input",
        )))
    graph.edges.append(EdgeNodeModel(**make_edge(
        "hook", "target", "handle-user-output", "hook-data",
    )))
    target = graph.nodes["target"]
    timeline = record_order(graph, hooked_edge)
    seen = []
    original_target_process = target.process

    async def capture_target_inputs(chat_log):
        seen.append(dict(target.inputs))
        async for event in original_target_process(chat_log):
            yield event

    target.process = capture_target_inputs
    events = await asyncio.wait_for(
        collect_events(graph), timeout=1,
    )
    assert seen == [{
        "hook-data": {"value": "payload", "calls": 1},
        "handle_parser_input": "payload",
    }], (seen, timeline, events)
    assert [payload for kind, payload in timeline if kind == "hook-start"] == ["payload"], timeline
    assert "handle_end_output" in graph.nodes["end"].outputs


async def collect_events(graph):
    return [event async for event in run_agent(graph)]


@pytest.mark.asyncio
async def test_ready_independent_branch_can_release_running_hook_after_wait_budget_expires():
    graph, hooked_edge = make_graph("normal")
    graph.timeout = 0.02
    release_hook = asyncio.Event()
    hook = graph.nodes["hook"]
    hook._timeout_seconds = 0.20
    original_execute_function = hook._execute_function

    # Gate the actual function inside NodeHook's own timeout. This creates a
    # dependency on an independent graph branch without modifying hook runtime
    # behavior or requiring network services.
    async def wait_for_independent_branch(func, context, chat_log):
        await release_hook.wait()
        return await original_execute_function(func, context, chat_log)

    hook._execute_function = wait_for_independent_branch
    timeline = record_order(graph, hooked_edge)

    class DelayedSource(Node):
        async def process(self, chat_log):
            # B's input arrives after its ordinary wait budget expires, while
            # A's Hook is still running and input waiting is paused.
            await asyncio.sleep(0.05)
            yield self.yield_static("independent", content_type="out")

    class ReleasingTarget(Node):
        async def process(self, chat_log):
            timeline.append(("independent-start", self.inputs["in"]))
            release_hook.set()
            yield self.yield_static(self.inputs["in"], content_type="out")

    graph.nodes["independent-source"] = DelayedSource(
        node_id="independent-source", node_type="test", debug=False,
    )
    graph.nodes["independent-target"] = ReleasingTarget(
        node_id="independent-target", node_type="test", debug=False,
    )
    graph.edges.append(EdgeNodeModel(**make_edge(
        "independent-source", "independent-target", "out", "in",
    )))
    events = await asyncio.wait_for(collect_events(graph), timeout=0.5)

    assert timeline.index(("independent-start", "independent")) < timeline.index(("hook-finish", "payload")), timeline
    assert timeline.index(("hook-finish", "payload")) < timeline.index(("target-start", "payload")), timeline
    assert "handle_end_output" in graph.nodes["end"].outputs
    assert not any(
        event.get("type") == "debug"
        and event.get("content", {}).get("error_type")
        for event in events
    ), events
