"""Nested edge Hooks gate delivery; skipped Hook routes preserve convergence."""

import pytest

from magic_agents.agt_flow import build, run_agent


DATA_HOOK = """async def hook(context, chat_log):
    return emit._node.yield_static(
        {'value': context.inputs['content']},
        content_type='handle-user-output',
    )
"""

NESTED_HOOK = """async def hook(context, chat_log):
    import asyncio
    await asyncio.sleep(0)
    return emit.user('nested:' + str(context.inputs['content']['value']))
"""


def edge(source, target, source_handle, target_handle, hook=None):
    result = {
        "id": f"{source}-{source_handle}-{target}-{target_handle}",
        "source": source,
        "target": target,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
    }
    if hook:
        result["hooks"] = {"enabled": True, "hook_node_id": hook}
    return result


def parser(node_id, text="{{ handle_parser_input }}"):
    return {"id": node_id, "type": "parser", "data": {"text": text}}


def record_deliveries(node, timeline, name):
    class RecordingInputs(dict):
        def __setitem__(self, key, value):
            if key == "handle_parser_input":
                timeline.append((name, value))
            super().__setitem__(key, value)

    node.inputs = RecordingInputs(node.inputs)


def record_hook(node, timeline, name, nested=False):
    original = node.process

    async def process(chat_log):
        payload = node.inputs[node.INPUT_HANDLE_HOOK_CONTEXT].inputs["content"]
        value = payload["value"] if nested else payload
        timeline.append((name + "-start", value))
        async for output in original(chat_log):
            yield output
        timeline.append((name + "-finish", value))

    node.process = process


@pytest.mark.asyncio
@pytest.mark.parametrize("loop", [False, True])
async def test_nested_hook_finishes_before_both_original_inputs_are_delivered(loop):
    nodes = [
        {"id": "hook-b", "type": "hook", "data": {"function_template": NESTED_HOOK}},
        {"id": "hook-a", "type": "hook", "data": {"function_template": DATA_HOOK}},
        {"id": "input", "type": "user_input"},
        parser("target"),
        parser("side"),
        {"id": "end", "type": "end"},
    ]
    edges = [edge("hook-a", "side", "handle-user-output", "handle_parser_input", "hook-b")]
    if loop:
        nodes.append({"id": "loop", "type": "loop"})
        edges.extend([
            edge("input", "loop", "handle_user_message", "handle_list"),
            edge("loop", "target", "handle_item", "handle_parser_input", "hook-a"),
            edge("target", "loop", "handle_parser_output", "handle_loop"),
            edge("loop", "end", "handle_end", "handle_flow_input"),
        ])
        message = '["alpha", "beta"]'
        expected = ["alpha", "beta"]
    else:
        edges.extend([
            edge("input", "target", "handle_user_message", "handle_parser_input", "hook-a"),
            edge("target", "end", "handle_parser_output", "handle_flow_input"),
        ])
        message = "payload"
        expected = ["payload"]
    graph = build({"nodes": nodes, "edges": edges, "debug": False}, message=message)
    timeline = []
    record_deliveries(graph.nodes["target"], timeline, "target-deliver")
    record_deliveries(graph.nodes["side"], timeline, "side-deliver")
    record_hook(graph.nodes["hook-a"], timeline, "a")
    record_hook(graph.nodes["hook-b"], timeline, "b", nested=True)

    events = [event async for event in run_agent(graph)]

    assert [value for kind, value in timeline if kind == "a-start"] == expected, timeline
    assert [value for kind, value in timeline if kind == "b-start"] == expected, timeline
    for index, (kind, payload) in enumerate(timeline):
        if kind in {"target-deliver", "side-deliver"}:
            value = payload["value"] if kind == "side-deliver" else payload
            assert ("a-finish", value) in timeline[:index], timeline
            assert ("b-finish", value) in timeline[:index], timeline
    assert {value for kind, value in timeline if kind == "target-deliver"} == set(expected)
    assert [value["value"] for kind, value in timeline if kind == "side-deliver"]
    assert "handle_end_output" in graph.nodes["end"].outputs
    assert not any(
        event.get("type") == "debug" and event.get("content", {}).get("error_type")
        for event in events
    ), events


@pytest.mark.asyncio
@pytest.mark.parametrize("active_alternative", [False, True])
async def test_skipped_hook_side_route_preserves_only_active_convergent_input(active_alternative):
    nodes = [
        {"id": "hook-a", "type": "hook", "data": {"function_template": DATA_HOOK}},
        {"id": "input", "type": "user_input"},
        {"id": "loop", "type": "loop"},
        {"id": "route", "type": "conditional", "data": {
            "condition": "{{ 'yes' }}", "output_handles": ["yes", "no"],
        }},
        parser("target", "{{ handle_parser_input.value }}"),
        parser("dead"),
        parser("side"),
        {"id": "end", "type": "end"},
    ]
    edges = [
        edge("input", "loop", "handle_user_message", "handle_list"),
        edge("loop", "route", "handle_item", "handle_input"),
        edge("route", "target", "yes", "handle_parser_input"),
        edge("route", "dead", "no", "handle_parser_input", "hook-a"),
        edge("hook-a", "side", "handle-user-output", "handle_parser_input"),
        edge("target", "loop", "handle_parser_output", "handle_loop"),
        edge("loop", "end", "handle_end", "handle_flow_input"),
    ]
    if active_alternative:
        edges.append(edge("route", "side", "yes", "handle_parser_input"))
    graph = build({"nodes": nodes, "edges": edges, "debug": False}, message='["alpha", "beta"]')
    seen = []
    side = graph.nodes["side"]
    original = side.process

    async def process(chat_log):
        seen.append(side.inputs.get("handle_parser_input"))
        async for output in original(chat_log):
            yield output

    side.process = process
    events = [event async for event in run_agent(graph)]

    if active_alternative:
        assert [payload["value"] for payload in seen] == ["alpha", "beta"], seen
    else:
        assert seen == [], seen
    assert graph.nodes["hook-a"].outputs == {}
    assert graph.nodes["dead"].outputs == {}
    assert graph.nodes["loop"].outputs["handle_end"]["content"] == ["alpha", "beta"]
    assert "handle_end_output" in graph.nodes["end"].outputs
    assert not any(
        event.get("type") == "debug" and event.get("content", {}).get("error_type")
        for event in events
    ), events
