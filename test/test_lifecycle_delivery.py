"""Lifecycle delivery interception before input assignment on both executors."""
import asyncio

import pytest

from magic_agents.agt_flow import build, run_agent


def edge(source, target, source_handle, target_handle, *, hooked=False):
    value = {"id": f"{source}-{target}", "source": source, "target": target,
             "sourceHandle": source_handle, "targetHandle": target_handle}
    if hooked:
        value["hooks"] = {"hook_node_id": "intercept", "enabled": True}
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["normal", "static", "item", "feedback", "end"])
async def test_delivery_hook_replaces_value_before_target_assignment(position):
    nodes = [
        {"id": "input", "type": "user_input"},
        {"id": "parser", "type": "parser", "data": {"text": "{{ value }}"}},
        {"id": "end", "type": "end"},
        {"id": "intercept", "type": "hook", "data": {
            "lifecycle_event": "onDeliver", "failure_policy": "fail",
            "function_template": """async def intercept(context, chat_log):
    import asyncio
    import json
    await asyncio.sleep(0)
    value = context.request["content"]
    if isinstance(value, list):
        value = [str(item).upper() for item in value]
    else:
        value = str(value).upper()
    return {"action": "input", "content": value}
"""}},
    ]
    if position == "normal":
        message = "alpha"
        edges = [edge("input", "parser", "handle_user_message", "value", hooked=True),
                 edge("parser", "end", "handle_parser_output", "handle_flow_input")]
    else:
        message = '["alpha", "beta"]'
        nodes.append({"id": "loop", "type": "loop"})
        edges = [
            edge("input", "loop", "handle_user_message", "handle_list", hooked=position == "static"),
            edge("loop", "parser", "handle_item", "value", hooked=position == "item"),
            edge("parser", "loop", "handle_parser_output", "handle_loop", hooked=position == "feedback"),
            edge("loop", "end", "handle_end", "handle_flow_input", hooked=position == "end"),
        ]
    graph = build({"type": "graph", "timeout": 2, "nodes": nodes, "edges": edges}, message=message)
    intercepted = next(item for item in edges if item.get("hooks"))
    target = graph.nodes[intercepted["target"]]
    hook = graph.nodes["intercept"]
    timeline = []

    class Inputs(dict):
        def __setitem__(self, key, value):
            if key == intercepted["targetHandle"]:
                timeline.append(("delivered", value))
            super().__setitem__(key, value)

    target.inputs = Inputs(target.inputs)
    invoke = hook.invoke_control

    async def record_hook(context, chat_log):
        timeline.append(("hook-start", context.request["content"]))
        result = await invoke(context, chat_log)
        timeline.append(("hook-finish", result["content"]))
        return result

    hook.invoke_control = record_hook

    async def consume():
        return [item async for item in run_agent(graph)]

    await asyncio.wait_for(consume(), 3)
    count = 2 if position in ("item", "feedback") else 1
    assert len(timeline) == 3 * count, timeline
    for index in range(0, len(timeline), 3):
        assert [event for event, _ in timeline[index:index + 3]] == ["hook-start", "hook-finish", "delivered"]
        assert timeline[index + 1][1] == timeline[index + 2][1]
    assert graph.nodes["end"].inputs["handle_flow_input"] == ("ALPHA" if position == "normal" else ["ALPHA", "BETA"])
