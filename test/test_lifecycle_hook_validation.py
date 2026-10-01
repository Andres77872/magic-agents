"""Production build-boundary lifecycle checks, without executing templates."""
from copy import deepcopy

import pytest

from magic_agents.agt_flow import validate_graph
from magic_agents.util.lifecycle_hook_validation import validate_lifecycle_hooks


def graph(*, event="onError", target="primary", destination=None, handle="handle_fetch_input", hook_handles=None):
    nodes = [
        {"id": "input", "type": "user_input"},
        {"id": "primary", "type": "fetch", "data": {}},
        destination or {"id": "backup", "type": "fetch", "data": {"tool_mode": True}},
        {"id": "hook", "type": "hook", "data": {
            "function_template": "async def hook(context, chat_log):\n    return None",
            "lifecycle_event": event, "target_node_id": target,
            "handles": hook_handles or {},
        }},
    ]
    edges = [
        {"id": "primary-call", "source": "input", "target": "primary",
         "sourceHandle": "handle_user_message", "targetHandle": "handle_fetch_input"},
        {"id": "child", "source": "hook", "target": "backup",
         "sourceHandle": (hook_handles or {}).get("child_call", "handle-child-call"),
         "targetHandle": handle},
    ]
    return nodes, edges


@pytest.mark.parametrize("event", ["onStart", "onDeliver", "onError", "onFinish", "onCancel"])
def test_all_events_allow_unused_declared_child_capabilities(event):
    nodes, edges = graph(event=event)
    assert validate_graph(nodes, edges) == {"valid": True, "errors": []}


@pytest.mark.parametrize("target", ["missing", "hook", "annotation", "", 0])
def test_missing_nonexecuting_and_invalid_targets_rejected_at_build_boundary(target):
    nodes, edges = graph(target=target)
    nodes.append({"id": "annotation", "type": "note"})
    result = validate_graph(nodes, edges)
    assert not result["valid"]
    assert any("target" in item["error_message"] for item in result["errors"])


@pytest.mark.parametrize("event", ["onOops", "", False, [], {}])
def test_invalid_event_returns_errors_instead_of_raising(event):
    nodes, edges = graph(event=event)
    assert any(item["context"].get("field") == "lifecycle_event"
               for item in validate_lifecycle_hooks(nodes, edges))


def test_scope_conflict_and_disabled_old_attachment():
    nodes, edges = graph()
    edges[0]["hooks"] = {"hook_node_id": "hook", "enabled": True}
    assert any("combine node scope" in item["error_message"] for item in validate_graph(nodes, edges)["errors"])
    edges[0]["hooks"]["enabled"] = False
    assert validate_graph(nodes, edges)["valid"]


def test_edge_scope_needs_enabled_attachment_and_keeps_single_binding_rule():
    nodes, edges = graph(target=None)
    assert any("enabled connection" in item["error_message"] for item in validate_graph(nodes, edges)["errors"])
    edges[0]["hooks"] = {"hook_node_id": "hook"}
    assert validate_graph(nodes, edges)["valid"]
    edges.append({"id": "another", "source": "primary", "target": "backup",
                  "hooks": {"hook_node_id": "hook"}})
    assert any("multiple enabled edges" in item["error_message"] for item in validate_graph(nodes, edges)["errors"])


def test_legacy_observer_preserved_but_no_target_or_child_capability():
    nodes, edges = graph(event=None, target=None)
    assert any("lifecycle Hook" in item["error_message"] for item in validate_graph(nodes, edges)["errors"])
    edges.pop()
    edges[0]["hooks"] = {"hook_node_id": "hook"}
    assert validate_graph(nodes, edges)["valid"]
    nodes[-1]["data"]["target_node_id"] = "primary"
    assert any("needs a lifecycle_event" in item["error_message"] for item in validate_graph(nodes, edges)["errors"])


@pytest.mark.parametrize("destination,handle", [
    ({"id": "backup", "type": "parser", "data": {"text": "{{ custom_payload }}"}}, "custom_payload"),
    ({"id": "backup", "type": "parser", "data": {"text": "{{ provider }}"}}, "provider"),
    ({"id": "backup", "type": "llm", "data": {"handles": {"user_message": "prompt"}}}, "prompt"),
    ({"id": "backup", "type": "memory", "data": {"handles": {"input": "memo"}}}, "memo"),
    ({"id": "backup", "type": "python_exec", "data": {"code": "def run(handler): return handler"}}, "custom_payload"),
    ({"id": "backup", "type": "chat", "data": {"handles": {"messages": "conversation"}}}, "conversation"),
    ({"id": "backup", "type": "inner", "data": {"handles": {"input": "task"}}}, "task"),
])
def test_generic_and_custom_data_inputs_are_supported(destination, handle):
    nodes, edges = graph(destination=destination, handle=handle, hook_handles={"child_call": "run-child"})
    assert validate_lifecycle_hooks(nodes, edges) == []


@pytest.mark.parametrize("destination,handle", [
    ({"id": "backup", "type": "hook", "data": {}}, "handle-hook-context"),
    ({"id": "backup", "type": "section", "data": {}}, "content"),
    ({"id": "backup", "type": "client", "data": {}}, "handle-client-model"),
    ({"id": "backup", "type": "llm", "data": {}}, "handle-client-provider"),
    ({"id": "backup", "type": "llm", "data": {}}, "handle-tool-0"),
    ({"id": "backup", "type": "llm", "data": {"handles": {"tool_prefix": "registered-"}}}, "registered-0"),
    ({"id": "backup", "type": "llm", "data": {"handles": {"client": "engine"}}}, "engine"),
    ({"id": "backup", "type": "llm", "data": {"handles": {"temperature": "sampling"}}}, "sampling"),
    ({"id": "backup", "type": "llm", "data": {}}, "handle_generated_content"),
    ({"id": "backup", "type": "memory", "data": {"handles": {"client": "engine"}}}, "engine"),
    ({"id": "backup", "type": "conditional", "data": {"handles": {"client": "engine"}}}, "engine"),
    ({"id": "backup", "type": "python_exec", "data": {"code": "def run(handler): return handler", "handles": {"timeout": "budget"}}}, "budget"),
    ({"id": "backup", "type": "fetch", "data": {}}, "handle_fetch_output"),
    ({"id": "backup", "type": "fetch", "data": {}}, None),
])
def test_child_resource_configuration_output_and_missing_input_ports_rejected(destination, handle):
    nodes, edges = graph(destination=destination, handle=handle)
    assert any(item["context"].get("edge_id") == "child" for item in validate_lifecycle_hooks(nodes, edges))


def test_invalid_child_alias_missing_target_and_failure_policy():
    nodes, edges = graph(hook_handles={"child_call": "run-child"})
    edges[-1]["sourceHandle"] = "handle-child-call"
    assert validate_lifecycle_hooks(nodes, edges)
    edges[-1]["sourceHandle"] = "run-child"
    edges[-1]["target"] = "missing"
    assert validate_lifecycle_hooks(nodes, edges)
    nodes[-1]["data"]["failure_policy"] = "ignore"
    assert any(item["context"].get("field") == "failure_policy" for item in validate_lifecycle_hooks(nodes, edges))


def test_validation_does_not_mutate_graph_definitions():
    nodes, edges = graph()
    original = deepcopy((nodes, edges))
    validate_lifecycle_hooks(nodes, edges)
    assert (nodes, edges) == original
