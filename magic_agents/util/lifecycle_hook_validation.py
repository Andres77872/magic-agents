"""Validate lifecycle bindings and declared child capabilities before execution.

This uses raw graph definitions so API imports receive the same scope rules as
the editor. It does not instantiate nodes or execute Hook templates.
"""
from magic_agents.util.handle_registry import get_canonical_input_handles
from magic_agents.util.const import HANDLE_VOID


LIFECYCLE_EVENTS = frozenset({"onStart", "onDeliver", "onError", "onFinish", "onCancel"})
EXECUTABLE_TYPES = frozenset({
    "chat", "llm", "end", "text", "constant", "user_input", "parser", "fetch",
    "client", "send_message", "void", "loop", "inner", "conditional",
    "python_exec", "mcp", "node_tool", "memory", "codex",
})

# Input aliases follow node constructor precedence. Output aliases must never
# make an output port look like a permitted child-call input.
INPUT_ALIASES = {
    "llm": {
        "handle-client-provider": ("client_provider", "client"),
        "handle-chat": ("chat",),
        "handle-system-context": ("system_context", "system"),
        "handle_user_message": ("user_message", "message"),
        **{f"handle-llm-{key}": (key,) for key in (
            "temperature", "top_p", "max_tokens", "reasoning_effort", "stream", "iterate")},
        "handle-llm-json_output": ("json_output", "json_mode"),
    },
    "fetch": {
        "handle-url": ("url",), "handle-fetch-method": ("method",),
        "handle-fetch-data": ("data",), "handle-fetch-json_data": ("json_data",),
        "handle-fetch-headers": ("headers",), "handle_fetch_input": ("input",),
    },
    "chat": {
        "handle-system-context": ("system_context", "system"),
        "handle_user_message": ("user_message", "message"),
        "handle_messages": ("messages",), "handle_user_files": ("user_files", "files"),
        "handle_user_images": ("user_images", "images"),
    },
    "conditional": {"handle_input": ("input", "context"),
                    "handle-client-provider": ("client_provider", "client")},
    "memory": {"handle_memory_input": ("input",), "handle-client-provider": ("client",)},
    "loop": {"handle_list": ("input_list", "list"), "handle_loop": ("input_loop", "loop")},
    "inner": {"handle_user_message": ("input", "user_message"),
              "handle_client_extras": ("client_extras",)},
    "python_exec": {f"handle-python_exec-{key}": (key,)
                    for key in ("safety_mode", "timeout", "max_output_chars")},
    "codex": {"handle_codex_input": ("input",)},
    "send_message": {"handle_send_extra": ("send_extra", "extra")},
}
CONFIG_INPUTS = {
    "llm": {f"handle-llm-{key}" for key in (
        "temperature", "top_p", "max_tokens", "reasoning_effort", "stream", "iterate", "json_output")},
    "python_exec": {f"handle-python_exec-{key}" for key in (
        "safety_mode", "timeout", "max_output_chars")},
}
RESOURCE_INPUTS = {
    "llm": {"handle-client-provider"},
    "conditional": {"handle-client-provider"},
    "memory": {"handle-client-provider"},
}


def _data(node):
    value = node.get("data")
    return value if isinstance(value, dict) else node


def _handles(node):
    value = _data(node).get("handles")
    return value if isinstance(value, dict) else {}


def _resolve(node, canonical):
    overrides = _handles(node)
    for alias in INPUT_ALIASES.get(node.get("type"), {}).get(canonical, ()):
        if isinstance(overrides.get(alias), str):
            return overrides[alias]
    return canonical


def child_call_target_error(node, handle):
    """Return a reason for resource/config/output ports, allowing real data inputs."""
    kind = node.get("type")
    if kind not in EXECUTABLE_TYPES or kind == "client":
        return "Child calls require an executable data node, not a Hook, annotation, or Client resource."
    if kind == "loop":
        return "Child calls cannot execute Loop orchestration directly; use an Inner Flow containing the Loop."
    if not isinstance(handle, str) or not handle.strip():
        return "Child calls require an explicit target data input handle."
    forbidden = CONFIG_INPUTS.get(kind, set()) | RESOURCE_INPUTS.get(kind, set())
    if handle in {_resolve(node, item) for item in forbidden}:
        return "Child calls cannot target Client, Tool, or execution-setting inputs."
    if kind == "llm":
        tool_prefix = _handles(node).get("tool_prefix", "handle-tool-")
        if handle == "handle-tool-definition" or isinstance(tool_prefix, str) and tool_prefix and handle.startswith(tool_prefix):
            return "Child calls cannot target Tool definition inputs."
    # These processors consume caller-declared names as operation data. A name
    # alone (for example 'provider') is not a live-resource semantic port.
    if kind in {"parser", "conditional"}:
        return None
    if kind == "python_exec" and _data(node).get("tool_mode") is not True and _data(node).get("code"):
        return None
    inputs = set(get_canonical_input_handles(kind)) | set(INPUT_ALIASES.get(kind, {}))
    if kind == "void":
        inputs.add(HANDLE_VOID)
    if handle in {_resolve(node, item) for item in inputs}:
        return None
    return f"'{handle}' is not a data input of node type '{kind}'."


def validate_lifecycle_hooks(nodes, edges):
    """Return structured graph errors; legacy observers retain their own checks."""
    by_id = {node.get("id"): node for node in nodes}
    enabled = {}
    errors = []

    def error(message, **context):
        errors.append({"error_type": "HookValidationError", "error_message": message,
                       "context": context})

    for edge in edges:
        binding = edge.get("hooks")
        if isinstance(binding, dict) and binding.get("enabled", True) is not False:
            enabled.setdefault(binding.get("hook_node_id"), []).append(edge)
    for node in nodes:
        if node.get("type") != "hook":
            continue
        node_id, data = node.get("id"), _data(node)
        event, target = data.get("lifecycle_event"), data.get("target_node_id")
        valid_event = isinstance(event, str) and event in LIFECYCLE_EVENTS
        if event is not None and not valid_event:
            error(f"Hook '{node_id}' has an invalid lifecycle_event.", node_id=node_id, field="lifecycle_event")
        policy = data.get("failure_policy", "preserve")
        if policy not in ("preserve", "fail"):
            error(f"Hook '{node_id}' failure_policy must be preserve or fail.", node_id=node_id, field="failure_policy")
        if target is not None:
            if not isinstance(target, str) or not target.strip():
                error(f"Hook '{node_id}' target_node_id must be a non-empty node ID or null.", node_id=node_id, field="target_node_id")
            elif event is None:
                error(f"Hook '{node_id}' needs a lifecycle_event to bind a target node.", node_id=node_id, target_node_id=target)
            elif target not in by_id or by_id[target].get("type") not in EXECUTABLE_TYPES:
                error(f"Hook '{node_id}' target '{target}' must reference an executable non-Hook node.", node_id=node_id, target_node_id=target)
            if isinstance(target, str) and by_id.get(target, {}).get("type") == "loop" and valid_event and event != "onDeliver":
                error(f"Hook '{node_id}' cannot target Loop orchestration with {event}; use onDeliver or target a node inside the Loop.",
                      node_id=node_id, target_node_id=target, field="target_node_id")
            if enabled.get(node_id):
                error(f"Hook '{node_id}' cannot combine node scope with an enabled connection binding.",
                      node_id=node_id, edge_ids=[item.get("id") for item in enabled[node_id]])
        elif event is not None and not enabled.get(node_id):
            error(f"Lifecycle Hook '{node_id}' needs a target node or an enabled connection binding.", node_id=node_id)

        if target is None and valid_event:
            for binding in enabled.get(node_id, []):
                destination = by_id.get(binding.get("target"), {})
                origin = by_id.get(binding.get("source"), {})
                if destination.get("type") == "loop" and event != "onDeliver":
                    error(f"Hook '{node_id}' cannot intercept Loop orchestration with {event}; use onDeliver or target a node inside the Loop.",
                          node_id=node_id, edge_id=binding.get("id"), field="lifecycle_event")
                if origin.get("type") == "node_tool" and destination.get("type") == "llm":
                    prefix = _handles(destination).get("tool_prefix", "handle-tool-")
                    handle = binding.get("targetHandle") or ""
                    if isinstance(prefix, str) and prefix and handle.startswith(prefix):
                        error(f"Hook '{node_id}' cannot control a Client Tool operation executed outside this server; use node scope to prepare its schema.",
                              node_id=node_id, edge_id=binding.get("id"), field="lifecycle_event")

        child_handle = _handles(node).get("child_call", "handle-child-call")
        for edge in edges:
            if edge.get("source") != node_id or edge.get("sourceHandle") not in (child_handle, "handle-child-call"):
                continue
            edge_id = edge.get("id")
            if not valid_event or edge.get("sourceHandle") != child_handle:
                error(f"Edge '{edge_id}' requires the configured Child call output of a lifecycle Hook.",
                      node_id=node_id, edge_id=edge_id)
                continue
            destination = by_id.get(edge.get("target"))
            reason = child_call_target_error(destination, edge.get("targetHandle")) if destination else "Child call target node does not exist."
            if reason:
                error(f"Edge '{edge_id}': {reason}", edge_id=edge_id, node_id=node_id,
                      target=edge.get("target"), targetHandle=edge.get("targetHandle"))
    return errors
