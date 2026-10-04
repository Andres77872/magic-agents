"""Fail-closed coordination topology validation before node construction.

Each ordinary graph/Inner invocation defines its own publication barrier. A
shared server budget does not turn all nested scopes into one barrier.
"""
from __future__ import annotations

from collections import deque
import ast
from dataclasses import dataclass
from typing import Any

from magic_agents.models.coordination import CoordinationPolicy, MessagingConfig


@dataclass(frozen=True)
class CoordinationDiagnostic:
    code: str
    message: str
    node_path: tuple[str, ...] = ()
    dependency_path: tuple[str, ...] = ()
    edge_path: tuple[str, ...] = ()


class CoordinationValidationError(ValueError):
    def __init__(self, diagnostics: list[CoordinationDiagnostic]):
        self.diagnostics = tuple(diagnostics)
        self.code = diagnostics[0].code
        super().__init__(f"{self.code}: {diagnostics[0].message}")


def supported_participant_callback(event, template):
    """The attached actor route currently admits plain async lifecycle bodies.

    Delivery before input readiness and synchronous/decorated callbacks need
    separate ownership/executor support. The selected callable is also checked
    by NodeHook immediately before invocation.
    """
    if event not in ('onStart', 'onError', 'onFinish', 'onCancel') or not isinstance(template, str):
        return False
    try: body = ast.parse(template).body
    except SyntaxError: return False
    return len(body) == 1 and isinstance(body[0], ast.AsyncFunctionDef) and not body[0].decorator_list


def normalize_definition(definition: dict[str, Any]) -> dict[str, Any]:
    """Accept saved/content wrappers without dropping executable graph policy."""
    if not isinstance(definition.get("content"), dict):
        return definition
    content = definition["content"]
    merged = {**content, **{k: v for k, v in definition.items() if k != "content"}}
    merged["nodes"] = content.get("nodes", [])
    merged["edges"] = content.get("edges", [])
    if "coordination" in content and "coordination" in definition and content["coordination"] != definition["coordination"]:
        raise CoordinationValidationError([CoordinationDiagnostic("invalid_coordination_config", "Conflicting wrapper and content coordination policies")])
    return merged


def validate_coordination_definition(definition: dict[str, Any], *, path: tuple[str, ...] = (), invocation=False) -> list[CoordinationDiagnostic]:
    graph = normalize_definition(definition)
    policy = CoordinationPolicy.model_validate(graph["coordination"]) if graph.get("coordination") is not None else None
    raw_nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(raw_nodes, list) or not isinstance(edges, list):
        return [CoordinationDiagnostic("invalid_coordination_config", "Authored graph nodes and edges must be lists", path)]
    from magic_agents.hooks.messages_hook import effective_messages_nodes, MessagesHookBindingError
    try:
        raw_nodes = effective_messages_nodes(raw_nodes, edges)
    except MessagesHookBindingError as exc:
        return [CoordinationDiagnostic('invalid_coordination_config', str(exc), path)]
    nodes = {n["id"]: n for n in raw_nodes if isinstance(n, dict) and "id" in n}
    diagnostics: list[CoordinationDiagnostic] = []
    actors: dict[str, tuple[str, MessagingConfig]] = {}

    def error(code, message, node_id=None, dependency=(), edge_ids=()):
        diagnostics.append(CoordinationDiagnostic(code, message, path + ((node_id,) if node_id else ()), tuple(dependency), tuple(edge_ids)))

    if any(marker in graph for marker in ("unresolved_messaging", "unresolved_hook_target", "unresolved_hook_child_calls")):
        error("invalid_coordination_config", "Resolve editor draft references before execution")

    for node_id, node in nodes.items():
        data = node.get("data") or {}
        if any(marker in data for marker in ("unresolved_messaging", "unresolved_hook_target", "unresolved_hook_child_calls")):
            error("invalid_coordination_config", "Resolve editor draft references before execution", node_id)
        if data.get("messaging") is not None:
            config = MessagingConfig.model_validate(data["messaging"])
            if node.get("type") != "llm":
                error("invalid_coordination_config", "Only LLM nodes may declare messaging", node_id)
            elif config.enabled:
                if config.role in actors:
                    error("invalid_coordination_config", "Participant role aliases must be unique within a scope", node_id)
                actors[config.role] = (node_id, config)
        if node.get("type") == "inner":
            child = next((data[k] for k in ("magic_flow", "flow", "graph", "subgraph") if data.get(k) is not None), None)
            if isinstance(child, dict):
                child_diagnostics = validate_coordination_definition(child, path=path + (node_id,), invocation=invocation)
                diagnostics.extend(child_diagnostics)
                child_policy = normalize_definition(child).get("coordination") or {}
                if data.get("tool_mode") and child_policy.get("enabled"):
                    error("unsupported_coordination_topology", "Coordinated Inner tools require explicit invocation scheduling support", node_id)

    if policy is None or not policy.enabled:
        if policy is None and actors:
            error("invalid_coordination_config", "Enabled participants require coordination in their enclosing graph")
        return diagnostics
    if graph.get("type", "graph") not in ("graph", "chat") or any(n.get("type") == "loop" for n in nodes.values()):
        error("unsupported_coordination_topology", "Coordination requires an ordinary concurrently scheduled graph")
    if set(actors) != set(policy.allowed_participants):
        error("invalid_coordination_config", "allowedParticipants must exactly name enabled roles in this scope")
    if len(nodes) != len(raw_nodes):
        error("invalid_coordination_config", "Coordinated graphs require unique node identities")
    participant_ids = {item[0] for item in actors.values()}
    for role, (node_id, config) in actors.items():
        if set(config.peers) - set(actors):
            error("invalid_coordination_config", "Peer aliases must resolve within the admitted scope", node_id)
        if role in config.peers:
            error("invalid_coordination_config", "Self messaging is not supported", node_id)

    adjacency: dict[str, list[tuple[str, str]]] = {node_id: [] for node_id in nodes}
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source not in nodes or target not in nodes:
            error("invalid_coordination_config", "Coordination edges must reference existing nodes")
            continue
        source_node, target_node = nodes[source], nodes[target]
        source_data = source_node.get("data") or {}
        source_handles = source_data.get("handles") or {}
        is_child = source_node.get("type") == "hook" and edge.get("sourceHandle") in ("handle-child-call", source_handles.get("child_call", "handle-child-call"))
        if is_child:
            if target in participant_ids:
                error("unsupported_coordination_topology", "On-demand participants require one proven actor-owned activation path", target)
            continue
        if target in participant_ids and (source_node.get("type") == "node_tool" or
                                         source_node.get("type") == "skills" and not invocation):
            reason = "Schema-only client tools cannot be mixed with native messaging" if source_node.get("type") == "node_tool" else "Skills continuation requires compatible canonical checkpoints"
            error("unsupported_coordination_topology", reason, target)
        hooks = edge.get("hooks") or {}
        if hooks.get("enabled") and target in participant_ids and not invocation:
            hook_data = (nodes.get(hooks.get('hook_node_id'), {}).get('data') or {})
            if not supported_participant_callback(hook_data.get('lifecycle_event'), hook_data.get('function_template')):
                error("unsupported_coordination_topology", "Participant lifecycle callbacks require a supported async owner route; onDeliver is not yet admitted", target)
        adjacency[source].append((target, str(edge.get("id", f"{source}->{target}"))))
    for node in nodes.values():
        data = node.get("data") or {}
        from magic_agents.hooks.target_binding import hook_target_ids
        try:
            targets = set(hook_target_ids(data)) if node.get('type') == 'hook' else set()
        except ValueError:
            error('invalid_coordination_config', 'Invalid Hook target identities', node.get('id'))
            continue
        if targets.intersection(participant_ids) and not invocation:
            if not supported_participant_callback(data.get('lifecycle_event'), data.get('function_template')):
                error("unsupported_coordination_topology", "Participant lifecycle callbacks require a supported async owner route; onDeliver is not yet admitted", node.get("id"))

    for origin in sorted(participant_ids):
        pending = deque([(origin, (origin,), ())]); visited = {origin}
        while pending:
            current, route, route_edges = pending.popleft()
            for target, edge_id in adjacency[current]:
                next_route, next_edges = route + (target,), route_edges + (edge_id,)
                if target in participant_ids:
                    error("coordination_dependency_cycle", "A publication participant depends on another withheld participant output", origin, next_route, next_edges)
                if target not in visited:
                    visited.add(target); pending.append((target, next_route, next_edges))
    return diagnostics


def require_valid_coordination(definition: dict[str, Any], *, invocation=False) -> None:
    diagnostics = validate_coordination_definition(definition, invocation=invocation)
    if diagnostics:
        raise CoordinationValidationError(diagnostics)
