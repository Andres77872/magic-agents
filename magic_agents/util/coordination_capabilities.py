"""Loaded implementation capabilities, independent of authored graph JSON."""
from magic_agents.models.coordination import CoordinationCapabilities
from magic_agents.util.coordination_validation import CoordinationDiagnostic, CoordinationValidationError


def coordination_capabilities() -> CoordinationCapabilities:
    # The schema foundation may be saved before a mailbox/scheduler is available.
    # Capability flags advance only alongside tested execution implementations.
    return CoordinationCapabilities()


def require_runtime_coordination(graph, context=None) -> None:
    policy = getattr(graph, "coordination", None)
    active = policy is not None and policy.enabled
    if policy is None:
        active = any(getattr(getattr(node, "messaging", None), "enabled", False)
                     for node in graph.nodes.values())
    if active and context is None:
        raise CoordinationValidationError([CoordinationDiagnostic(
            "unsupported_coordination_capability",
            "The loaded runtime has coordination schemas but no admitted coordination execution context",
        )])
