import copy

import pytest
from pydantic import ValidationError

from magic_agents.models.coordination import CoordinationLimits, CoordinationPolicy, MessagingConfig
from magic_agents.util.coordination_validation import (
    CoordinationValidationError, normalize_definition, require_valid_coordination,
    validate_coordination_definition,
)


def graph():
    return {
        "type": "graph",
        "coordination": {"enabled": True, "allowedParticipants": ["research", "images"]},
        "nodes": [
            {"id": "input", "type": "text", "data": {"text": "prompt"}},
            {"id": "a", "type": "llm", "data": {"messaging": {"enabled": True, "role": "research", "peers": ["images"]}}},
            {"id": "b", "type": "llm", "data": {"messaging": {"enabled": True, "role": "images", "peers": ["research"]}}},
            {"id": "join", "type": "parser", "data": {"template": "done"}},
        ],
        "edges": [
            {"id": "ia", "source": "input", "target": "a", "sourceHandle": "handle-text", "targetHandle": "handle_user_message"},
            {"id": "ib", "source": "input", "target": "b", "sourceHandle": "handle-text", "targetHandle": "handle_user_message"},
            {"id": "aj", "source": "a", "target": "join", "sourceHandle": "handle_generated_content", "targetHandle": "research"},
            {"id": "bj", "source": "b", "target": "join", "sourceHandle": "handle_generated_content", "targetHandle": "images"},
        ],
    }


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), "30", True])
def test_wait_limit_rejects_nonpositive_nonfinite_and_coerced_values(value):
    with pytest.raises(ValidationError):
        CoordinationLimits(maxSingleWaitSeconds=value)


@pytest.mark.parametrize("field,value", [("schemaVersion", 2), ("schemaVersion", True), ("schemaVersion", 1.0), ("enabled", "true"), ("tenant", "other"), ("workgroupId", "spoof")])
def test_authored_policy_is_strict_and_has_no_runtime_authority(field, value):
    with pytest.raises(ValidationError):
        CoordinationPolicy.model_validate({field: value})


@pytest.mark.parametrize("changes", [
    {"role": None}, {"peers": ["images", "images"]},
    {"canWakePeers": ["foreign"]}, {"canWakePeers": ["research"], "peers": ["research"]},
    {"tools": ["arbitraryFunction"]}, {"senderActorId": "foreign"},
])
def test_messaging_binds_narrow_capabilities(changes):
    with pytest.raises(ValidationError):
        MessagingConfig.model_validate({"enabled": True, "role": "research", "peers": ["images"], **changes})


def test_finite_durable_horizon_cannot_extend_original_deadline():
    with pytest.raises(ValidationError):
        CoordinationPolicy(lifetime="durable", continuation={"retentionSeconds": 600, "wakeHorizonSeconds": 300})
    with pytest.raises(ValidationError):
        CoordinationPolicy(continuation={"retentionSeconds": 240, "wakeHorizonSeconds": 240})
    with pytest.raises(ValidationError):
        CoordinationPolicy(lifetime="durable", continuation={"retentionSeconds": 120, "wakeHorizonSeconds": 240})


def test_server_policy_is_required_and_cannot_be_expanded():
    requested = CoordinationLimits(maxModelTurns=100, maxCost={"amount": "99.9", "currency": "USD"})
    with pytest.raises(ValueError, match="complete finite"):
        requested.intersect_server_policy(CoordinationLimits())
    server = CoordinationLimits(maxModelTurns=10, maxInputTokens=1000, maxOutputTokens=100,
                                maxToolCalls=20, maxImageJobs=4, maxCost={"amount": "1.10", "currency": "USD"})
    effective = requested.intersect_server_policy(server)
    assert effective.max_model_turns == 10
    assert effective.max_cost.amount == "1.10"
    assert effective.max_input_tokens == 1000
    server.max_cost.currency = "EUR"
    assert effective.max_cost.currency == "USD"
    with pytest.raises(ValueError, match="currencies"):
        requested.intersect_server_policy(server)


def test_independent_peers_share_inputs_and_join_without_mutating_definition():
    definition = graph(); original = copy.deepcopy(definition)
    assert validate_coordination_definition(definition) == []
    assert definition == original
    policy = CoordinationPolicy.model_validate(definition["coordination"])
    assert CoordinationPolicy.model_validate(policy.model_dump(by_alias=True)) == policy


@pytest.mark.parametrize("indirect", [False, True])
def test_barrier_rejects_dependency_with_concrete_node_and_edge_path(indirect):
    definition = graph()
    if indirect:
        definition["edges"].append({"id": "jb", "source": "join", "target": "b"})
    else:
        definition["edges"].append({"id": "ab", "source": "a", "target": "b"})
    errors = validate_coordination_definition(definition)
    match = next(e for e in errors if e.code == "coordination_dependency_cycle" and e.dependency_path[0] == "a")
    assert match.dependency_path == (("a", "join", "b") if indirect else ("a", "b"))
    assert match.edge_path == (("aj", "jb") if indirect else ("ab",))


def test_disabled_ordinary_dependent_graph_is_unchanged():
    definition = graph(); definition["coordination"]["enabled"] = False
    definition["edges"].append({"source": "a", "target": "b"})
    assert validate_coordination_definition(definition) == []


def test_explicitly_disabled_group_retains_dormant_node_settings():
    from magic_agents.agt_flow import build
    from magic_agents.util.coordination_capabilities import require_runtime_coordination
    definition = graph(); definition["coordination"]["enabled"] = False
    built = build(definition, "test")
    assert built.nodes["a"].messaging.enabled
    require_runtime_coordination(built)


@pytest.mark.parametrize("marker", ["unresolved_messaging", "unresolved_hook_target", "unresolved_hook_child_calls"])
def test_editor_draft_references_are_not_silently_stripped(marker):
    definition = graph(); definition["coordination"]["enabled"] = False
    definition["nodes"][1]["data"][marker] = {"reason": "copied_without_target"}
    with pytest.raises(CoordinationValidationError, match="Resolve editor draft"):
        require_valid_coordination(definition)


def test_nested_scopes_with_reused_roles_are_independent():
    assert validate_coordination_definition({"nodes": [
        {"id": "left", "type": "inner", "data": {"magic_flow": graph()}},
        {"id": "right", "type": "inner", "data": {"magic_flow": graph()}},
    ], "edges": []}) == []


@pytest.mark.parametrize("kind", ["skills", "node_tool", "hook"])
def test_unproven_participant_compositions_fail_before_advertising_peer(kind):
    definition = graph()
    definition["nodes"].append({"id": "resource", "type": kind, "data": {}})
    definition["edges"].append({"id": "ra", "source": "resource", "target": "a", "sourceHandle": "handle-child-call" if kind == "hook" else "output"})
    with pytest.raises(CoordinationValidationError) as exc:
        require_valid_coordination(definition)
    assert exc.value.code == "unsupported_coordination_topology"


def test_wrapper_preserves_policy_and_rejects_conflicting_authority():
    definition = graph()
    assert normalize_definition({"content": definition}) == definition
    with pytest.raises(CoordinationValidationError):
        normalize_definition({"content": definition, "coordination": {"enabled": False}})


def test_authored_node_map_is_rejected_before_runtime_build():
    definition = graph()
    definition["nodes"] = {n["id"]: n for n in definition["nodes"]}
    with pytest.raises(CoordinationValidationError, match="must be lists"):
        require_valid_coordination(definition)


@pytest.mark.parametrize("mutation", ["missing_group", "missing_role", "foreign_peer", "duplicate_role", "step"])
def test_invalid_admission_is_not_a_successful_empty_bus(mutation):
    definition = graph()
    if mutation == "missing_group": definition.pop("coordination")
    elif mutation == "missing_role": definition["coordination"]["allowedParticipants"] = ["research"]
    elif mutation == "foreign_peer": definition["nodes"][1]["data"]["messaging"]["peers"] = ["foreign"]
    elif mutation == "duplicate_role": definition["nodes"][2]["data"]["messaging"]["role"] = "research"
    else: definition["type"] = "step"
    assert validate_coordination_definition(definition)


def test_build_retains_coordination_and_runtime_refuses_unavailable_service():
    from magic_agents.agt_flow import build
    from magic_agents.util.coordination_capabilities import require_runtime_coordination
    built = build({"content": graph()}, "test")
    assert built.coordination.allowed_participants == ["research", "images"]
    assert built.nodes["a"].messaging.role == "research"
    with pytest.raises(CoordinationValidationError, match="unsupported_coordination_capability"):
        require_runtime_coordination(built)


@pytest.mark.asyncio
async def test_direct_executor_cannot_silently_run_enabled_configuration():
    from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
    from magic_agents.execution.reactive_executor import execute_graph_reactive, execute_graph_loop_reactive
    built = AgentFlowModel(nodes={}, edges=[], coordination={"enabled": True, "allowedParticipants": ["test"]})
    for execute in (execute_graph_reactive, execute_graph_loop_reactive):
        with pytest.raises(CoordinationValidationError, match="unsupported_coordination_capability"):
            await anext(execute(built))
