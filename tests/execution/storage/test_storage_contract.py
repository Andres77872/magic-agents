"""Pure engine-selection and authoritative storage DTO contracts; no I/O."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from magic_agents.coordination.invocation import invocation_runtime
from magic_agents.coordination.service import CoordinationError
from magic_agents.execution.storage import (
    ExecutionCause, ExecutionIdentity, ExecutionSnapshot, ExecutionState,
    ExecutionTransition, ExecutionRecord, EffectRecord, CoordinatorCheckpoint,
    CoordinationScopeConfig, canonical_bytes,
)
from magic_agents.hooks.messages_hook import messages_scope_engine
from magic_agents.models.coordination import CoordinationPolicy, MessagingConfig
from magic_agents.models.factory.Nodes.HookNodeModel import HookNodeModel
from magic_agents.util.coordination_validation import require_valid_coordination


def graph():
    return {'type': 'graph', 'coordination': {'enabled': True, 'allowedParticipants': ['a', 'b']},
        'nodes': [{'id': name, 'type': 'llm', 'data': {}} for name in ('a', 'b')] + [
            {'id': 'hook-' + name, 'type': 'hook', 'data': {'hook_mode': 'messages',
                'target_node_ids': [name], 'messaging_by_target': {name: {
                    'enabled': True, 'role': name, 'peers': []}}}} for name in ('a', 'b')],
        'edges': []}


def test_engine_default_and_exact_enum():
    raw = graph()['nodes'][2]['data']
    assert HookNodeModel.model_validate(raw).messaging_engine == 'in_memory'
    for invalid in ('database', None, True):
        with pytest.raises(ValidationError):
            HookNodeModel.model_validate({**raw, 'messaging_engine': invalid})
    with pytest.raises(ValidationError):
        HookNodeModel(hook_mode='python', messaging_engine='db_persistence')


def test_mixed_enabled_engines_reject_without_mutating_authored_graph():
    raw = graph(); raw['nodes'][2]['data']['messaging_engine'] = 'db_persistence'
    before = deepcopy(raw)
    with pytest.raises(ValueError, match='same messaging engine'):
        require_valid_coordination(raw, invocation=True)
    assert raw == before
    raw['coordination']['enabled'] = False
    require_valid_coordination(raw, invocation=True)


def test_disabled_binding_ignored_and_inline_binding_is_memory():
    raw = graph(); raw['nodes'][2]['data']['messaging_engine'] = 'db_persistence'
    raw['nodes'][3]['data']['messaging_by_target']['b']['enabled'] = False
    raw['coordination']['allowedParticipants'] = ['a']
    require_valid_coordination(raw, invocation=True)
    assert messages_scope_engine(raw['nodes'], []) == 'db_persistence'
    raw['nodes'][1]['data']['messaging'] = {'enabled': True, 'role': 'b'}
    raw['nodes'].pop()
    raw['coordination']['allowedParticipants'] = ['a', 'b']
    with pytest.raises(ValueError, match='same messaging engine'):
        require_valid_coordination(raw, invocation=True)


def test_db_engine_cannot_silently_run_in_memory():
    node = SimpleNamespace(messaging=MessagingConfig(enabled=True, role='a'),
                           _messaging_engine='db_persistence')
    built = SimpleNamespace(coordination=CoordinationPolicy(enabled=True, allowedParticipants=['a']),
                            nodes={'a': node})
    with pytest.raises(CoordinationError) as denied:
        invocation_runtime(built)
    assert denied.value.code == 'coordination_storage_unavailable'


def snapshot():
    definition = {'nodes': [], 'edges': []}
    return ExecutionSnapshot(identity=ExecutionIdentity(run_id='run-1', root_execution_id='exec-1',
        conversation_id='conversation-1'), graph_definition=definition,
        graph_digest=hashlib.sha256(canonical_bytes(definition)).hexdigest(),
        runtime_revision='native-v1', cause=ExecutionCause(kind='normal'))


def test_snapshot_exact_digest_and_json_round_trip():
    value = snapshot()
    assert ExecutionSnapshot.model_validate_json(value.model_dump_json()) == value
    raw = value.model_dump(); raw['graph_definition']['nodes'].append({'id': 'changed'})
    with pytest.raises(ValidationError, match='Graph digest'):
        ExecutionSnapshot.model_validate(raw)


def test_transition_replay_digest_binds_original_state_and_cause():
    a = ExecutionTransition(operation_id='op-1', expected_version=1,
        state=ExecutionState(status='running', checkpoint={'step': 1}),
        cause=ExecutionCause(kind='coordinator', actor_id='actor-1', causal_event_id='event-1'))
    b = ExecutionTransition.model_validate_json(a.model_dump_json())
    assert a.content_digest == b.content_digest
    changed = b.model_copy(update={'cause': ExecutionCause(kind='agent', execution_id='caller-1')})
    assert a.content_digest != changed.content_digest
    with pytest.raises(ValidationError):
        ExecutionTransition.model_validate({**a.model_dump(), 'expected_version': True})


def test_nonfinite_payload_and_false_schema_version_rejected():
    with pytest.raises(ValidationError):
        ExecutionState(checkpoint={'usage': float('nan')})
    with pytest.raises(ValidationError):
        ExecutionState(schema_version=True)


def test_effect_result_must_match_its_digest():
    with pytest.raises(ValidationError, match='result digest'):
        EffectRecord(effect_id='effect-1', attempt_id='attempt-1', kind='provider', status='succeeded',
            request_digest='a' * 64, result_digest='b' * 64, result={'value': 1},
            cause=ExecutionCause(kind='normal'))


def test_memory_engine_preserves_core_checkpoint_but_not_operational_coordinator_state():
    original = snapshot()
    record = ExecutionRecord(snapshot=original,
        state=ExecutionState(checkpoint={'nodes': {'a': {'state': 'completed', 'output': 'retained'}}}),
        version=1, fence=1)
    assert record.state.checkpoint['nodes']['a']['output'] == 'retained'
    with pytest.raises(ValidationError, match='operational coordinator state'):
        ExecutionRecord(snapshot=original, state=ExecutionState(coordinator_state=(CoordinatorCheckpoint(
                scope_instance_id='scope-1', state={'queue': []}),)),
            version=1, fence=1)
    public = ExecutionIdentity(run_id='run-1', root_execution_id='exec-1', conversation_id=None)
    assert public.conversation_id is None


def test_nested_db_scope_does_not_upgrade_memory_parent():
    original = snapshot().model_copy(update={'coordination_scopes': (
        CoordinationScopeConfig(node_path=(), messaging_engine='in_memory'),
        CoordinationScopeConfig(node_path=('inner',), messaging_engine='db_persistence'))})
    child = CoordinatorCheckpoint(scope_instance_id='scope-child', node_path=('inner',), state={'queue': []})
    record = ExecutionRecord(snapshot=original, state=ExecutionState(coordinator_state=(child,)), version=1, fence=1)
    assert ExecutionRecord.model_validate_json(record.model_dump_json()) == record
    with pytest.raises(ValidationError, match='Only DB scopes'):
        ExecutionRecord(snapshot=original, state=ExecutionState(coordinator_state=(
            child.model_copy(update={'node_path': ()}),)), version=1, fence=1)
