"""Pure inspection and canonical data restoration for manual native continuation."""
from __future__ import annotations
import base64
import importlib
import time
from datetime import date, datetime
from typing import NamedTuple
from .storage import ExecutionStorageError


class RecoveryInspection(NamedTuple):
    recoverable: bool
    reason: str | None
    volatile_coordination_lost: bool


def inspect_recovery(record, *, now_ms=None):
    state = record.state
    if state.status in {'completed', 'cancelled', 'failed'}:
        return RecoveryInspection(False, 'execution_terminal', False)
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if record.owner_id and (record.lease_until_ms or 0) > now_ms:
        return RecoveryInspection(False, 'execution_owned', False)
    checkpoint = state.checkpoint
    if not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != 1 or not isinstance(checkpoint.get('graphs'), dict):
        return RecoveryInspection(False, 'execution_checkpoint_unavailable', False)
    db_paths = {scope.node_path for scope in record.snapshot.coordination_scopes if scope.messaging_engine == 'db_persistence'}
    memory_paths = {scope.node_path for scope in record.snapshot.coordination_scopes if scope.messaging_engine == 'in_memory'}
    ledger_scopes = {row['core_scope_instance_id']: row for row in
                     checkpoint.get('coordination_ledger', {}).get('scopes', [])}
    lost_paths = set()
    graph_values = list(checkpoint['graphs'].values())
    for identity, graph in checkpoint['graphs'].items():
        path = tuple(graph['node_path'])
        if path not in memory_paths or graph['status'] == 'completed':
            continue
        saved = ledger_scopes.get(identity)
        if saved is None or saved.get('engine') != 'in_memory':
            return RecoveryInspection(False, 'coordination_budget_unavailable', True)
        if saved['state'] != 'sealed':
            # These are identities/counters, never mailbox reconstruction.
            if not isinstance(saved.get('actors'), list):
                return RecoveryInspection(False, 'coordination_identity_unavailable', True)
            lost_paths.add(path)
    for effect in state.effects:
        if effect.kind != 'hook_child': continue
        if effect.status in {'prepared', 'dispatched', 'unknown'}:
            return RecoveryInspection(False, 'hook_child_operation_unresolved', bool(lost_paths))
        if effect.status != 'succeeded': continue
        # A confirmed child result does not make its enclosing arbitrary Python
        # Hook safe to re-enter with a newly generated invocation ID.
        witnessed = False
        for graph in graph_values:
            if (graph.get('run_id') != effect.cause.run_id
                    or tuple(graph['node_path']) != effect.cause.node_path[:-1]): continue
            for node in graph.get('nodes', {}).values():
                start, end = node.get('node_start', {}), node.get('node_end', {})
                if (effect.cause.actor_id is None and start.get('execution_id') == effect.cause.execution_id
                        and end.get('execution_id') == effect.cause.execution_id):
                    witnessed = True
                lifecycle = node.get('coordination_extensions', {}).get('coordination_lifecycle', {})
                if lifecycle.get('invocation', {}).get('outcome', {}).get('status') != 'success': continue
                for entry in lifecycle.get('childOperations', ()):
                    ref = entry.get('resultRef', {})
                    if (entry.get('state') == 'completed' and entry.get('ownerActorId') == effect.cause.actor_id
                            and entry.get('ownerActivationId') == effect.cause.activation_id
                            and entry.get('node_id') == effect.cause.node_path[-1]
                            and ref == {'version': 1, 'effect_id': effect.effect_id, 'attempt_id': effect.attempt_id,
                                        'request_digest': effect.request_digest, 'result_digest': effect.result_digest}):
                        witnessed = True
        if not witnessed:
            return RecoveryInspection(False, 'hook_lifecycle_unresolved', bool(lost_paths))
    if any(effect.status in {'dispatched', 'unknown'} and not (effect.kind == 'coordinator'
            and effect.cause.node_path[:-1] in db_paths | lost_paths) for effect in state.effects):
        return RecoveryInspection(False, 'external_effect_unresolved', bool(lost_paths))
    for graph in graph_values:
        loop = graph.get('loop')
        if loop is not None and (loop.get('schema_version') != 1 or loop.get('lifecycle_pending') is not None):
            return RecoveryInspection(False, 'loop_lifecycle_unresolved', bool(lost_paths))
        for node in graph.get('nodes', {}).values():
            if 'node_start' not in node or 'node_end' in node:
                continue
            kind = node['node_start'].get('node_type')
            if kind == 'NodeLLM':
                if set(node.get('provider_attempts', {})) - set(node.get('confirmed_attempts', {})):
                    return RecoveryInspection(False, 'provider_response_unconfirmed', False)
                if 'llm_response' not in node and 'llm_checkpoint' not in node:
                    return RecoveryInspection(False, 'llm_checkpoint_unavailable', False)
            elif kind == 'NodeLoop':
                if loop is None or loop.get('execution_id') != node['node_start'].get('execution_id'):
                    return RecoveryInspection(False, 'loop_checkpoint_unavailable', bool(lost_paths))
            elif kind not in {'NodeUserInput', 'NodeSystemContext', 'NodeParser', 'NodeEND',
                               'NodeClientLLM', 'NodeSkills', 'NodeTool', 'NodeInner'}:
                return RecoveryInspection(False, 'node_recovery_unqualified', False)
    return RecoveryInspection(True, 'coordination_state_lost' if lost_paths else None, bool(lost_paths))


def equivalent_encoded_inputs(left, right):
    """Ignore mapping insertion order only; ordered values remain exact."""
    from .storage import canonical_bytes

    def ordered(value):
        kind = value.get('type')
        if kind == 'dict':
            return {**value, 'items': sorted(
                [[key, ordered(item)] for key, item in value['items']], key=lambda pair: pair[0])}
        if kind in ('list', 'tuple'):
            return {**value, 'items': [ordered(item) for item in value['items']]}
        if kind == 'chat':
            return {**value, 'messages': ordered(value['messages']),
                    'extra_args': ordered(value['extra_args'])}
        return value

    return canonical_bytes(ordered(left)) == canonical_bytes(ordered(right))


def decode_value(value, scope):
    kind = value['type']
    if kind == 'value': return value['value']
    if kind == 'dict': return {key: decode_value(item, scope) for key, item in value['items']}
    if kind in ('list', 'tuple'):
        result = [decode_value(item, scope) for item in value['items']]
        return tuple(result) if kind == 'tuple' else result
    if kind == 'bytes': return base64.b64decode(value['value'], validate=True)
    if kind == 'datetime': return datetime.fromisoformat(value['value'])
    if kind == 'date': return date.fromisoformat(value['value'])
    if kind == 'chat':
        from magic_llm.model.ModelChat import ModelChat
        result = ModelChat(max_input_tokens=value['max_input_tokens'], extra_args=decode_value(value['extra_args'], scope))
        result.messages = decode_value(value['messages'], scope)
        if value['complete_context_required']:
            result.require_complete_context()
        return result
    if kind == 'hook_context':
        from dataclasses import fields
        import warnings
        from magic_agents.hooks.flow_hooks import HookContext
        context = decode_value(value['value'], scope)
        allowed = {item.name for item in fields(HookContext)} - {'emit', 'error'}
        if type(context) is not dict or set(context) != allowed:
            raise ExecutionStorageError('execution_value_unsupported', 'Invalid retained HookContext fields')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return HookContext(**context)
    if kind == 'model':
        name = value['model']
        if not name.startswith(('magic_llm.model.', 'magic_agents.models.')):
            raise ExecutionStorageError('execution_value_unsupported', 'Retained model codec is not a native model')
        module, member = name.rsplit('.', 1)
        cls = getattr(importlib.import_module(module), member)
        return cls.model_validate(value['value'])
    if kind == 'resource':
        ref = value['reference']
        if tuple(ref['node_path'][:-1]) != scope.path:
            raise ExecutionStorageError('execution_resource_unavailable', 'Retained resource belongs to another graph scope')
        node = scope.graph.nodes[ref['node_path'][-1]]
        # These are already-built passive resources. Never call Node.process,
        # Node.__call__, a lifecycle Hook or a provider/tool during rehydration.
        from magic_agents.node_system.NodeClientLLM import NodeClientLLM
        from magic_agents.node_system.NodeSkills import NodeSkills
        if type(node) is NodeClientLLM and ref['handle'] == node.OUTPUT_HANDLE and ref.get('item_path') == ['content']:
            if node.client is None or node._resolve_runtime_engine_model() != (node._current_engine, node._current_model):
                raise ExecutionStorageError('execution_resource_unavailable', 'Retained client requires its original built configuration')
            result = node.client
        elif type(node) is NodeSkills and ref['handle'] == node.OUTPUT_HANDLE and ref.get('item_path') == ['content']:
            result = node._bundle
        else:
            raise ExecutionStorageError('execution_resource_unavailable', 'Resource has no pure native materializer')
        scope.recorder.resources[id(result)] = (result, ref)
        return result
    raise ExecutionStorageError('execution_value_unsupported', 'Unsupported native value codec')


def has_resource(value):
    if isinstance(value, dict):
        return value.get('type') == 'resource' or any(has_resource(item) for item in value.values())
    if isinstance(value, list):
        return any(has_resource(item) for item in value)
    return False


def volatile_scope_lost(scope):
    restore = scope.recorder.restore
    if restore is None:
        return False
    return any(row.get('core_scope_instance_id') == scope.instance_id
               and row.get('engine') == 'in_memory' and row.get('state') != 'sealed'
               for row in (restore.state.checkpoint or {}).get('coordination_ledger', {}).get('scopes', []))
