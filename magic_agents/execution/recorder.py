"""Mandatory native history and checkpoints, outside observer error isolation.

One root owner shares short serialized commits with concurrent node executions.
The stored data is private core state; it must never be copied to trace metadata.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import inspect
import hashlib
import uuid
import time
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime
from functools import wraps

from anyio import CancelScope
from pydantic import BaseModel

from magic_agents.util.const import SYSTEM_EVENT_STREAMING

from .recovery import inspect_recovery
from .storage import (ExecutionCause, ExecutionEvent, ExecutionState,
                      ExecutionStorageError, ExecutionTransition, VisibleOutput, EffectRecord, CoordinatorCheckpoint, canonical_bytes)

_scope = ContextVar('native_execution_scope', default=None)
_node = ContextVar('native_execution_node', default=None)


def current_scope():
    return _scope.get()


def current_node():
    return _node.get()


class CoreRecorder:
    def __init__(self, store, snapshot, stop_requested=None, restore=None):
        self.store, self.snapshot = store, snapshot
        self.stop_requested = stop_requested
        self.restore = restore
        self.state = ExecutionState()
        self.lease = None
        self.lock = asyncio.Lock()
        self.owner = asyncio.current_task()
        self.owner_id = uuid.uuid4().hex
        self.sequence = 0
        self.failure = None
        self.renewal = None
        self.resources = {}
        self.output_offsets = {}

    def fail(self, error, *, boundary=None):
        if self.failure is None:
            self.failure = error if isinstance(error, ExecutionStorageError) else ExecutionStorageError(
                'execution_storage_failed', 'Mandatory execution storage operation failed (' + (boundary or 'storage') + ':' + type(error).__name__ + ')')
        if self.owner is not asyncio.current_task() and not self.owner.done() and not self.owner.cancelling():
            self.owner.cancel()
        return self.failure

    async def start(self):
        try:
            record = self.restore or await self.store.create(self.snapshot)
            if self.restore is None:
                if record.state.status != 'pending' or record.version != 0:
                    raise ExecutionStorageError('execution_already_started', 'Use explicit manual continuation for retained executions')
            else:
                from .recovery import inspect_recovery
                eligible = inspect_recovery(record)
                if not eligible.recoverable:
                    raise ExecutionStorageError(eligible.reason, 'Retained execution is not safely resumable')
                if record.snapshot != self.snapshot:
                    raise ExecutionStorageError('execution_snapshot_conflict', 'Manual continuation must use the original snapshot')
            self.state = record.state
            self.sequence = (record.state.checkpoint or {}).get('producer_sequence', 0)
            self.lease = await self.store.claim(self.snapshot.identity, expected_version=record.version,
                                                owner_id=self.owner_id, lease_seconds=60.0)
        except Exception as error:
            raise self.fail(error) from error
        self.renewal = asyncio.create_task(self._renew(), name='native_execution_lease')

    async def externally_cancelled(self, error):
        if getattr(error, 'code', None) not in {'run_closed', 'version_conflict', 'lease_lost', 'stale_lease'}:
            return False
        try:
            record = await self.store.load(self.snapshot.identity)
        except Exception:
            return False
        if record is None or record.snapshot != self.snapshot or record.state.status != 'cancelled':
            return False
        self.state = record.state
        if self.owner is not asyncio.current_task() and not self.owner.done() and not self.owner.cancelling():
            self.owner.cancel()
        return True

    async def _renew(self):
        try:
            while True:
                await asyncio.sleep(20)
                async with self.lock:
                    self.lease = await self.store.claim(self.snapshot.identity,
                        expected_version=self.lease.version, owner_id=self.owner_id, lease_seconds=60.0)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not await self.externally_cancelled(error):
                self.fail(error)

    async def user_stopped(self):
        if self.stop_requested is None:
            return False
        value = self.stop_requested()
        if inspect.isawaitable(value):
            value = await value
        if type(value) is not bool:
            raise ExecutionStorageError('execution_stop_invalid', 'Stop intent callback must return a boolean')
        return value

    async def close(self):
        if self.renewal is not None:
            self.renewal.cancel()
            await asyncio.gather(self.renewal, return_exceptions=True)

    def encode(self, value, *, resource=None):
        """Versioned typed values; live clients/tools are authored graph references.

        No repr(), object __dict__, or credential-container dump is permitted.
        References are materialized from the original graph by a restore adapter.
        """
        if value is None or type(value) in (bool, int, float, str):
            canonical_bytes(value)
            return {'type': 'value', 'value': value}
        reference = self.resources.get(id(value))
        if reference is not None:
            return {'type': 'resource', 'reference': reference[1]}
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ExecutionStorageError('execution_value_unsupported', 'Native dictionary keys must be strings')
            return {'type': 'dict', 'items': [[key, self.encode(item, resource=({**resource, 'item_path': [*resource.get('item_path', []), key]} if resource else None))] for key, item in value.items()]}
        if type(value) in (list, tuple):
            return {'type': type(value).__name__, 'items': [self.encode(item, resource=({**resource, 'item_path': [*resource.get('item_path', []), index]} if resource else None)) for index, item in enumerate(value)]}
        if isinstance(value, bytes):
            return {'type': 'bytes', 'value': base64.b64encode(value).decode('ascii')}
        if isinstance(value, (datetime, date)):
            return {'type': type(value).__name__, 'value': value.isoformat()}
        from magic_llm.model.ModelChat import ModelChat
        if isinstance(value, ModelChat):
            if value._provider_payload_guard is not None or value._observer_projection is not None:
                raise ExecutionStorageError('execution_value_unsupported', 'Live chat guards require their original native binding')
            return {'type': 'chat', 'messages': self.encode(value.messages),
                    'max_input_tokens': value.max_input_tokens, 'extra_args': self.encode(value.extra_args),
                    'complete_context_required': value.complete_context_required}
        from magic_agents.hooks.flow_hooks import HookContext
        if type(value) is HookContext:
            # Emit helpers and exception objects are live capabilities, never
            # retained values. Record the declared data fields recursively;
            # inputs may contain already bound authored resource references.
            from dataclasses import fields
            context = {item.name: getattr(value, item.name) for item in fields(HookContext)
                       if item.name not in {'emit', 'error'}}
            return {'type': 'hook_context', 'value': self.encode(context, resource=resource)}
        if isinstance(value, BaseModel):
            return {'type': 'model', 'model': type(value).__module__ + '.' + type(value).__qualname__,
                    'value': value.model_dump(mode='json')}
        if resource is not None:
            from magic_llm import MagicLLM
            from magic_agents.skills import SkillPromptBundle
            if isinstance(value, (MagicLLM, SkillPromptBundle)) or callable(value):
                self.resources[id(value)] = (value, resource)
                return {'type': 'resource', 'reference': resource}
        raise ExecutionStorageError('execution_value_unsupported',
                                    'Native value has no canonical codec or authored resource reference')

    async def record(self, scope, kind, data, *, node_id=None, status=None, visible_output=None, effect=None, private_state=None, coordinator_state=None, coordination_ledger=None, canonical_nodes=(), loop_state=None, effect_records=()):
        if self.state.status == 'cancelled':
            raise asyncio.CancelledError('Native execution was explicitly stopped')
        if self.failure is not None:
            raise self.failure
        phase = 'state'
        try:
            async with self.lock:
                checkpoint = copy.deepcopy(self.state.checkpoint or {'schema_version': 1, 'graphs': {}})
                graphs = checkpoint['graphs']
                graph = graphs.setdefault(scope.instance_id, {'node_path': list(scope.path),
                    'run_id': scope.run_id, 'execution_id': scope.execution_id, 'parent_execution_id': scope.parent_execution_id,
                    'nodes': {}, 'deliveries': {}, 'status': 'running'})
                for canonical in canonical_nodes:
                    target = graphs[canonical['scope_instance_id']]['nodes'].setdefault(canonical['node_id'], {})
                    target['llm_checkpoint'] = {'boundary': canonical['boundary'], 'checkpoint': canonical['checkpoint']}
                    target['coordination_extensions'] = canonical.get('extensions', {})
                    active = current_node()
                    if active is not None and active.node_id == canonical['node_id'] and active.active_response_id:
                        target['llm_checkpoint']['physical_attempt_id'] = active.active_response_id
                        if canonical['boundary'] in {'tool_results', 'candidate'}:
                            target.setdefault('confirmed_attempts', {})[active.active_response_id] = True
                    if canonical['boundary'] == 'tool_results':
                        target.pop('llm_response', None)
                if node_id is not None:
                    node = graph['nodes'].setdefault(node_id, {})
                    if kind == 'node_start' and node.get('node_start', {}).get('execution_id') not in (None, data['execution_id']):
                        retained = {key: node[key] for key in ('llm_checkpoint', 'coordination_extensions') if key in node} if data.get('activation_id') is not None and node.get('node_start', {}).get('execution_phase') == data.get('execution_phase') else {}
                        node.clear()
                        node.update(retained)
                    if kind in {'node_start', 'node_output', 'node_end', 'node_error', 'llm_checkpoint', 'llm_response'}:
                        node[kind] = private_state if private_state is not None else data
                    if kind in {'llm_start', 'llm_end'}:
                        node.setdefault('provider_attempts', {})[data['execution_id']] = data.get('status', 'running')
                    if kind == 'llm_response' and private_state.get('physical_attempt_id'):
                        node.setdefault('confirmed_attempts', {})[private_state['physical_attempt_id']] = True
                    if kind == 'llm_checkpoint' and data.get('boundary') in {'tool_results', 'candidate'} and private_state.get('physical_attempt_id'):
                        node.setdefault('confirmed_attempts', {})[private_state['physical_attempt_id']] = True
                    if kind == 'llm_checkpoint' and data.get('boundary') == 'tool_results':
                        node.pop('llm_response', None)
                if kind == 'loop_checkpoint':
                    graph['loop'] = private_state
                if loop_state is not None:
                    graph['loop'] = loop_state
                if coordination_ledger is not None:
                    checkpoint['coordination_ledger'] = coordination_ledger
                if kind == 'delivery':
                    graph['deliveries'][data['edge_id']] = data
                if kind in ('graph_start', 'graph_end'):
                    graph['status'] = data['status']
                if visible_output is not None and visible_output.get('response_id') is not None:
                    checkpoint.setdefault('visible_offsets', {})[visible_output['response_id']] = {
                        'text_offset': visible_output['text_offset'], 'reasoning_offset': visible_output['reasoning_offset']}
                effects = {(item.effect_id, item.attempt_id): item for item in self.state.effects}
                updates = (*effect_records, *((effect,) if effect is not None else ()))
                if (any(not isinstance(item, EffectRecord) for item in updates)
                        or len({(item.effect_id, item.attempt_id) for item in updates}) != len(updates)):
                    raise ExecutionStorageError('execution_effect_conflict', 'Duplicate or invalid effect update')
                for item in updates:
                    effects[(item.effect_id, item.attempt_id)] = item
                checkpoint['producer_sequence'] = self.sequence + 1
                next_state = self.state.model_copy(update={
                    'checkpoint': checkpoint, 'effects': tuple(effects.values()),
                    'coordinator_state': (tuple(item for item in self.state.coordinator_state if item.scope_instance_id not in {entry.scope_instance_id for entry in coordinator_state}) + tuple(coordinator_state) if coordinator_state is not None else self.state.coordinator_state),
                    'status': status or self.state.status,
                    'output_cursor': self.state.output_cursor + int(visible_output is not None),
                })
                output = (VisibleOutput(expected_cursor=self.state.output_cursor,
                    next_cursor=next_state.output_cursor, **visible_output) if visible_output is not None else None)
                invocation = current_node()
                cause = (invocation.cause if invocation is not None else self.snapshot.cause)
                self.sequence += 1
                phase = 'event'
                event = ExecutionEvent(event_id=uuid.uuid4().hex, sequence=self.sequence,
                    kind=kind, cause=cause, data={'scope_instance_id': scope.instance_id,
                        'node_path': list(scope.path), 'node_id': node_id,
                        'run_id': scope.run_id, 'execution_id': scope.execution_id,
                        'parent_run_id': scope.parent_run_id, 'parent_execution_id': scope.parent_execution_id, **data})
                phase = 'transition'
                transition = ExecutionTransition(operation_id=uuid.uuid4().hex,
                    expected_version=self.lease.version, state=next_state, events=(event,), cause=cause, visible_output=output)
                phase = 'commit'
                receipt = await self.store.commit(self.lease, transition)
                phase = 'receipt'
                if (receipt.operation_id != transition.operation_id or
                        receipt.content_digest != transition.content_digest or
                        receipt.identity != self.snapshot.identity or receipt.fence != self.lease.fence):
                    raise ExecutionStorageError('execution_receipt_invalid', 'Storage receipt does not match the committed transition')
                self.lease = self.lease.model_copy(update={'version': receipt.version})
                self.state = next_state
                return next_state.output_cursor
        except Exception as error:
            if await self.externally_cancelled(error):
                raise asyncio.CancelledError('Native execution was explicitly stopped') from error
            raise self.fail(error, boundary=kind + '.' + phase) from error


@dataclass
class NodeInvocation:
    node_id: str
    execution_id: str
    cause: ExecutionCause
    active_response_id: str | None = None
    saved: dict | None = None


@dataclass
class CoreScope:
    recorder: CoreRecorder
    graph: object
    path: tuple
    run_id: str
    execution_id: str
    instance_id: str
    is_root: bool
    parent_run_id: str | None = None
    parent_execution_id: str | None = None
    resumed: bool = False
    execution_phase: str | None = None

    async def record(self, kind, data, **kwargs):
        return await self.recorder.record(self, kind, data, **kwargs)

    async def commit_coordinator(self, state):
        entry = CoordinatorCheckpoint(scope_instance_id=self.instance_id, node_path=self.path, state=state)
        allowed = {item.node_path for item in self.recorder.snapshot.coordination_scopes
                   if item.messaging_engine == 'db_persistence'}
        if self.path not in allowed:
            raise ExecutionStorageError('coordination_engine_conflict', 'Operational state requires a DB scope')
        # Merge inside record's root lock, not before another scope can commit.
        return await self.record('coordinator_checkpoint',
            {'state_digest': hashlib.sha256(canonical_bytes(state)).hexdigest()},
            coordinator_state=(entry,))

    async def commit_coordination(self, checkpoints, ledger, *, canonical_nodes=(), effect_records=()):
        entries = tuple(checkpoints)
        allowed = {item.node_path for item in self.recorder.snapshot.coordination_scopes
                   if item.messaging_engine == 'db_persistence'}
        if any(not isinstance(item, CoordinatorCheckpoint) or item.node_path not in allowed for item in entries):
            raise ExecutionStorageError('coordination_engine_conflict', 'Operational state requires a DB scope')
        if len({item.scope_instance_id for item in entries}) != len(entries):
            raise ExecutionStorageError('coordination_checkpoint_conflict', 'Duplicate operational scope')
        digest = hashlib.sha256(canonical_bytes({'ledger': ledger,
            'scopes': [item.model_dump(mode='json') for item in entries]})).hexdigest()
        return await self.record('coordination_checkpoint', {'state_digest': digest},
            coordinator_state=entries, coordination_ledger=ledger, canonical_nodes=canonical_nodes, effect_records=effect_records)

    def mark_output(self, node_id, chunk):
        if getattr(chunk, '_native_output_owner', None) is self.recorder:
            return chunk
        choices = getattr(chunk, 'choices', None)
        if not choices:
            return chunk
        invocation = current_node()
        identity = ([invocation.execution_id, invocation.active_response_id]
                    if invocation is not None else [node_id, 'settled-output'])
        response_id = hashlib.sha256(canonical_bytes([self.instance_id, node_id, identity])).hexdigest()
        offsets = self.recorder.output_offsets.setdefault(response_id, [0, 0])
        delta = choices[0].delta
        text, reasoning = (getattr(delta, 'content', None) or ''), (getattr(delta, 'reasoning_content', None) or '')
        marker = {'response_id': response_id, 'text_start': offsets[0], 'reasoning_start': offsets[1]}
        offsets[0] += len(text.encode('utf-8')); offsets[1] += len(reasoning.encode('utf-8'))
        marker.update(text_offset=offsets[0], reasoning_offset=offsets[1])
        extras = dict(chunk.extras) if isinstance(chunk.extras, dict) else {}
        extras['native_execution_output'] = marker
        chunk = chunk.model_copy(update={'extras': extras})
        object.__setattr__(chunk, '_native_output_owner', self.recorder)
        return chunk

    def filter_visible_output(self, text, reasoning, marker):
        if marker is None:
            return text, reasoning, {}
        response_id = marker.get('response_id')
        if not isinstance(response_id, str) or len(response_id) != 64:
            raise ExecutionStorageError('execution_output_invalid', 'Invalid native response identity')
        previous = (self.recorder.state.checkpoint or {}).get('visible_offsets', {}).get(response_id, {})
        output = []
        for value, key in ((text, 'text'), (reasoning, 'reasoning')):
            start, end = marker.get(key + '_start'), marker.get(key + '_offset')
            raw = value.encode('utf-8')
            if type(start) is not int or type(end) is not int or start < 0 or end - start != len(raw):
                raise ExecutionStorageError('execution_output_invalid', 'Native output byte offsets do not match content')
            committed = previous.get(key + '_offset', 0)
            if start > committed:
                raise ExecutionStorageError('execution_output_gap', 'Native output skipped an uncommitted interval')
            try:
                output.append(raw[max(0, committed - start):].decode('utf-8'))
            except UnicodeDecodeError as error:
                raise ExecutionStorageError('execution_output_invalid', 'Native output splits a UTF-8 codepoint') from error
        return *output, {'response_id': response_id,
            'text_offset': max(previous.get('text_offset', 0), marker['text_offset']),
            'reasoning_offset': max(previous.get('reasoning_offset', 0), marker['reasoning_offset'])}

    async def commit_visible_output(self, *, assistant_message_id, assistant_text, reasoning,
                                    response_id=None, text_offset=None, reasoning_offset=None):
        payload = {'assistant_message_id': assistant_message_id,
                   'assistant_text': assistant_text, 'reasoning': reasoning,
                   'response_id': response_id, 'text_offset': text_offset, 'reasoning_offset': reasoning_offset}
        metadata = {'assistant_message_id': assistant_message_id,
                    'text_bytes': len(assistant_text.encode('utf-8')),
                    'reasoning_bytes': len(reasoning.encode('utf-8')),
                    'content_digest': hashlib.sha256(canonical_bytes(payload)).hexdigest()}
        return await self.record('visible_output', metadata, visible_output=payload)



class CoreLoopCursor:
    """One current cursor for the executor-owned Loop; no independent runner."""
    def __init__(self, scope, node):
        self.scope, self.node = scope, node
        graph = ((scope.recorder.restore.state.checkpoint or {}).get('graphs', {}).get(scope.instance_id, {})
                 if scope.recorder.restore is not None else {})
        self.restored = copy.deepcopy(graph.get('loop'))
        self.value = copy.deepcopy(self.restored) if self.restored is not None else {
            'schema_version': 1, 'node_id': node.node_id, 'execution_id': uuid.uuid4().hex,
            'phase': 'static', 'next_index': 0, 'aggregate': scope.recorder.encode([]),
            'started_at': None, 'start_time': None, 'items_digest': None,
            'lifecycle': 'new', 'lifecycle_pending': None,
        }
        if self.value['node_id'] != node.node_id or self.value['lifecycle_pending'] is not None:
            raise ExecutionStorageError('loop_checkpoint_unavailable', 'Loop has no safe continuation boundary')
        scope.execution_phase = 'loop:static'

    async def save(self, **changes):
        self.value.update(changes)
        await self.scope.record('loop_checkpoint', {'node_id': self.node.node_id,
            'phase': self.value['phase'], 'next_index': self.value['next_index'],
            'state_digest': hashlib.sha256(canonical_bytes(self.value)).hexdigest()},
            private_state=copy.deepcopy(self.value))

    async def start(self):
        if self.value['lifecycle'] == 'ended':
            return
        span = {'execution_id': self.value['execution_id'], 'parent_execution_id': self.scope.execution_id}
        await self.scope.record('node_start', {'node_type': 'NodeLoop',
            'inputs': self.scope.recorder.encode(self.node.inputs), 'execution_phase': 'loop',
            'resumed': self.value['lifecycle'] == 'started', **span}, node_id=self.node.node_id)

    async def before_lifecycle(self, event):
        await self.save(lifecycle_pending=event)

    async def after_lifecycle(self, event):
        if event == 'end':
            self.value.update(lifecycle_pending=None, lifecycle='ended')
            # Cursor and Loop span finish together, never a terminal cursor
            # paired with an unfinished node after a process interruption.
            await self.scope.record('node_end', {'execution_id': self.value['execution_id'],
                'parent_execution_id': self.scope.execution_id, 'status': 'completed',
                'outputs': self.scope.recorder.encode(self.node.outputs)}, node_id=self.node.node_id,
                loop_state=copy.deepcopy(self.value))
        else:
            await self.save(lifecycle_pending=None, lifecycle='started')

    async def prepare(self, items, max_iterations):
        digest = hashlib.sha256(canonical_bytes(self.scope.recorder.encode(items))).hexdigest()
        if self.value['items_digest'] is not None and (self.value['items_digest'] != digest
                or self.value.get('max_iterations') != max_iterations):
            raise ExecutionStorageError('loop_input_conflict', 'Loop continuation requires its original items and limit')
        if self.value['items_digest'] is None:
            await self.save(items_digest=digest, max_iterations=max_iterations, start_time=time.time())
        from .recovery import decode_value
        aggregate = decode_value(self.value['aggregate'], self.scope)
        if type(self.value['next_index']) is not int or not 0 <= self.value['next_index'] <= min(len(items), max_iterations) or len(aggregate) != self.value['next_index']:
            raise ExecutionStorageError('loop_checkpoint_invalid', 'Loop iteration cursor is inconsistent')
        return aggregate, self.value['start_time'], self.value['next_index']

    async def iteration(self, index):
        self.scope.execution_phase = 'loop:iteration:' + str(index)
        await self.save(phase='iteration', next_index=index)

    async def completed_iteration(self, index, aggregate):
        await self.save(next_index=index + 1, aggregate=self.scope.recorder.encode(aggregate))

    async def post(self):
        self.scope.execution_phase = 'loop:post'
        await self.save(phase='post')


def recorded_graph(function):
    """Wrap both executor entry points without changing their public signature."""
    signature = inspect.signature(function)
    @wraps(function)
    async def invoke(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        graph = bound.arguments['graph']
        runtime = bound.arguments.get('runtime_config')
        parent = current_scope()
        # Reactive-to-Loop delegation belongs to the same graph invocation.
        if parent is not None and parent.graph is graph:
            async with aclosing(function(*args, **kwargs)) as source:
                async for item in source:
                    yield item
            return
        store = getattr(runtime, 'execution_store', None)
        snapshot = getattr(runtime, 'execution_snapshot', None)
        restore = getattr(runtime, 'execution_restore', None)
        if parent is None and store is None and snapshot is None and restore is None:
            async with aclosing(function(*args, **kwargs)) as source:
                async for item in source:
                    yield item
            return
        if parent is None:
            if store is None or snapshot is None:
                raise ExecutionStorageError('execution_storage_required', 'Native recording requires its store and authored snapshot')
            recorder = CoreRecorder(store, snapshot, getattr(runtime, 'execution_stop_requested', None), restore)
            await recorder.start()
            run_id, execution_id = snapshot.identity.run_id, snapshot.identity.root_execution_id
            path = ()
        else:
            recorder = parent.recorder
            node = current_node()
            if node is None:
                raise ExecutionStorageError('execution_cause_missing', 'Nested execution requires an owning node')
            path = (*parent.path, node.node_id)
            run_id = bound.arguments.get('run_id') or uuid.uuid4().hex
            execution_id = uuid.uuid4().hex
        scope = CoreScope(recorder, graph, path, run_id, execution_id, uuid.uuid4().hex, parent is None,
            parent.run_id if parent else None, current_node().execution_id if parent else None)
        if recorder.restore is not None:
            candidates = [(identity, value) for identity, value in recorder.restore.state.checkpoint['graphs'].items()
                          if tuple(value['node_path']) == path and value.get('parent_execution_id') == scope.parent_execution_id]
            if len(candidates) > 1:
                raise ExecutionStorageError('execution_scope_ambiguous', 'Nested continuation scope is not unique')
            if candidates:
                scope.instance_id, saved_graph = candidates[0]
                scope.resumed = True
                scope.run_id, scope.execution_id = saved_graph['run_id'], saved_graph['execution_id']
                run_id = scope.run_id
                if saved_graph.get('status') == 'completed':
                    from .recovery import decode_value
                    # No executor, lifecycle or coordination activation is rerun.
                    # Terminal graph values are references to confirmed node work.
                    for node_id, saved_node in saved_graph.get('nodes', {}).items():
                        if 'node_end' in saved_node:
                            graph.nodes[node_id].outputs = decode_value(saved_node['node_end']['outputs'], scope)
                    for delivery in saved_graph.get('deliveries', {}).values():
                        value = decode_value(delivery['value'], scope)
                        graph.nodes[delivery['target_node_id']].inputs[delivery['handle']] = value
                    result = bound.arguments.get('result')
                    if result is not None:
                        result.update(has_errors=False, failed_nodes=[], node_errors={})
                    return
        token = _scope.set(scope)
        bound.arguments['run_id'] = run_id
        result = bound.arguments.get('result')
        if result is None:
            result = {}
            bound.arguments['result'] = result
        error = None
        try:
            if parent is None and restore is not None:
                for effect in tuple(recorder.state.effects):
                    if effect.kind == 'provider' and effect.status == 'prepared':
                        await scope.record('effect_abandoned', {'effect_id': effect.effect_id, 'reason': 'not_dispatched'},
                            effect=effect.model_copy(update={'status': 'failed'}))
            await scope.record('graph_start', {'status': 'running', 'snapshot_digest': recorder.snapshot.graph_digest, 'resumed': scope.resumed}, status='running' if parent is None else None)
            if parent is None:
                yield {'type': 'execution_started', 'content': {'run_id': run_id,
                    'version': recorder.lease.version, 'output_cursor': recorder.state.output_cursor,
                    'snapshot_digest': recorder.snapshot.graph_digest, 'resumed': restore is not None}}
            async with aclosing(function(*bound.args, **bound.kwargs)) as source:
                async for item in source:
                    yield item
            before_complete = getattr(runtime, 'execution_before_complete', None)
            if parent is None and before_complete is not None and not result.get('has_errors'):
                pending = before_complete(scope)
                if inspect.isawaitable(pending):
                    await pending
        except BaseException as caught:
            error = caught
            raise
        finally:
            with CancelScope(shield=True):
                try:
                    if recorder.failure is None and recorder.state.status != 'cancelled':
                        status = (('cancelled' if await recorder.user_stopped() else 'blocked') if isinstance(error, (asyncio.CancelledError, GeneratorExit)) else
                                  'failed' if error is not None or result.get('has_errors') else 'completed')
                        await scope.record('graph_end', {'status': status,
                            'error_type': type(error).__name__ if error is not None else None},
                            status=status if parent is None else None)
                finally:
                    if parent is None:
                        await recorder.close()
                    _scope.reset(token)
            if recorder.failure is not None:
                raise recorder.failure
    return invoke


def recorded_node(function):
    @wraps(function)
    async def invoke(node, *args, **kwargs):
        scope = current_scope()
        if scope is None:
            async with aclosing(function(node, *args, **kwargs)) as source:
                async for item in source:
                    yield item
            return
        chat_log = args[0] if args else kwargs.get('chat_log')
        coordinator = getattr(chat_log, 'coordination', None)
        binding = getattr(coordinator, 'bindings', {}).get(node.node_id)
        inherited = scope.recorder.snapshot.cause
        caller = getattr(binding, 'caller', None)
        actor = caller.service._actors[caller.actor_id] if caller is not None else None
        trigger = next((caller.service._messages[key] for key in actor.inbox
                        if caller.service._messages[key].state == 'accepted'), None) if actor is not None else None
        woke = actor is not None and bool(actor.activations or actor.wakes)
        cause = (ExecutionCause(kind='agent' if trigger is not None and not trigger.runtime_generated else 'coordinator',
            run_id=scope.run_id, execution_id=scope.execution_id, node_path=(*scope.path, node.node_id),
            actor_id=caller.actor_id, activation_id=caller.activation_id,
            sender_actor_id=trigger.sender if trigger is not None else None,
            message_id=trigger.id if trigger is not None else None,
            request_id=trigger.request_id if trigger is not None else None) if woke else inherited)
        saved = None
        if scope.recorder.restore is not None:
            saved = scope.recorder.restore.state.checkpoint['graphs'].get(scope.instance_id, {}).get('nodes', {}).get(node.node_id)
            if saved and (saved.get('node_start', {}).get('activation_id') != getattr(caller, 'activation_id', None) or saved.get('node_start', {}).get('execution_phase') != scope.execution_phase):
                saved = None
            if saved and 'node_end' in saved:
                from .recovery import decode_value
                node.outputs = decode_value(saved['node_end']['outputs'], scope)
                if 'selected_handle' in saved['node_end']:
                    node.selected_handle = saved['node_end']['selected_handle']
                for handle, content in node.outputs.items():
                    yield {'type': handle, 'content': content}
                return
        execution_id = saved['node_start']['execution_id'] if saved and 'node_start' in saved else uuid.uuid4().hex
        if not woke:
            cause = inherited.model_copy(update={'run_id': scope.run_id, 'execution_id': execution_id, 'node_path': (*scope.path, node.node_id)})
        invocation = NodeInvocation(node.node_id, execution_id, cause, saved=saved)
        token = _node.set(invocation)
        core = scope.recorder
        span = {'execution_id': invocation.execution_id, 'parent_execution_id': scope.execution_id}

        try:
            encoded_inputs = core.encode(node.inputs)
            if saved and 'node_start' in saved:
                from .recovery import decode_value, equivalent_encoded_inputs
                retained_inputs = saved['node_start']['inputs']
                if not equivalent_encoded_inputs(retained_inputs, encoded_inputs):
                    raise ExecutionStorageError('execution_identity_conflict',
                        'Rebuilt inputs differ from the interrupted node inputs')
                # Concurrent edge replay may arrive in a different order. Only
                # this same span/activation/Loop phase reuses its original map.
                node.inputs = decode_value(retained_inputs, scope)
                encoded_inputs = copy.deepcopy(retained_inputs)
            await scope.record('node_start', {'inputs': encoded_inputs,
                'node_type': type(node).__name__, 'execution_phase': scope.execution_phase, 'activation_id': getattr(caller, 'activation_id', None), 'resumed': bool(saved and 'node_start' in saved), **span}, node_id=node.node_id)
            async with aclosing(function(node, *args, **kwargs)) as source:
                async for item in source:
                    kind = item.get('type') if isinstance(item, dict) else None
                    if kind and kind not in ('debug', 'debug_summary', 'loop_progress', SYSTEM_EVENT_STREAMING, getattr(node, 'OUTPUT_HANDLE_CONTENT', None)):
                        reference = {'node_path': [*scope.path, node.node_id], 'handle': kind}
                        value = core.encode(item.get('content'), resource=reference)
                        await scope.record('node_output', {'handle': kind, 'value': value, **span}, node_id=node.node_id)
                    yield item
            await scope.record('node_end', {'outputs': core.encode(node.outputs), 'selected_handle': getattr(node, 'selected_handle', None), 'status': 'completed', **span}, node_id=node.node_id)
        except BaseException as error:
            with CancelScope(shield=True):
                if core.failure is None and core.state.status != 'cancelled':
                    await scope.record('node_error', {'error_type': type(error).__name__,
                        'status': 'cancelled' if isinstance(error, (asyncio.CancelledError, GeneratorExit)) else 'failed',
                        **span}, node_id=node.node_id)
            raise
        finally:
            _node.reset(token)
    return invoke
