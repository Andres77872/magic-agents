"""Native coordinator checkpoints under the existing execution-store CAS.

Only DB scopes retain operational mailboxes. Both engines retain one shared
allowance ledger. This module opens no connections and schedules no work.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
from dataclasses import fields
from decimal import Decimal

from magic_agents.coordination.budget import WorkgroupBudget, UsageBound, _Totals, _Reservation, _RESOURCE_FIELDS
from magic_agents.coordination.service import CoordinationError, _Actor, _Message
from magic_agents.coordination.events import PrivateEventJournal, EventJournalError
from magic_agents.execution.storage import CoordinatorCheckpoint, ExecutionStorageError, canonical_bytes
from magic_agents.models.coordination import CoordinationLimits

_VERSION = 'native-coordination-db/v1'
_EXTENSIONS = ('coordination_skills', 'coordination_lifecycle')
_SETS = {'offered', 'consumed', 'requests', 'activities', 'causal_roots'}
_ACTOR_FIELDS = tuple(f.name for f in fields(_Actor) if f.name not in {'config', 'owner_token', 'checkpoint', 'activity_placeholders', 'output'})


def _invalid():
    raise ExecutionStorageError('coordination_checkpoint_invalid', 'Stored coordination state is incompatible')


def _digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _amounts(value):
    if type(value) is not dict or set(value) != set(_RESOURCE_FIELDS): _invalid()
    result = dict(value)
    try: result['cost'] = Decimal(result['cost'])
    except Exception: _invalid()
    try: UsageBound(**result)
    except (TypeError, ValueError): _invalid()
    return result


def _wire_amounts(value):
    return {name: str(value[name]) if name == 'cost' else value[name] for name in _RESOURCE_FIELDS}


def _usage(value):
    return _wire_amounts({name: getattr(value, name) for name in _RESOURCE_FIELDS})


class CommittedJournal(PrivateEventJournal):
    """Expose existing metadata only after its coordinator CAS is confirmed."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._published_sequence = self._published_lost = 0
        self._published = ()

    def publish(self):
        self._published_sequence, self._published_lost = self._sequence, self._lost_through
        self._published = tuple(sorted((*self._ordinary, *self._terminal), key=lambda row: row[0]))
        self.changed.set()

    @property
    def sequence(self):
        return self._published_sequence

    def read(self, *, after=0):
        if type(after) is not int or not 0 <= after <= self._published_sequence:
            raise EventJournalError('invalid_event_cursor')
        if after < self._published_lost:
            raise EventJournalError('event_retention_unavailable')
        import json
        return tuple(json.loads(encoded) for sequence, encoded in self._published if sequence > after)

    async def wait_after(self, sequence):
        while self._published_sequence <= sequence:
            self.changed.clear()
            if self._published_sequence > sequence: break
            await self.changed.wait()


class CommitCondition(asyncio.Condition):
    """One root lock serializes mutation and the awaited storage commit.

    A wait commits before releasing the lock. A failed/unknown write makes this
    owner unusable, including paths which would otherwise return a receipt.
    """
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    async def __aenter__(self):
        await super().__aenter__()
        if self.owner.failure is not None:
            self.release()
            raise self.owner.failure
        return self

    async def __aexit__(self, *args):
        try: await self.owner.commit()
        finally: self.release()

    async def wait(self):
        await self.owner.commit()
        return await super().wait()


class CoordinatorPersistence:
    def __init__(self, runtime, core):
        self.runtime, self.core = runtime, core
        self.failure = None
        self.scopes, self.budgets, self.scope_budgets = {}, {}, {}
        self.saved_scopes, self.canonical_committed = {}, {}
        self.last_digest = None
        self.pending_child_effects = {}
        runtime.budget._shared.condition = CommitCondition(self)
        previous = runtime.events
        runtime.events = CommittedJournal(max_events=previous.max_events,
            max_bytes=previous.max_bytes, record_bytes=previous.record_bytes)
        self.budgets[runtime.budget.id] = runtime.budget
        saved = core.recorder.state.checkpoint.get('coordination_ledger')
        if saved is not None: self._restore_ledger(saved)
        self.saved_entries = {entry.scope_instance_id: entry for entry in core.recorder.state.coordinator_state}

    def _restore_ledger(self, saved):
        if type(saved) is not dict or saved.get('version') != _VERSION: _invalid()
        rows = saved.get('budgets')
        if type(rows) is not list or not 1 <= len(rows) <= 1024: _invalid()
        root = self.runtime.budget
        restored = {}
        for row in rows:
            if type(row) is not dict or row.get('id') in restored: _invalid()
            limits = CoordinationLimits.model_validate(row['limits'])
            parent = restored.get(row['parent']) if row['parent'] is not None else None
            if parent is None:
                if restored or row['id'] != saved['root_budget'] or limits != root._limits: _invalid()
                budget = root
            else:
                budget = WorkgroupBudget(limits, _parent=parent, absolute_deadline=row['deadline'])
            if type(row['deadline']) not in (int, float) or row['deadline'] > budget.deadline: _invalid()
            budget.id, budget.deadline = row['id'], row['deadline']
            totals = row['totals']
            if any(type(totals.get(k)) is not int or totals[k] < 0 for k in ('model_turns', 'active_models', 'active_jobs')): _invalid()
            budget._totals = _Totals(**{k: _amounts(totals[k]) for k in ('spent', 'reserved', 'uncertain')},
                **{k: totals[k] for k in ('model_turns', 'active_models', 'active_jobs')})
            restored[budget.id] = budget
        self.budgets = restored
        self.runtime.workgroup_id, self.runtime.epoch_id = saved['workgroup_id'], saved['epoch_id']
        shared = root._shared
        for name in ('cancelled', 'exceeded'):
            if type(saved[name]) is not bool: _invalid()
            setattr(shared, name, saved[name])
        for name in ('queue_order', 'priority_streak'):
            if type(saved[name]) is not int or saved[name] < 0: _invalid()
            setattr(shared, name, saved[name])
        for name in ('tool_operations', 'tool_operation_dispatches'):
            pairs = saved[name]
            if type(pairs) is not list or any(type(p) is not list or len(p) != 2 or p[0] not in restored or type(p[1]) is not str for p in pairs): _invalid()
            setattr(shared, name, set(map(tuple, pairs)))
        for row in saved['journal']:
            if row['id'] in shared.journal or row['scope'] not in restored: _invalid()
            if row['kind'] not in ('model', 'job') or row['state'] not in ('queued', 'active', 'settled', 'uncertain'): _invalid()
            shared.journal[row['id']] = _Reservation(restored[row['scope']], UsageBound(**_amounts(row['estimate'])),
                row['kind'], row['state'], UsageBound(**_amounts(row['actual'])) if row['actual'] is not None else None,
                row['priority'], row['queue_order'])
        for row in saved['scopes']:
            identity = row['core_scope_instance_id']
            if identity in self.saved_scopes or row['budget_id'] not in restored: _invalid()
            self.saved_scopes[identity] = copy.deepcopy(row)
            self.scope_budgets[identity] = row['budget_id']
        self.runtime.events._sequence = saved['event_sequence']
        self.runtime.events._lost_through = saved['event_sequence']
        self.runtime.events.publish()
        # Delegated budget reservation precedes physical effect intent. Absence
        # (or prepared-only abandonment) proves no dispatch. Release exposure,
        # preserving cumulative admitted model-turn counts and operation IDs.
        effects = {effect.attempt_id: effect for effect in self.core.recorder.state.effects}
        for identity, reservation in shared.journal.items():
            effect = effects.get(identity)
            if reservation.state not in ('active', 'queued'): continue
            if reservation.kind != 'model' or (effect is not None and effect.status not in ('prepared', 'failed')):
                continue
            for scope in reservation.scope._path:
                totals = scope._totals
                for name in _RESOURCE_FIELDS: totals.reserved[name] -= getattr(reservation.estimate, name)
                if reservation.state == 'active': totals.active_models -= 1
                if any(value < 0 for value in totals.reserved.values()) or totals.active_models < 0: _invalid()
            reservation.state, reservation.actual = 'settled', UsageBound()

    def register(self, scope, core, engine):
        if core.instance_id in self.scopes: _invalid()
        saved = self.saved_scopes.get(core.instance_id)
        if saved is not None:
            if tuple(saved['node_path']) != core.path or saved['engine'] != engine: _invalid()
            budget = self.budgets[saved['budget_id']]
            if budget._limits != scope.budget._limits: _invalid()
            scope.budget = budget
            if scope.service is not None:
                scope.service.budget, scope.service.deadline = budget, budget.deadline
                scope.service._condition = budget.condition
            if engine == 'db_persistence' and scope.service is not None:
                entry = self.saved_entries.get(core.instance_id)
                if entry is None or entry.node_path != core.path: _invalid()
                self._restore_scope(scope, core, entry.state)
            elif scope.service is not None:
                self._restore_memory(scope, core, saved)
        else:
            for budget in scope.budget._path: self.budgets[budget.id] = budget
            self.scope_budgets[core.instance_id] = scope.budget.id
        scope._core_scope, scope.messaging_engine = core, engine
        if saved is not None and engine == 'in_memory':
            # Common child effects survive either engine; mailbox state does not.
            scope.child_operations = copy.deepcopy(saved.get('child_operations', {}))
        for entry in scope.child_operations.values():
            if 'resultRef' in entry: scope._retained_child_effect(entry)
        self.scopes[core.instance_id] = scope

    def _restore_memory(self, scope, core, saved):
        # Restore common invocation identity/counters, never volatile queues.
        from magic_agents.execution.recovery import decode_value
        service = scope.service
        rows = saved.get('actors')
        if type(rows) is not list: _invalid()
        configured = {actor.path[-1]: actor.config for actor in service._actors.values()}
        nodes = core.recorder.state.checkpoint['graphs'][core.instance_id]['nodes']
        actors, roles = {}, {}
        for row in rows:
            node_id = row['node_id']
            config = configured.pop(node_id, None)
            if config is None or row['role'] != config.role or row['config_digest'] != _digest(config.model_dump(mode='json')):
                _invalid()
            if row['id'] in actors: _invalid()
            actor = _Actor(row['id'], (*core.path, node_id), config)
            for key in ('sequence', 'wakes', 'output_revision'):
                if type(row[key]) is not int or row[key] < 0: _invalid()
                setattr(actor, key, row[key])
            actor.activation_id, actor.state = row['activation_id'], row['state']
            if actor.state in {'awaiting_reply', 'awaiting_tools'}: actor.state = 'running'
            # A queued wake belonged to lost volatile messages. It must not
            # rerun a completed node. A new explicit message can wake it again.
            if actor.state in {'queued', 'completed'}: actor.state = 'quiescent'
            node = nodes.get(node_id, {})
            retained = node.get('llm_checkpoint')
            if retained is not None:
                actor.checkpoint = {**copy.deepcopy(retained['checkpoint']),
                                    **copy.deepcopy(node.get('coordination_extensions', {}))}
                actor.consumed = set(retained['checkpoint'].get('consumed_message_ids', []))
                service._checkpoint_boundaries[actor.id] = retained['boundary']
                self.canonical_committed[actor.id] = _digest(actor.checkpoint)
            if actor.state == 'quiescent':
                start, end = node.get('node_start', {}), node.get('node_end', {})
                if (start.get('activation_id') != actor.activation_id or not end
                        or start.get('execution_id') != end.get('execution_id')): _invalid()
                actor.output = decode_value(end['outputs'], core)
                if actor.activation_id: actor.activations[actor.activation_id] = 'finished'
            actor.failure_reason = row.get('failure_reason')
            actors[actor.id], roles[config.role] = actor, actor.id
        if configured: _invalid()
        service._actors, service._roles = actors, roles
        service._restored_actors = set(actors)
        service._volatile_state_lost = saved['state'] != 'sealed'
        scope.scope_id = service.scope_id = saved['scope_id']
        service.workgroup_id, service.epoch_id = self.runtime.workgroup_id, self.runtime.epoch_id

    def _canonical(self, actor, core, boundary):
        if actor.checkpoint is None: return None, None
        checkpoint = copy.deepcopy(actor.checkpoint)
        extensions = {key: checkpoint.pop(key) for key in _EXTENSIONS if key in checkpoint}
        ref = {'scope_instance_id': core.instance_id, 'node_id': actor.path[-1], 'digest': _digest(actor.checkpoint)}
        return ref, {'scope_instance_id': core.instance_id, 'node_id': actor.path[-1],
                     'boundary': boundary, 'checkpoint': checkpoint, 'extensions': extensions}

    def _output_ref(self, scope, actor):
        if actor.output is None: return None
        core = scope._core_scope
        node = core.recorder.state.checkpoint['graphs'][core.instance_id]['nodes'].get(actor.path[-1], {})
        start, end = node.get('node_start'), node.get('node_end')
        if not isinstance(start, dict) or not isinstance(end, dict): _invalid()
        if start.get('activation_id') != actor.activation_id or end.get('execution_id') != start.get('execution_id'):
            _invalid()
        ref = {'scope_instance_id': core.instance_id, 'node_id': actor.path[-1],
               'activation_id': actor.activation_id, 'execution_id': end['execution_id'],
               'output_revision': actor.output_revision, 'digest': _digest(end['outputs'])}
        if _digest(self._load_output(core, actor, ref)) != _digest(actor.output): _invalid()
        return ref

    def _load_output(self, core, actor, ref):
        if ref is None: return None
        from magic_agents.execution.recovery import decode_value
        if (set(ref) != {'scope_instance_id', 'node_id', 'activation_id', 'execution_id', 'output_revision', 'digest'}
                or ref['scope_instance_id'] != core.instance_id or ref['node_id'] != actor.path[-1]
                or ref['activation_id'] != actor.activation_id or ref['output_revision'] != actor.output_revision):
            _invalid()
        node = core.recorder.state.checkpoint['graphs'][core.instance_id]['nodes'].get(actor.path[-1], {})
        start, end = node.get('node_start', {}), node.get('node_end', {})
        if (start.get('activation_id') != ref['activation_id']
                or start.get('execution_id') != ref['execution_id']
                or end.get('execution_id') != ref['execution_id']
                or 'outputs' not in end or _digest(end['outputs']) != ref['digest']):
            _invalid()
        return decode_value(end['outputs'], core)

    def _scope_state(self, scope):
        service, core = scope.service, scope._core_scope
        actors, canonical = [], []
        for actor in service._actors.values():
            row = {name: sorted(getattr(actor, name)) if name in _SETS else copy.deepcopy(getattr(actor, name)) for name in _ACTOR_FIELDS}
            row['path'] = list(actor.path)
            row['config_digest'] = _digest(actor.config.model_dump(mode='json'))
            if actor.activity_placeholders or actor.activities: _invalid()
            ref, node = self._canonical(actor, core, getattr(service, '_checkpoint_boundaries', {}).get(actor.id, 'tool_results'))
            row['checkpoint_ref'] = ref
            row['output_ref'] = self._output_ref(scope, actor)
            if node is not None and self.canonical_committed.get(actor.id) != ref['digest']: canonical.append(node)
            actors.append(row)
        messages = [{f.name: list(getattr(item, f.name)) if f.name == 'roots' else copy.deepcopy(getattr(item, f.name)) for f in fields(_Message)} for item in service._messages.values()]
        publication = None
        if service._publication is not None:
            refs = {row['id']: row['output_ref'] for row in actors}
            publication = {}
            for role, value in service._publication.items():
                actor = service._actors[service._roles[role]]
                if value['revision'] != actor.output_revision or _digest(value['output']) != _digest(actor.output):
                    _invalid()
                publication[role] = {'output_ref': refs[actor.id], 'revision': value['revision'],
                    **({'failure': copy.deepcopy(value['failure'])} if 'failure' in value else {})}
        state = {'version': _VERSION, 'scope_id': service.scope_id, 'epoch_id': service.epoch_id,
            'workgroup_id': service.workgroup_id, 'budget_id': service.budget.id,
            'policy_digest': _digest(service.policy.model_dump(mode='json')), 'actors': actors, 'messages': messages,
            'requests': copy.deepcopy(service._requests),
            'operations': [[actor, key, digest.hex(), copy.deepcopy(result)] for (actor, key), (digest, result) in service._operations.items()],
            'accepted': service._accepted, 'reserved_replies': service._reserved_replies, 'wakes': service._wakes,
            'state': service._state, 'publication': publication,
            'waits': copy.deepcopy(getattr(service, '_retained_waits', {})), 'child_operations': copy.deepcopy(scope.child_operations)}
        return state, canonical

    def _restore_scope(self, scope, core, state):
        service = scope.service
        if state.get('version') != _VERSION or state.get('budget_id') != scope.budget.id or state.get('policy_digest') != _digest(service.policy.model_dump(mode='json')): _invalid()
        if state['workgroup_id'] != self.runtime.workgroup_id or state['epoch_id'] != self.runtime.epoch_id: _invalid()
        configured = {actor.path: actor.config for actor in service._actors.values()}
        actors, roles = {}, {}
        for row in state['actors']:
            path = tuple(row['path'])
            config = configured.pop(path, None)
            if config is None or row['config_digest'] != _digest(config.model_dump(mode='json')): _invalid()
            kwargs = {name: set(row[name]) if name in _SETS else copy.deepcopy(row[name]) for name in _ACTOR_FIELDS}
            kwargs['path'] = path
            actor = _Actor(**kwargs, config=config)
            actor.output = self._load_output(core, actor, row['output_ref'])
            if actor.id in actors or actor.activities: _invalid()
            ref = row['checkpoint_ref']
            if ref is not None:
                if ref['scope_instance_id'] != core.instance_id or ref['node_id'] != path[-1]: _invalid()
                node = core.recorder.state.checkpoint['graphs'][core.instance_id]['nodes'][path[-1]]
                retained = node.get('llm_checkpoint')
                if not isinstance(retained, dict): _invalid()
                actor.checkpoint = {**copy.deepcopy(retained['checkpoint']), **copy.deepcopy(node.get('coordination_extensions', {}))}
                if _digest(actor.checkpoint) != ref['digest']: _invalid()
                self.canonical_committed[actor.id] = ref['digest']
                service._checkpoint_boundaries[actor.id] = retained['boundary']
            actors[actor.id], roles[config.role] = actor, actor.id
        if configured: _invalid()
        messages = {}
        for row in state['messages']:
            item = _Message(**{**row, 'roots': tuple(row['roots'])})
            if item.id in messages or item.sender not in actors or item.target not in actors: _invalid()
            messages[item.id] = item
        for actor in actors.values():
            if not (set(actor.inbox) | actor.offered | actor.consumed).issubset(messages): _invalid()
        service._actors, service._roles, service._messages = actors, roles, messages
        service._requests = copy.deepcopy(state['requests'])
        if any(mid not in messages for mid in service._requests.values()): _invalid()
        service._operations = {}
        for actor, key, digest, result in state['operations']:
            if actor not in actors or (actor, key) in service._operations or result['receipt']['messageId'] not in messages: _invalid()
            service._operations[(actor, key)] = (bytes.fromhex(digest), copy.deepcopy(result))
        for name in ('accepted', 'reserved_replies', 'wakes'):
            if type(state[name]) is not int or state[name] < 0: _invalid()
            setattr(service, '_' + name, state[name])
        service._state, service._publication = state['state'], None
        if state['publication'] is not None:
            if service._state != 'sealed' or set(state['publication']) != set(roles): _invalid()
            refs = {row['id']: row['output_ref'] for row in state['actors']}
            publication = {}
            for role, value in state['publication'].items():
                actor = actors[roles[role]]
                if (set(value) - {'output_ref', 'revision', 'failure'}
                        or value['revision'] != actor.output_revision or value['output_ref'] != refs[actor.id]):
                    _invalid()
                if 'failure' in value and value['failure'] != {'state': actor.state, 'reason': actor.failure_reason}:
                    _invalid()
                publication[role] = {'output': copy.deepcopy(actor.output), 'revision': value['revision'],
                    **({'failure': copy.deepcopy(value['failure'])} if 'failure' in value else {})}
            service._publication = publication
        elif service._state == 'sealed':
            _invalid()
        service._retained_waits = copy.deepcopy(state['waits'])
        scope.child_operations = copy.deepcopy(state['child_operations'])
        scope.scope_id = service.scope_id = state['scope_id']
        service.epoch_id, service.workgroup_id = state['epoch_id'], state['workgroup_id']
        service._restored_actors = set(actors)

    def _ledger(self):
        root, scopes = self.runtime.budget, dict(self.saved_scopes)
        for identity, scope in self.scopes.items():
            scopes[identity] = {'core_scope_instance_id': identity, 'node_path': list(scope.path),
                'scope_id': scope.scope_id, 'state': scope.service._state if scope.service else 'sealed' if scope.finished else 'open',
                'budget_id': scope.budget.id, 'engine': scope.messaging_engine,
                **({'child_operations': copy.deepcopy(scope.child_operations)}
                   if scope.messaging_engine == 'in_memory' else {}),
                'actors': [{'id': actor.id, 'node_id': actor.path[-1], 'role': actor.config.role,
                            'activation_id': actor.activation_id, 'state': actor.state,
                            'sequence': actor.sequence, 'wakes': actor.wakes,
                            'output_revision': actor.output_revision, 'failure_reason': actor.failure_reason,
                            'config_digest': _digest(actor.config.model_dump(mode='json'))}
                           for actor in scope.service._actors.values()] if scope.service else []}
        budgets = []
        for budget in self.budgets.values():
            totals = budget._totals
            budgets.append({'id': budget.id, 'parent': budget._path[-2].id if len(budget._path) > 1 else None,
                'limits': budget._limits.model_dump(mode='json'), 'deadline': budget.deadline,
                'totals': {**{key: _wire_amounts(getattr(totals, key)) for key in ('spent', 'reserved', 'uncertain')},
                           **{key: getattr(totals, key) for key in ('model_turns', 'active_models', 'active_jobs')}}})
        shared = root._shared
        return {'version': _VERSION, 'root_budget': root.id, 'workgroup_id': self.runtime.workgroup_id,
            'epoch_id': self.runtime.epoch_id, 'budgets': budgets, 'scopes': list(scopes.values()),
            'journal': [{'id': identity, 'scope': row.scope.id, 'estimate': _usage(row.estimate), 'kind': row.kind,
                         'state': row.state, 'actual': _usage(row.actual) if row.actual is not None else None,
                         'priority': row.priority, 'queue_order': row.queue_order} for identity, row in shared.journal.items()],
            'tool_operations': sorted(map(list, shared.tool_operations)),
            'tool_operation_dispatches': sorted(map(list, shared.tool_operation_dispatches)),
            'cancelled': shared.cancelled, 'exceeded': shared.exceeded,
            'queue_order': shared.queue_order, 'priority_streak': shared.priority_streak,
            'event_sequence': self.runtime.events._sequence}

    async def commit(self):
        if self.failure is not None: raise self.failure
        if not self.scopes: return
        try:
            entries, nodes = [], []
            for scope in self.scopes.values():
                if scope.messaging_engine == 'db_persistence' and scope.service is not None:
                    state, canonical = self._scope_state(scope)
                    entries.append(CoordinatorCheckpoint(scope_instance_id=scope._core_scope.instance_id, node_path=scope.path, state=state))
                    nodes.extend(canonical)
                elif scope.service is not None:
                    # Skills/lifecycle canonical state is common execution
                    # metadata in BOTH modes; no memory mailbox is serialized.
                    for actor in scope.service._actors.values():
                        ref, canonical = self._canonical(actor, scope._core_scope,
                            scope.service._checkpoint_boundaries.get(actor.id, 'input'))
                        if canonical is not None and self.canonical_committed.get(actor.id) != ref['digest']:
                            nodes.append(canonical)
            ledger = self._ledger()
            digest = _digest({'ledger': ledger, 'scopes': [entry.model_dump(mode='json') for entry in entries]})
            effects = tuple(self.pending_child_effects.values())
            if digest == self.last_digest and not nodes and not effects: return
            await self.core.commit_coordination(entries, ledger, canonical_nodes=nodes, effect_records=effects)
            self.pending_child_effects.clear()
            self.last_digest = digest
            self.runtime.events.publish()
            for scope in self.scopes.values():
                if scope.service is not None:
                    for actor in scope.service._actors.values():
                        if actor.checkpoint is not None: self.canonical_committed[actor.id] = _digest(actor.checkpoint)
        except BaseException as error:
            self.failure = error
            self.core.recorder.fail(error)
            raise
