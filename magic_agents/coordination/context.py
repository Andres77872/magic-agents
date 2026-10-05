"""Execution-owned coordination injection; never authored or provider state."""
from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import inspect
import contextvars
import re
from contextlib import aclosing
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from magic_llm.agent.control import AgentLoopCheckpoint, AgentControlError
from magic_llm.agent.types import AgentBudgetExceeded
from magic_llm.engine.attempt_control import ProviderAttemptControlError
from magic_llm.exception.ChatException import RequestValidationError
from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL, ToolExecutor

from magic_agents.coordination.budget import BudgetError, UsageBound, WorkgroupBudget
from magic_agents.coordination.control import ActorLoopControl, BudgetAttemptControl
from magic_agents.coordination.service import CoordinationError, CoordinationService, _json, LIFECYCLE_CHECKPOINT_KEY
from magic_agents.models.coordination import CoordinationLimits


@dataclass(frozen=True)
class _OwnerGuard:
    service: CoordinationService
    caller: object

    def check(self):
        self.service.check_owner_open(self.caller)


_OWNERS = contextvars.ContextVar('coordination_owned_invocation', default=())


class CoordinationRuntime:
    """Trusted host adapters plus the original finite workgroup authority.

    Pricing and resource authorization are host-owned callbacks, mandatory for
    every real attempt. No model or graph can manufacture these callbacks.
    """
    def __init__(self, *, server_limits: CoordinationLimits, authorize: Callable,
                 estimate: Callable, usage: Callable, authorize_attempt: Callable,
                 tool_estimate: Callable | None = None, tool_usage: Callable | None = None,
                 authorize_skills: Callable | None = None,
                 authorize_external: Callable | None = None, external_estimate: Callable | None = None,
                 external_usage: Callable | None = None, invocation=False):
        self.invocation_mode = invocation
        self.server_limits = server_limits.model_copy(deep=True)
        self.authorize, self.estimate, self.usage = authorize, estimate, usage
        self.authorize_attempt = authorize_attempt
        self.authorize_skills = authorize_skills
        self.authorize_external, self.external_estimate, self.external_usage = authorize_external, external_estimate, external_usage
        from magic_agents.coordination.events import PrivateEventJournal
        self.events = PrivateEventJournal()
        self.tool_estimate, self.tool_usage = tool_estimate, tool_usage
        self.budget = WorkgroupBudget(self.server_limits, invocation=invocation)
        self.scopes: list[CoordinationScope] = []
        self.workgroup_id = f'group-{self.budget.id}'
        self.epoch_id = f'epoch-{uuid4().hex}'
        self._root = None

    def enter_graph(self, graph, *, parent=None, path=()):
        self.authorize()
        if parent is None:
            if self._root is not None:
                raise CoordinationError('runtime_already_used', 'An admitted runtime owns one root graph invocation')
        scope_type = getattr(self, "scope_type", CoordinationScope)
        scope = scope_type(self, graph, parent=parent, path=path)
        self.scopes.append(scope)
        if parent is None: self._root = scope
        return scope


@dataclass
class _Binding:
    caller: object
    control: ActorLoopControl | None = None


class CoordinationScope:
    def __init__(self, runtime, graph, *, parent=None, path=()):
        self.runtime, self.path, self.graph = runtime, tuple(path), graph
        self.parent = parent
        self._inherited_owners = tuple(_OWNERS.get())
        self.budget = parent.budget if parent else runtime.budget
        self.service = None
        self.roles = {}
        self.bindings = {}
        self.skills = {}
        self.finished = self.succeeded = False
        self.completed_nodes = set()
        self.child_operations = {}
        self.scope_id = f'scope-{uuid4().hex}'
        self.run_id = None
        policy = getattr(graph, 'coordination', None)
        if policy is not None and policy.enabled:
            if graph.type not in ('graph', 'chat'):
                raise CoordinationError('unsupported_coordination_topology', 'Peer groups require concurrent graph scheduling')
            members = {}
            for nid, node in graph.nodes.items():
                config = getattr(node, 'messaging', None)
                if config is not None and config.enabled:
                    self.roles[nid] = config.role
                    members[config.role] = (self.path + (nid,), config)
            self.service = CoordinationService(policy, members, server_limits=runtime.server_limits,
                                               authorize=runtime.authorize, budget=self.budget,
                                               inherit_limits=self._has_active_ancestor())
            self.budget = self.service.budget
            self.service.epoch_id = runtime.epoch_id
            self.scope_id = self.service.scope_id
        for node_id, node in graph.nodes.items():
            if (getattr(node, '_skills_source_node_ids', ()) or getattr(node, '_skills_source_node_id', None)
                    or getattr(node, 'INPUT_HANDLER_SKILLS', None) in getattr(node, 'inputs', {})):
                if not runtime.invocation_mode and (runtime.authorize_skills is None or node_id not in self.roles):
                    raise CoordinationError('unsupported_coordination_capability',
                        'Coordinated Skills require an actor and a trusted pinned-resource authorizer')
        from magic_agents.util.coordination_validation import supported_participant_callback
        callbacks = [node for node in graph.nodes.values()
                     if set(getattr(node, 'target_node_ids', ())).intersection(self.roles)
                     and getattr(node, 'hook_mode', 'python') != 'messages']
        for edge in graph.edges:
            source = graph.nodes.get(edge.source)
            if (getattr(source, 'lifecycle_event', None) and edge.sourceHandle == getattr(source, 'OUTPUT_HANDLE_CALL', None)
                    and edge.target in self.roles):
                raise CoordinationError('unsupported_coordination_topology', 'Participant children require scheduler-owned activation routing')
            if edge.target in self.roles and edge.hooks and edge.hooks.enabled:
                callbacks.append(graph.nodes.get(edge.hooks.hook_node_id))
        for callback in callbacks:
            if not runtime.invocation_mode and not supported_participant_callback(getattr(callback, 'lifecycle_event', None), getattr(callback, '_function_template', None)):
                raise CoordinationError('unsupported_coordination_topology',
                    'Participant lifecycle callbacks require a plain async function; pre-readiness onDeliver is unsupported')
        self.withheld_descendants = set()
        pending = list(self.roles)
        while pending:
            source = pending.pop()
            for edge in graph.edges:
                if edge.source == source and edge.target not in self.withheld_descendants:
                    self.withheld_descendants.add(edge.target)
                    pending.append(edge.target)
        if self.withheld_descendants.intersection(self.roles):
            raise CoordinationError('coordination_dependency_cycle', 'A participant depends on a withheld participant output')

    def dispatch_session(self, node_id):
        from magic_agents.coordination.dispatch import DispatchSession
        return DispatchSession(self, node_id)

    def bind_execution(self, *, run_id=None, hooks=None):
        # Persistence creates its actual ID during graph start, before nodes run.
        from magic_agents.hooks.persistence import GraphPersistenceHook
        exported = hooks.export_hooks() if hooks is not None else {}
        persisted = {hook.run_id for tier in ('global', 'graph') for hook in exported.get(tier, ())
                     if isinstance(hook, GraphPersistenceHook) and hook.run_id}
        if len(persisted) > 1:
            raise CoordinationError('ambiguous_run_identity', 'Coordination requires one actual run per scope')
        actual = next(iter(persisted), None) or run_id or f'run-{uuid4().hex}'
        if self.run_id is not None and self.run_id != actual:
            raise CoordinationError('run_identity_conflict', 'An admitted scope cannot change execution identity')
        root_id = self.runtime._root.run_id if self.parent is not None else actual
        slots = 0
        if self.service is not None:
            limits = self.service.limits
            # Retain final activation/state, request, job and epoch dispositions.
            slots = (3 * (len(self.roles) + limits.max_wakeups_per_workgroup)
                     + limits.max_accepted_messages + 2 * limits.max_tool_calls + 2)
        self.runtime.events.bind_scope(root_run_id=root_id, run_id=actual,
            workgroup_id=self.runtime.workgroup_id, epoch_id=self.runtime.epoch_id,
            scope_id=self.scope_id, terminal_slots=slots)
        self.run_id = actual
        if self.service is not None:
            self.service.events = self.runtime.events
        return actual

    def _has_active_ancestor(self):
        scope = self.parent
        while scope is not None:
            if scope.service is not None: return True
            scope = scope.parent
        return False

    def record_completed(self, node):
        self.completed_nodes.add(node.node_id)

    def finish_graph(self, *, succeeded):
        self.finished, self.succeeded = True, bool(succeeded)

    async def preserve_on_interruption(self):
        return False

    async def persist_transition(self):
        return None

    async def checkpoint_child_operation(self, entry, *, phase):
        if phase == 'dispatched': return
        self.child_operations[entry['operation_id']] = copy.deepcopy(entry)
        await self.persist_transition()

    def child_operation_result(self, entry):
        if 'result' not in entry:
            raise CoordinationError('invalid_lifecycle_checkpoint', 'Child result requires its original execution store')
        return copy.deepcopy(entry['result'])

    def _child_frame_references(self, journal, children):
        """Detach stored child outcomes; live Hook callback values stay unchanged."""
        result = copy.deepcopy(journal)
        by_id = {entry['operation_id']: entry for entry in children if 'resultRef' in entry}
        for frame in result.get('frames', ()):
            entry = by_id.get(frame.get('id'))
            if entry is None: continue
            retained = self.child_operation_result(entry)
            for key in ('outcome', 'original_outcome'):
                outcome = frame.get(key)
                if (isinstance(outcome, dict) and outcome.get('status') == 'success'
                        and 'content' in outcome and outcome['content'] == retained):
                    frame[key] = {name: value for name, value in outcome.items() if name != 'content'}
                    frame[key]['resultRef'] = copy.deepcopy(entry['resultRef'])
        return result

    def _validate_child_frame_references(self, journal, children):
        by_id = {entry['operation_id']: entry for entry in children}
        for frame in journal.get('frames', ()):
            for key in ('outcome', 'original_outcome'):
                outcome = frame.get(key)
                if not isinstance(outcome, dict) or 'resultRef' not in outcome: continue
                entry = by_id.get(frame.get('id'))
                if (entry is None or outcome.get('status') != 'success' or 'content' in outcome
                        or outcome['resultRef'] != entry.get('resultRef')):
                    raise CoordinationError('invalid_lifecycle_checkpoint', 'Child frame result reference changed')
                self.child_operation_result(entry)

    async def run_child_operation(self, node_id, operation_id, content, operation):
        """Admit one Hook child effect without cloning its budget authority.

        The local journal prevents a repeated invocation identity from issuing
        another effect. Durable recovery must checkpoint this journal through
        its authoritative storage port before dispatch/acknowledgement.
        """
        guard = self.capture_guard(node_id)
        guard()
        if node_id in self.roles:
            raise CoordinationError('unsupported_coordination_topology', 'Hook child participants require scheduler-owned activation')
        if not isinstance(operation_id, str) or not 1 <= len(operation_id) <= 256:
            raise CoordinationError('operation_identity_required', 'Child operations require a bounded runtime identity')
        node = self.graph.nodes.get(node_id)
        if node is None: raise CoordinationError('unknown_ref', 'Child target is outside this invocation scope')
        digest = hashlib.sha256(_json({'node': node_id, 'content': content}, max_bytes=256 * 1024)).hexdigest()
        prior = self.child_operations.get(operation_id)
        if prior is not None:
            owners = self._owners()
            actor_id = owners[-1].caller.actor_id if owners else None
            if prior.get('ownerActorId') != actor_id:
                raise CoordinationError('operation_owner_mismatch', 'Child operation belongs to another initiating actor')
            if prior['digest'] != digest:
                raise CoordinationError('idempotency_conflict', 'Child operation identity has different arguments')
            if prior['state'] == 'completed': return self.child_operation_result(prior)
            raise CoordinationError('operation_unresolved', 'Child operation must reconcile before another dispatch')
        guard()
        if len(self.child_operations) >= 100:
            raise CoordinationError('invocation_limit', 'Child operation journal is bounded per graph invocation')
        from magic_agents.node_system.NodeLLM import NodeLLM
        from magic_agents.node_system.NodeInner import NodeInner
        from magic_agents.coordination.dispatch import dispatches_externally, invocation_charge
        provider_owned = isinstance(node, (NodeLLM, NodeInner))
        dispatch_owned = not provider_owned and dispatches_externally(node)
        identity = hashlib.sha256(repr((self.budget.id, self.path, node_id, operation_id)).encode()).hexdigest()
        path, name = self.path + (node_id,), 'child:' + str(node.node_type)
        invocation_local = self.runtime.invocation_mode and not provider_owned and not dispatch_owned
        if invocation_local:
            await self.budget.charge_tool_call(identity, new_guard=guard)
        elif not provider_owned and not dispatch_owned:
            if self.runtime.tool_estimate is None or self.runtime.tool_usage is None:
                raise CoordinationError('usage_bound_unavailable', 'Child operations require trusted cost and resource adapters')
            estimate = self.runtime.tool_estimate(path, name, (content,), {})
            if not isinstance(estimate, UsageBound) or estimate.tool_calls < 1:
                raise CoordinationError('usage_bound_unavailable', 'Each child operation must reserve its physical allowance')
            await self.budget.reserve(identity, estimate, kind='job', guard=guard)
        entry = {'digest': digest, 'state': 'pending', 'node_id': node_id, 'operation_id': operation_id}
        owners = self._owners()
        if owners:
            entry['ownerActorId'] = owners[-1].caller.actor_id
            entry['ownerActivationId'] = owners[-1].caller.activation_id
        await self.checkpoint_child_operation(entry, phase='prepared')
        await self.checkpoint_child_operation(entry, phase='dispatched')
        actual, uncertain = None, True
        try:
            if invocation_local:
                async with asyncio.timeout(max(0, self.budget.deadline - self.budget.clock())):
                    result = await operation()
            elif dispatch_owned:
                async with invocation_charge(self, identity, guard):
                    result = await operation()
            else:
                result = await operation()
            # Retain detached bounded data, never live clients or tasks.
            import json
            detached = json.loads(_json(result, max_bytes=4 * 1024 * 1024))
            if not provider_owned and not dispatch_owned and not invocation_local:
                actual = self.runtime.tool_usage(path, name, result)
                uncertain = actual is None
            guard()
        except BaseException:
            entry['state'] = 'unknown'
            await self.checkpoint_child_operation(entry, phase='unknown')
            raise
        finally:
            if not provider_owned and not dispatch_owned and not invocation_local:
                try:
                    await self.budget.settle(identity, actual, uncertain=uncertain)
                except BaseException:
                    entry['state'] = 'unknown'
                    await self.checkpoint_child_operation(entry, phase='unknown')
                    raise
        entry.update(state='completed', result=detached)
        await self.checkpoint_child_operation(entry, phase='succeeded')
        return copy.deepcopy(detached)

    def _owners(self):
        owners = []
        for owner in (*self._inherited_owners, *_OWNERS.get()):
            if not any(item.service is owner.service and item.caller is owner.caller for item in owners):
                owners.append(owner)
        return tuple(owners)

    def capture_guard(self, node_id):
        """Freeze original callers; a later activation cannot authorize old work."""
        owners = list(self._owners())
        if node_id in self.roles:
            binding = self.bindings.get(node_id)
            if binding is None:
                raise CoordinationError('activation_required', 'Participant has no admitted activation owner')
            owners.append(_OwnerGuard(self.service, binding.caller))
        def guard():
            self._guard_resources(node_id)
            for owner in owners: owner.check()
        return guard

    def _guard_resources(self, node_id):
        self.runtime.authorize()
        self.budget._open()
        if node_id in self.skills:
            self.runtime.authorize_skills(self.path + (node_id,), self.skills[node_id].authorization_manifest)

    def guard(self, node_id):
        self.capture_guard(node_id)()

    def _canonical_checkpoint(self, node_id, retained):
        """Validate private lifecycle state, then pass only native/Skills state on."""
        snapshot = copy.deepcopy(retained)
        lifecycle = snapshot.pop(LIFECYCLE_CHECKPOINT_KEY, None)
        if lifecycle is None: return snapshot
        binding = self.bindings.get(node_id)
        expected = {'schemaVersion', 'nodeId', 'activationId', 'effectiveOutput', 'invocation',
                    'childOperations', 'canonicalDigest'}
        if (not isinstance(lifecycle, dict) or set(lifecycle) != expected
                or type(lifecycle.get('schemaVersion')) is not int or lifecycle['schemaVersion'] != 1
                or lifecycle['nodeId'] != node_id or not isinstance(lifecycle['activationId'], str)
                or not lifecycle['activationId'] or not isinstance(lifecycle['effectiveOutput'], dict)
                or not isinstance(lifecycle['invocation'], dict)
                or lifecycle['invocation'].get('node_id') != node_id
                or not isinstance(lifecycle['invocation'].get('outcome'), dict)
                or lifecycle['invocation']['outcome'].get('status') != 'success'
                or not isinstance(lifecycle['canonicalDigest'], str)
                or re.fullmatch(r'[a-f0-9]{64}', lifecycle['canonicalDigest']) is None
                or not isinstance(lifecycle['childOperations'], list) or len(lifecycle['childOperations']) > 100
                or binding is None):
            raise CoordinationError('invalid_lifecycle_checkpoint', 'Lifecycle checkpoint is incompatible with this actor')
        _json(lifecycle, max_bytes=self.service._checkpoint_bytes)
        if lifecycle['canonicalDigest'] != hashlib.sha256(_json(snapshot, max_bytes=self.service._checkpoint_bytes)).hexdigest():
            raise CoordinationError('invalid_lifecycle_checkpoint', 'Canonical state changed outside its lifecycle envelope')
        restored = {}
        for entry in lifecycle['childOperations']:
            if (not isinstance(entry, dict) or entry.get('state') != 'completed'
                    or set(entry) not in ({'state','node_id','operation_id','digest','ownerActorId','ownerActivationId','result'},
                                          {'state','node_id','operation_id','digest','ownerActorId','ownerActivationId','resultRef'})
                    or entry.get('ownerActorId') != binding.caller.actor_id
                    or not isinstance(entry.get('ownerActivationId'), str) or not entry['ownerActivationId']
                    or entry.get('node_id') not in self.graph.nodes or entry['node_id'] in self.roles
                    or not isinstance(entry.get('operation_id'), str)
                    or not 1 <= len(entry['operation_id']) <= 256
                    or not isinstance(entry.get('digest'), str)
                    or re.fullmatch(r'[a-f0-9]{64}', entry['digest']) is None):
                raise CoordinationError('invalid_lifecycle_checkpoint', 'Child effect journal is invalid')
            self.child_operation_result(entry)
            prior = self.child_operations.get(entry['operation_id'])
            if prior is not None and prior != entry:
                raise CoordinationError('invalid_lifecycle_checkpoint', 'Completed child effect identity changed')
            if entry['operation_id'] in restored:
                raise CoordinationError('invalid_lifecycle_checkpoint', 'Duplicate child effect identity')
            restored[entry['operation_id']] = copy.deepcopy(entry)
        if len(set(self.child_operations) | set(restored)) > 100:
            raise CoordinationError('invocation_limit', 'Restored child operation journal exceeds the invocation bound')
        self._validate_child_frame_references(lifecycle['invocation'], lifecycle['childOperations'])
        self.child_operations.update(restored)
        return snapshot

    async def prepare_skills(self, node):
        """Resume the pinned private catalog instead of fresh invocation input."""
        from magic_agents.coordination.skills import PinnedSkills, split_checkpoint
        binding = self.bindings.get(node.node_id)
        retained = await self.service.retained_checkpoint(binding.caller) if binding is not None else None
        prior = self.skills.get(node.node_id)
        if retained is not None:
            loop, saved = split_checkpoint(self._canonical_checkpoint(node.node_id, retained))
            if saved is not None:
                restored = PinnedSkills.restore(saved, loop)
                if prior is not None and prior.digest != restored.digest:
                    raise CoordinationError('skills_checkpoint_invalid', 'Pinned Skills manifest changed across continuation')
                self.skills[node.node_id] = prior = restored
            elif prior is not None:
                raise CoordinationError('skills_checkpoint_invalid', 'Continuation lost its private Skills state')
        if prior is not None:
            node._skills_bundle = prior.bundle
            node._skills_request_guard = node._skills_relay = None
            active = bool(node._skills_bundle.skills)
        else:
            active = node._prepare_skills()
            if active:
                if binding is None or self.runtime.authorize_skills is None:
                    raise CoordinationError('unsupported_coordination_capability',
                        'Skills continuation requires a trusted pinned-resource authorizer')
                if retained is not None:
                    raise CoordinationError('skills_checkpoint_invalid', 'A resumed actor cannot acquire an unpinned Skills catalog')
                self.skills[node.node_id] = PinnedSkills(node._skills_bundle, node._max_input_tokens)
        self.guard(node.node_id)
        return active

    def attempt_control(self, node_id):
        path = self.path + (node_id,)
        guard = self.capture_guard(node_id)
        def authorize(attempt):
            guard()
            self.runtime.authorize_attempt(path, attempt)
        return BudgetAttemptControl(self.budget,
            estimate=lambda attempt: self.runtime.estimate(path, attempt),
            usage=lambda attempt, outcome: self.runtime.usage(path, attempt, outcome),
            authorize=authorize)

    def validate_client(self, node, client):
        from magic_agents.agt_flow import is_task_subagents_enabled
        self.guard(node.node_id)
        if is_task_subagents_enabled() or getattr(client, '_task_executor', None) is not None:
            raise CoordinationError('unsupported_coordination_capability', 'Nested task loops require inherited coordination admission')
        if node.node_id in self.roles:
            method = 'run_agent_stream_async' if node.stream else 'run_agent_async'
            if not callable(getattr(client, method, None)):
                raise CoordinationError('unsupported_coordination_capability', 'Participants require the native async loop')

    def _bounded_tool(self, node_id, name, function):
        from magic_agents.coordination.dispatch import dispatches_externally, invocation_charge
        dispatch_owned = dispatches_externally(function)
        if not dispatch_owned and (self.runtime.tool_estimate is None or self.runtime.tool_usage is None):
            raise CoordinationError('usage_bound_unavailable', 'Callable tools require trusted operation bounds and usage adapters')
        if not (inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(getattr(function, '__call__', None))):
            raise CoordinationError('unsupported_coordination_capability', 'Synchronous tools require outstanding-work reconciliation before coordination admission')
        guard = self.capture_guard(node_id)
        @functools.wraps(function)
        async def invoke(*args, **kwargs):
            call = CURRENT_TOOL_CALL.get()
            if call is None:
                raise CoordinationError('operation_identity_required', 'Tool dispatch requires a canonical tool-call identity')
            guard()
            path = self.path + (node_id,)
            identity = hashlib.sha256(repr((self.budget.id, path, call.id)).encode()).hexdigest()
            if dispatch_owned:
                async with invocation_charge(self, identity, guard):
                    return await function(*args, **kwargs)
            estimate = self.runtime.tool_estimate(path, name, args, kwargs)
            if not isinstance(estimate, UsageBound) or estimate.tool_calls < 1:
                raise CoordinationError('usage_bound_unavailable', 'Every physical tool invocation must reserve its tool-call allowance')
            await self.budget.reserve(identity, estimate, kind='job', guard=guard)
            result, actual, uncertain = None, None, True
            try:
                result = await function(*args, **kwargs)
                actual = self.runtime.tool_usage(path, name, result)
                uncertain = actual is None
                guard()
                return result
            finally:
                await self.budget.settle(identity, actual, uncertain=uncertain)
        invoke._disable_dedup = True
        return invoke

    async def configure_native(self, node, schemas, functions):
        """Return invocation-local loop options without modifying shared clients."""
        from magic_agents.coordination.tools import CoordinationTools
        self.guard(node.node_id)
        for name, function in list(functions.items()):
            if name == 'skills_load' and node.node_id in self.skills:
                functions[name] = self._bounded_skills_loader(node.node_id, function)
            else:
                functions[name] = self._bounded_tool(node.node_id, name, function)
        options = {'provider_attempt_control': self.attempt_control(node.node_id), 'builtin_todo_tools': False}
        executor = ToolExecutor()
        if not callable(getattr(executor, 'propagate_errors', None)):
            raise CoordinationError('unsupported_coordination_capability',
                'Coordinated tools require a runtime that preserves protected authority denials')
        executor.propagate_errors(BudgetError, CoordinationError, AgentControlError,
            ProviderAttemptControlError, AgentBudgetExceeded, RequestValidationError, PermissionError)
        binding = self.bindings.get(node.node_id)
        if binding is not None:
            names = [s.get('function', {}).get('name', s.get('name')) for s in schemas if isinstance(s, dict)]
            tools = CoordinationTools(binding.caller, existing_names=[name for name in names if name])
            schemas.extend(copy.deepcopy(tools.tools))
            functions.update(tools.tool_functions)
            tools.configure_executor(executor)
            binding.control = ActorLoopControl(binding.caller)
            retained = await self.service.retained_checkpoint(binding.caller)
            if retained is not None:
                from magic_agents.coordination.skills import split_checkpoint
                options['continuation'], _ = split_checkpoint(self._canonical_checkpoint(node.node_id, retained))
            if node.node_id in self.skills:
                from magic_agents.coordination.skills import SkillsActorLoopControl
                state = self.skills[node.node_id]
                binding.control = SkillsActorLoopControl(binding.caller, state, state.attach_guard(node))
            options['control'] = binding.control
        return options, executor

    def _bounded_skills_loader(self, node_id, function):
        guard = self.capture_guard(node_id)
        @functools.wraps(function)
        async def invoke(*args, **kwargs):
            call = CURRENT_TOOL_CALL.get()
            if call is None:
                raise CoordinationError('operation_identity_required', 'Skills loading requires a canonical tool-call identity')
            guard()
            identity = hashlib.sha256(repr((self.path, node_id, call.id)).encode()).hexdigest()
            await self.budget.charge_tool_call('skills:' + identity,
                guard=lambda: self.runtime.authorize(), new_guard=guard)
            result = await function(*args, **kwargs)
            guard()
            return result
        invoke._disable_dedup = True
        return invoke

    def final_candidate(self, node_id):
        binding = self.bindings.get(node_id)
        if binding is None or binding.control is None or binding.control.retained is None:
            return None
        return binding.control.retained.output_candidate

    async def terminate(self, node_id, *, state='failed', reason='node_failed'):
        if node_id in self.roles:
            await self.service.terminate(self.roles[node_id], state=state, reason=reason)

    async def watch_failure(self):
        """Interrupt admitted compute on group failure or the absolute deadline."""
        async with self.budget.condition:
            while True:
                self.runtime.authorize()
                self.budget._open()
                if self.service is not None:
                    self.service._expire()
                    if self.service._state not in ('open', 'sealed'):
                        raise CoordinationError(self.service._state, 'Coordination group reached a terminal failure')
                try:
                    await asyncio.wait_for(self.budget.condition.wait(),
                        timeout=max(0, self.budget.deadline - self.budget.clock()))
                except TimeoutError:
                    continue

    async def run_node(self, node, chat_log, *, hooks=None, observer=None):
        """A node task retains all activation episodes until its group seals."""
        node_id = node.node_id
        if node_id not in self.roles:
            async with aclosing(node(chat_log, hooks=hooks, observer=observer)) as source:
                async for item in source: yield item
            return
        caller = await self.service.activate(self.roles[node_id])
        if caller is None and self.service._state != 'sealed':
            caller = await self.service.wait_for_activation(self.roles[node_id])
        try:
            while caller is not None:
                self.bindings[node_id] = _Binding(caller)
                retained = await self.service.retained_checkpoint(caller)
                if retained is not None:
                    from magic_agents.coordination.skills import split_checkpoint
                    loop, _ = split_checkpoint(self._canonical_checkpoint(node_id, retained))
                    control = ActorLoopControl(caller)
                    control._last_checkpoint = loop.detached()
                    self.bindings[node_id].control = control
                node._response = None
                node._last_invocation_record = None
                node.outputs.clear()
                owner_token = _OWNERS.set((*self._owners(), _OwnerGuard(self.service, caller)))
                try:
                    async with aclosing(node(chat_log, hooks=hooks, observer=observer)) as source:
                        async for item in source:
                            if item.get('type') in ('content', getattr(node, 'OUTPUT_HANDLE_CONTENT', None)):
                                continue  # peer semantic chunks are private, even at final candidate
                            yield item
                    record = node._last_invocation_record
                    binding = self.bindings[node_id]
                    if binding.control is None or binding.control.retained is None:
                        if record is not None:
                            await self.service.terminate(self.roles[node_id], state='bypassed', reason='lifecycle_short_circuit')
                            raise CoordinationError('lifecycle_short_circuit', 'Lifecycle short circuit has no resumable canonical history')
                        raise CoordinationError('checkpoint_required', 'Participant returned without a complete canonical candidate')
                    children = [entry for entry in self.child_operations.values()
                                if entry.get('ownerActorId') == caller.actor_id]
                    if record is not None or children:
                        if any(entry['state'] != 'completed' for entry in children):
                            raise CoordinationError('lifecycle_child_unresolved', 'Child effects must settle before participant finalization')
                        # Flatten control frames: child nesting must not consume
                        # the JSON depth budget or masquerade as model history.
                        record = record or {'node_id': node_id, 'kind': 'participant_finalization',
                                            'outcome': {'status': 'success'}, 'executed': True}
                        frames, pending = [], list(record.get('child', ()))
                        while pending:
                            frame = pending.pop(0)
                            frames.append({key: value for key, value in frame.items() if key != 'child'})
                            pending.extend(frame.get('child', ()))
                        journal = {key: value for key, value in record.items() if key != 'child'}
                        journal['frames'] = frames
                        journal = self._child_frame_references(journal, children)
                        await self.service.checkpoint_lifecycle(caller, node_id=node_id, output=node.outputs,
                            invocation=journal, child_operations=children)
                    elif self.final_candidate(node_id) is None:
                        raise CoordinationError('checkpoint_required', 'Participant returned without a complete canonical candidate')
                    decision = await self.service.finish(caller, node.outputs)
                finally:
                    _OWNERS.reset(owner_token)
                if decision in ('continue', 'awaiting_reply', 'awaiting_tools'):
                    if decision != 'continue': await self.service.candidate_decision(caller)
                    continue
                caller = await self.service.wait_for_activation(self.roles[node_id])
            sealed = await self.service.seal()
            node.outputs = copy.deepcopy(sealed[self.roles[node_id]]['output'])
            if self.runtime.invocation_mode:
                # Normal graphs expose the final sealed participant through the
                # ordinary LLM content handle; no separate author is required.
                from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel
                value = node.outputs.get(node.OUTPUT_HANDLE_GENERATED)
                if isinstance(value, dict) and 'node' in value and 'content' in value:
                    value = value['content']
                if value is not None:
                    text = value if isinstance(value, str) else __import__('json').dumps(value, ensure_ascii=False)
                    yield node.yield_static(ChatCompletionModel(id='', model='', choices=[
                        ChoiceModel(delta=DeltaModel(content=text), finish_reason='stop')]),
                        content_type=node.OUTPUT_HANDLE_CONTENT)
        except asyncio.CancelledError:
            if not await self.preserve_on_interruption():
                await self.service.cancel()
            raise
        except Exception as error:
            reason = getattr(error, 'code', None) or getattr(error, 'error_code', None) or type(error).__name__
            if isinstance(error, AgentBudgetExceeded) and error.budget_type in {
                    'max_iterations', 'max_input_tokens', 'max_output_tokens', 'wall_clock_timeout'}:
                def number(value):
                    return str(value) if type(value) in (int, float) and -1e18 <= value <= 1e18 else 'unavailable'
                reason = (f'AgentBudgetExceeded:{error.budget_type}:'
                          f'limit={number(error.limit)},current={number(error.current)}')
            await self.terminate(node_id, reason=str(reason)[:128])
            if self.service.policy.failure_policy == 'publish_partial' and self.service._state in ('open', 'sealed'):
                await self.service.wait_for_activation(self.roles[node_id])
                sealed = await self.service.seal()
                node.outputs.clear()
                # A failed participant contributes explicit failure data, never
                # a stale successful output.
                yield node.yield_static({'coordinationFailure': sealed[self.roles[node_id]]['failure']},
                                        content_type=node.OUTPUT_HANDLE_GENERATED)
                return
            raise
        finally:
            self.bindings.pop(node_id, None)

    async def close(self, *, cancelled=False):
        if cancelled and not await self.preserve_on_interruption():
            await self.budget.cancel()
        elif not cancelled:
            await self.persist_transition()
