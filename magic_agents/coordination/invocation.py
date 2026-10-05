"""Messaging inside the ordinary graph invocation, with no persistence or host.

Provider clients, tool permissions and Skills stay owned by their existing nodes.
This mode accounts invocation limits; it does not claim financial/job accounting.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import functools
import inspect
import json
from uuid import uuid4

from magic_agents.coordination.budget import UsageBound
from magic_agents.coordination.context import CoordinationRuntime, CoordinationScope
from magic_agents.coordination.service import CoordinationError
from magic_agents.models.coordination import CoordinationLimits


async def owned_thread(function, *args, **kwargs):
    """Keep ordinary Python callback cancellation semantics.

    Cancelling the await does not terminate Python code already running in a
    thread. Messaging must not turn the existing Hook timeout into a join.
    """
    return await asyncio.to_thread(function, *args, **kwargs)


def _nothing(*args):
    return None


def _estimate(path, attempt):
    from magic_llm.model.ModelChat import ModelChat
    from magic_llm.util.tokenizer import from_openai
    chat = ModelChat()
    inputs = chat.num_tokens_from_messages(list(attempt.messages))
    tools = attempt.generation_options.get('tools')
    if tools: inputs += len(from_openai(json.dumps(tools, ensure_ascii=False)))
    outputs = next((attempt.generation_options[key] for key in
                    ('max_completion_tokens', 'max_output_tokens', 'max_tokens')
                    if type(attempt.generation_options.get(key)) is int and attempt.generation_options[key] > 0), 0)
    return UsageBound(input_tokens=inputs, output_tokens=outputs)


def _usage(path, attempt, outcome):
    value = outcome.usage
    if value is None: return None
    def read(name):
        result = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
        return result if type(result) is int and result >= 0 else None
    inputs, outputs = read('prompt_tokens'), read('completion_tokens')
    if inputs is None or outputs is None: return None
    return UsageBound(input_tokens=inputs, output_tokens=outputs)


class InvocationDispatch:
    def __init__(self, scope, node_id):
        self.scope, self.guard = scope, scope.capture_guard(node_id)

    async def call(self, kind, request, operation, *, operation_id=None, admission_guard=None):
        self.guard()
        if admission_guard is not None:
            result = admission_guard()
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result): result.close()
                raise CoordinationError('invalid_external_adapter', 'Existing dispatch guard must be synchronous')
            if result is not None:
                raise CoordinationError('invalid_external_adapter', 'Existing dispatch guard must raise on denial')
        await self.scope.budget.charge_tool_call(operation_id or 'external:' + uuid4().hex,
                                               new_guard=self.guard)
        async with asyncio.timeout(max(0, self.scope.budget.deadline - self.scope.budget.clock())):
            result = await operation()
        self.guard()
        return result


class InvocationScope(CoordinationScope):
    def __init__(self, runtime, graph, *, parent=None, path=()):
        super().__init__(runtime, graph, parent=parent, path=path)
        from magic_agents.execution.recorder import current_scope
        core = current_scope()
        if runtime.persistence is not None and (core is None
                or core.recorder is not runtime.persistence.core.recorder
                or core.graph is not graph or tuple(path) != core.path):
            raise CoordinationError('coordination_core_scope_mismatch',
                'Persistent messaging requires its exact native graph scope and recorder')
        engines = {getattr(node, '_messaging_engine', 'in_memory') for node in graph.nodes.values()
                   if getattr(getattr(node, 'messaging', None), 'enabled', False)}
        engine = next(iter(engines), 'in_memory') if self.service is not None else 'in_memory'
        self._core_scope, self.messaging_engine = core, engine
        if runtime.persistence is not None:
            runtime.persistence.register(self, core, engine)
        elif engine == 'db_persistence':
            raise CoordinationError('coordination_storage_unavailable', 'DB messaging requires native execution storage')

    async def preserve_on_interruption(self):
        if self._core_scope is None: return False
        # Explicit Stop is committed by the native root store. Transport loss
        # preserves operational state for later manual continuation.
        return not await self._core_scope.recorder.user_stopped()

    async def persist_transition(self):
        if self.runtime.persistence is not None:
            async with self.budget.condition: pass

    def _child_effect_identity(self, entry):
        from magic_agents.execution.storage import canonical_bytes
        return hashlib.sha256(canonical_bytes(['hook_child', self._core_scope.instance_id,
            entry['node_id'], entry['operation_id'], entry.get('ownerActorId'),
            entry.get('ownerActivationId')])).hexdigest()

    def _retained_child_effect(self, entry):
        from magic_agents.execution.storage import ExecutionStorageError, canonical_bytes
        ref = entry.get('resultRef')
        expected = self._child_effect_identity(entry)
        effect = next((item for item in self._core_scope.recorder.state.effects
                       if item.effect_id == expected and item.attempt_id == expected), None)
        if (not isinstance(ref, dict) or set(ref) != {'version','effect_id','attempt_id','request_digest','result_digest'}
                or type(ref['version']) is not int or ref['version'] != 1 or ref['effect_id'] != expected or ref['attempt_id'] != expected
                or ref['request_digest'] != entry['digest'] or effect is None or effect.kind != 'hook_child'
                or effect.request_digest != entry['digest'] or effect.result_digest != ref['result_digest']
                or effect.status == 'succeeded' and effect.result_digest != hashlib.sha256(canonical_bytes(effect.result)).hexdigest()
                or effect.cause.node_path != (*self.path, entry['node_id'])
                or effect.cause.actor_id != entry.get('ownerActorId')
                or effect.cause.activation_id != entry.get('ownerActivationId')
                or (entry['state'] == 'completed') != (effect.status == 'succeeded')):
            raise ExecutionStorageError('execution_child_conflict', 'Retained child effect binding changed')
        return effect

    def child_operation_result(self, entry):
        if 'resultRef' not in entry: return super().child_operation_result(entry)
        return copy.deepcopy(self._retained_child_effect(entry).result)

    async def checkpoint_child_operation(self, entry, *, phase):
        if self.runtime.persistence is None:
            return await super().checkpoint_child_operation(entry, phase=phase)
        from magic_agents.execution.recorder import current_node
        from magic_agents.execution.storage import EffectRecord, ExecutionStorageError, canonical_bytes
        persistence, core = self.runtime.persistence, self._core_scope
        identity = self._child_effect_identity(entry)
        async with self.budget.condition:
            previous = self.child_operations.get(entry['operation_id'])
            if phase == 'prepared':
                if previous is not None:
                    raise ExecutionStorageError('execution_child_conflict', 'Child effect already exists')
                active = current_node()
                cause = (active.cause if active is not None else core.recorder.snapshot.cause).model_copy(update={
                    'run_id': core.run_id, 'execution_id': active.execution_id if active is not None else core.execution_id,
                    'node_path': (*self.path, entry['node_id']),
                    'actor_id': entry.get('ownerActorId'), 'activation_id': entry.get('ownerActivationId')})
                effect = EffectRecord(effect_id=identity, attempt_id=identity, kind='hook_child',
                    status='prepared', request_digest=entry['digest'], cause=cause)
            else:
                if previous is None:
                    raise ExecutionStorageError('execution_child_conflict', 'Child effect has no prepared intent')
                effect = self._retained_child_effect(previous)
                effect = effect.model_copy(update={'status': phase})
            stored = copy.deepcopy(entry)
            if phase == 'succeeded':
                value = stored.pop('result')
                effect = effect.model_copy(update={'result': value,
                    'result_digest': hashlib.sha256(canonical_bytes(value)).hexdigest()})
            stored['resultRef'] = {'version': 1, 'effect_id': identity, 'attempt_id': identity,
                'request_digest': effect.request_digest, 'result_digest': effect.result_digest}
            self.child_operations[entry['operation_id']] = stored
            persistence.pending_child_effects[(identity, identity)] = effect

    def dispatch_session(self, node_id):
        return InvocationDispatch(self, node_id)

    def validate_client(self, node, client):
        self.guard(node.node_id)
        if node.node_id in self.roles:
            method = 'run_agent_stream_async' if node.stream else 'run_agent_async'
            if not callable(getattr(client, method, None)):
                raise CoordinationError('unsupported_coordination_capability', 'Messaging requires the native async LLM loop')

    async def prepare_skills(self, node):
        if node.node_id not in self.roles:
            return node._prepare_skills()
        return await super().prepare_skills(node)

    def _bounded_tool(self, node_id, name, function):
        from magic_agents.coordination.dispatch import dispatches_externally
        guard = self.capture_guard(node_id)
        @functools.wraps(function)
        async def invoke(*args, **kwargs):
            guard()
            if not dispatches_externally(function):
                await self.budget.charge_tool_call('tool:' + uuid4().hex, new_guard=guard)
            if inspect.iscoroutinefunction(function) or inspect.iscoroutinefunction(getattr(function, '__call__', None)):
                result = await function(*args, **kwargs)
            else:
                result = await owned_thread(function, *args, **kwargs)
                if inspect.isawaitable(result): result = await result
            guard()
            return result
        return invoke


class InvocationRuntime(CoordinationRuntime):
    scope_type = InvocationScope

    def __init__(self, limits):
        super().__init__(server_limits=limits, authorize=_nothing, estimate=_estimate,
                         usage=_usage, authorize_attempt=self._authorize_attempt,
                         authorize_skills=_nothing, invocation=True)
        from magic_agents.execution.recorder import current_scope
        core = current_scope()
        self.persistence = None
        if core is not None:
            from magic_agents.coordination.persistence import CoordinatorPersistence
            self.persistence = CoordinatorPersistence(self, core)

    def _authorize_attempt(self, path, attempt):
        scope = next(scope for scope in self.scopes if scope.path == path[:-1])
        if scope.budget._limits.max_output_tokens is not None and not _estimate(path, attempt).output_tokens:
            raise CoordinationError('output_limit_unavailable',
                'A configured group token limit requires the LLM client or node to set its output token limit')


def invocation_runtime(graph):
    policies = []
    def collect(current):
        policy = getattr(current, 'coordination', None)
        if policy is not None and policy.enabled:
            engines = {getattr(node, '_messaging_engine', 'in_memory')
                       for node in current.nodes.values()
                       if getattr(getattr(node, 'messaging', None), 'enabled', False)}
            if len(engines) > 1:
                raise CoordinationError('mixed_messaging_engines',
                    'Enabled participants in one scope must use the same messaging engine')
            from magic_agents.execution.recorder import current_scope
            if engines - {'in_memory', 'db_persistence'} or ('db_persistence' in engines and current_scope() is None):
                raise CoordinationError('coordination_storage_unavailable',
                    'DB messaging requires an integrated core execution storage adapter')
            if policy.lifetime != 'attached' or policy.delivery_mode != 'safe_boundary':
                raise CoordinationError('unsupported_capability', 'Normal graph messaging lasts only for the current invocation')
            if policy.limits.max_cost is not None or policy.limits.max_image_jobs is not None:
                raise CoordinationError('unsupported_invocation_policy',
                    'Normal graph messaging cannot enforce maxCost or maxImageJobs')
            policies.append(policy)
        for node in current.nodes.values():
            child = getattr(node, 'inner_graph', None)
            if child is not None: collect(child)
    collect(graph)
    if not policies: return None
    # A root policy is the root allowance. Inner-only messaging uses the maximum
    # requested ceilings as its shared outer envelope; each scope still narrows it.
    root_policy = getattr(graph, 'coordination', None)
    if root_policy is not None and root_policy.enabled:
        limits = root_policy.limits
    else:
        values = {}
        for key in CoordinationLimits.model_fields:
            candidates = [getattr(policy.limits, key) for policy in policies]
            values[key] = max(candidates) if all(value is not None for value in candidates) else None
        limits = CoordinationLimits.model_validate(values)
    return InvocationRuntime(limits)
