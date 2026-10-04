"""Messaging inside the ordinary graph invocation, with no persistence or host.

Provider clients, tool permissions and Skills stay owned by their existing nodes.
This mode accounts invocation limits; it does not claim financial/job accounting.
"""
from __future__ import annotations

import asyncio
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
