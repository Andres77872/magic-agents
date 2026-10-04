"""One claimed native actor; durable authority is supplied through awaited ports.

This adapter never creates a graph scheduler, coordination service, mailbox or
WorkgroupBudget. Its host owns admission, fencing, effects and canonical storage.
Unsupported composition is rejected before invoking the node, not reconstructed
from an observer transcript. No durable capability is advertised by this module.
"""
from __future__ import annotations

from contextlib import aclosing
from copy import deepcopy
from types import SimpleNamespace
import time
from pydantic import Field

from magic_agents.coordination.service import CoordinationError, LIFECYCLE_CHECKPOINT_KEY
from magic_agents.coordination.skills import CHECKPOINT_KEY, PinnedSkills, _bundle, split_checkpoint
from magic_agents.coordination.tools import _SPECS, _schema, _WaitAgent, _WaitAgentOptions, _WaitMessage, reserve_builtin_names
from magic_llm.agent.control import AgentControlError, parse_timestamp
from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL, ToolExecutor
from magic_llm.agent.types import AgentBudget, AgentBudgetExceeded
from magic_llm.engine.attempt_control import ProviderAttemptControlError


SUPPORTED_TOOLS = frozenset({'listAgents', 'inspectAgent', 'sendMessageToAgent', 'replyToAgent', 'waitForAgent', 'waitForMessage'})


class _DurableWaitOptions(_WaitAgentOptions):
    resumeToken: str | None = Field(default=None, min_length=1, max_length=128,
                                    pattern=r'^[A-Za-z0-9_.:-]+$')


class _DurableWaitAgent(_WaitAgent):
    options: _DurableWaitOptions


class _DurableWaitMessage(_WaitMessage):
    resumeToken: str | None = Field(default=None, min_length=1, max_length=128,
                                    pattern=r'^[A-Za-z0-9_.:-]+$')


def _durable_model(name):
    return {'waitForAgent': _DurableWaitAgent, 'waitForMessage': _DurableWaitMessage}.get(name, _SPECS[name][0])


def canonical_parts(value):
    """Keep composition metadata distinct, never project or silently discard it."""
    retained = deepcopy(value)
    lifecycle = retained.pop(LIFECYCLE_CHECKPOINT_KEY, None)
    loop, skills = split_checkpoint(retained)
    return loop, skills, lifecycle


class DurableActorScope:
    """NodeLLM-compatible scope with one persisted identity and no local authority.

    Awaited ports: guard, inbox, persist, candidate, before/after_attempt, tool,
    loader, authorize_skills and completed_node. restore_lifecycle and
    attach_lifecycle are synchronous validation/copy operations only.
    All operations that can dispatch or mutate are awaited.
    The trusted host supplies failure_type for native protected-error propagation.
    """
    def __init__(self, *, ports, node_path, role, tools, retained, skills_manifest,
                 max_input_tokens, deadline, resource_limits, subordinate=False):
        if (type(subordinate) is not bool or not node_path
                or not subordinate and not tools or set(tools) - SUPPORTED_TOOLS
                or subordinate and tools):
            raise CoordinationError('durable_restore_incompatible', 'Unsupported durable native tool composition')
        self.ports = ports
        self.subordinate = subordinate
        self.path, self.node_id = tuple(node_path[:-1]), node_path[-1]
        self.roles = {self.node_id: role}
        self.runtime = SimpleNamespace(public_source=None)
        self.tools, self.saved = tuple(tools), deepcopy(retained)
        self.deadline, self.resource_limits = deadline, dict(resource_limits)
        self.retained = None
        self.skills = None
        if retained is not None:
            self.retained, private_skills, lifecycle = canonical_parts(retained)
            if lifecycle is not None:
                self.ports.restore_lifecycle(lifecycle)
            if private_skills is not None:
                self.skills = PinnedSkills.restore(private_skills, self.retained)
        if skills_manifest is not None:
            admitted = PinnedSkills(_bundle(skills_manifest), max_input_tokens)
            if self.skills is not None and admitted.digest != self.skills.digest:
                raise CoordinationError('durable_restore_incompatible', 'Pinned Skills admission changed')
            if retained is not None and self.skills is None:
                raise CoordinationError('durable_restore_incompatible', 'Continuation lost admitted Skills')
            self.skills = self.skills or admitted
        elif self.skills is not None:
            raise CoordinationError('durable_restore_incompatible', 'Continuation acquired unadmitted Skills')
        self.skills_guard = None

    async def prepare_skills(self, node):
        await self.ports.guard()
        if self.skills is None:
            if node._prepare_skills():
                raise CoordinationError('durable_restore_incompatible', 'Unpinned Skills input')
            return False
        await self.ports.authorize_skills(self.skills.authorization_manifest)
        node._skills_bundle = self.skills.bundle
        node._skills_request_guard = node._skills_relay = None
        return bool(node._skills_bundle.skills)

    def validate_client(self, node, client):
        from magic_agents.agt_flow import is_task_subagents_enabled
        if (node.node_id != self.node_id or is_task_subagents_enabled()
                or getattr(client, '_task_executor', None) is not None
                or not callable(getattr(client, 'run_agent_stream_async' if node.stream else 'run_agent_async', None))):
            raise CoordinationError('durable_restore_incompatible', 'Durable actors require the exact native async client')

    def attempt_control(self, node_id):
        if node_id != self.node_id:
            raise CoordinationError('durable_restore_incompatible', 'Foreign actor attempt')
        return self

    async def before_attempt(self, attempt):
        await self.ports.before_attempt(attempt)

    async def after_attempt(self, attempt, outcome):
        await self.ports.after_attempt(attempt, outcome)

    def _snapshot(self, checkpoint):
        value = checkpoint.model_dump(mode='json')
        if self.skills is not None:
            value[CHECKPOINT_KEY] = self.skills.snapshot(checkpoint, self.skills_guard)
        return self.ports.attach_lifecycle(value)

    async def before_turn(self, checkpoint):
        await self.ports.guard()
        return await self.ports.inbox()

    async def checkpoint(self, checkpoint, boundary):
        await self.ports.persist(self._snapshot(checkpoint), boundary)
        self.retained = checkpoint.detached()

    async def finish_candidate(self, checkpoint):
        return await self.ports.candidate()

    def final_candidate(self, node_id):
        return self.retained.output_candidate if self.retained is not None and node_id == self.node_id else None

    def _tool(self, name):
        async def invoke(**arguments):
            parsed = _durable_model(name).model_validate(arguments)
            call = CURRENT_TOOL_CALL.get()
            if call is None or not call.id:
                raise CoordinationError('operation_identity_required', 'Missing native tool-call identity')
            return await self.ports.tool(name, parsed.model_dump(mode='json'), call.id)
        invoke.__name__ = name
        invoke._disable_dedup = invoke._require_complete_output = True
        return invoke

    async def configure_native(self, node, schemas, functions):
        await self.ports.guard()
        names = [schema.get('function', {}).get('name') for schema in schemas if isinstance(schema, dict)]
        reserve_builtin_names([name for name in names if name])
        if set(functions) - ({'skills_load'} if self.skills is not None else set()):
            raise CoordinationError('durable_restore_incompatible', 'Durable external tools need stored effect replay')
        if self.skills is not None:
            self.skills_guard = self.skills.attach_guard(node)
            loader = functions.get('skills_load')
            if loader is None:
                raise CoordinationError('durable_restore_incompatible', 'Pinned Skills loader is missing')
            async def bounded_loader(**arguments):
                call = CURRENT_TOOL_CALL.get()
                if call is None:
                    raise CoordinationError('operation_identity_required', 'Missing Skills call identity')
                await self.ports.authorize_skills(self.skills.authorization_manifest)
                return await self.ports.loader(arguments, call.id, lambda: loader(**arguments))
            functions['skills_load'] = bounded_loader
        for name in self.tools:
            schemas.append({'type': 'function', 'function': {
                'name': name, 'description': _SPECS[name][1],
                'parameters': _durable_model(name).model_json_schema() if name in ('waitForAgent', 'waitForMessage') else _schema(name)}})
            functions[name] = self._tool(name)
        executor = ToolExecutor()
        executor.propagate_errors(CoordinationError, AgentControlError, ProviderAttemptControlError,
                                  AgentBudgetExceeded, PermissionError, self.ports.failure_type)
        executor.exclude_from_dedup(*functions)
        executor.require_complete_output(*functions)
        executor.serialize_tools(*functions)
        now = time.time()
        if self.deadline <= now:
            raise CoordinationError('deadline_exceeded', 'Original durable deadline elapsed')
        # AgentLoop restores durations relative to the ORIGINAL start, not the
        # resume time. Passing deadline-now would subtract elapsed time twice.
        # Authored actor sub-deadlines also remain binding inside the wider
        # workgroup lifetime, including old checkpoints with a looser budget.
        started = parse_timestamp(self.retained.started_at) if self.retained is not None else now
        durations = [self.deadline - started]
        if node._agent_config is not None and node._agent_config.wall_clock_timeout is not None:
            durations.append(node._agent_config.wall_clock_timeout)
        if self.retained is not None:
            if self.retained.budget.wall_clock_timeout is not None:
                durations.append(self.retained.budget.wall_clock_timeout)
            if self.retained.absolute_deadline is not None:
                durations.append(parse_timestamp(self.retained.absolute_deadline) - started)
        duration = min(durations)
        if started + duration <= now:
            raise CoordinationError('deadline_exceeded', 'Original actor deadline elapsed')
        limits = self.resource_limits
        iterations = min(limits['model_attempts'], node._agent_config.max_iterations if node._agent_config else 150)
        options = dict(control=self, provider_attempt_control=self, builtin_todo_tools=False,
            budget=AgentBudget(max_iterations=iterations, max_input_tokens=limits['input_tokens'],
                               max_output_tokens=limits['output_tokens'], wall_clock_timeout=duration))
        if self.retained is not None:
            options['continuation'] = self.retained.detached()
        return options, executor


async def run_claimed_actor(node, scope, chat_log, *, completed_result=None):
    """Invoke only the admitted NodeLLM, withholding all private semantic output."""
    if getattr(node, '_invocation_control', None) is not None or getattr(node, 'tool_mode', False):
        raise CoordinationError('durable_restore_incompatible', 'Lifecycle/child invocation is not qualified for durable restore')
    await scope.ports.guard()
    chat_log.coordination = scope
    outputs = {}
    if completed_result is not None:
        if (not scope.subordinate or scope.tools or scope.skills is not None
                or scope.retained is None or scope.retained.output_candidate is None):
            raise CoordinationError('durable_restore_incompatible', 'Completed replay requires a tool-free subordinate candidate')
        source = node.replay_native_completion(chat_log, completed_result, scope.retained.output_candidate)
    else:
        source = node(chat_log)
    async with aclosing(source) as stream:
        async for item in stream:
            if completed_result is not None:
                await scope.ports.guard()
            # Match InvocationControl's actual output-handle map without
            # exposing private semantic frames to an attached observer.
            if item.get('type') not in ('debug', 'debug_summary'):
                content = item.get('content')
                if isinstance(content, dict) and 'node' in content and 'content' in content:
                    content = content['content']
                outputs[item['type']] = node._safe_value(content)
    if scope.retained is None or scope.retained.output_candidate is None:
        raise CoordinationError('checkpoint_required', 'Actor returned without a canonical final candidate')
    await scope.ports.guard()
    return await scope.ports.completed_node(scope._snapshot(scope.retained), outputs)
