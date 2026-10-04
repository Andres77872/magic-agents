"""Explicit owned background activities; never detached ordinary tool calls.

The attached journal reserves exposure before returning a handle, dispatches
under separate job permits, and delivers completion as new runtime input.
Unknown external outcomes are retained without retry or false cancellation.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import threading
from dataclasses import dataclass, field, replace
from typing import Callable, Literal
from types import MappingProxyType

from pydantic import Field, JsonValue, ValidationError

from magic_agents.coordination.budget import BudgetError, UsageBound
from magic_agents.coordination.service import ActorCaller, CoordinationError, _id, _json
from magic_agents.coordination.tools import (_Arguments, Identifier, JOB_TOOL_NAMES,
    _operation_key, reserve_builtin_names)
from magic_llm.agent.tool_executor import ToolExecutor


@dataclass(frozen=True)
class ActivityResult:
    result_ref: str
    provider_operation_id: str | None = None

    def __post_init__(self):
        if type(self.result_ref) is not str or not 1 <= len(self.result_ref) <= 1024:
            raise ValueError('An owned bounded result reference is required')
        if self.provider_operation_id is not None and (type(self.provider_operation_id) is not str or
                                                       not 1 <= len(self.provider_operation_id) <= 256):
            raise ValueError('Provider operation identity must be bounded')
        _json(self.__dict__, max_bytes=8192)


@dataclass(frozen=True)
class ActivityAdapter:
    """Trusted per-tool capability, including resource validation and prices.

    run receives a detached JSON argument object and immutable effect identity.
    A cancel callback returns True only with confirmed external cancellation.
    No callback is serialized or passed to the provider as a tool argument.
    """
    validate: Callable
    estimate: Callable
    run: Callable
    usage: Callable
    cancel: Callable | None = None
    # Trusted, synchronous, repeatable authority check; None means success.
    # Called before disclosure, including later cached reads/reconciliation.
    validate_result: Callable | None = None
    # A host must explicitly qualify this profile; async syntax alone is not
    # evidence that an adapter cooperates with cancellation.
    lifetime_profile: str | None = None


class ActivitySupervisor:
    """Finite process-local ownership, including callbacks which will not stop.

    This is not a process-kill supervisor. Quarantine retains tasks/managers and
    denies new dispatch until the callbacks actually exit; it never claims a
    finite join or a confirmed remote stop for an uncooperative callback.
    """
    def __init__(self, *, capacity=256):
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError('Finite activity supervision capacity required')
        self.capacity, self._owned, self._lock = capacity, {}, threading.RLock()

    def assert_dispatch(self):
        with self._lock:
            if any(job.quarantined for _, job in self._owned.values()):
                raise CoordinationError('activity_cleanup_pending', 'Owned callbacks remain quarantined')

    def reserve(self, manager, job):
        with self._lock:
            self.assert_dispatch()
            if len(self._owned) >= self.capacity:
                raise CoordinationError('activity_supervision_capacity', 'Local activity owner capacity exhausted')
            self._owned[job.id] = (manager, job)

    def release(self, job):
        with self._lock:
            self._owned.pop(job.id, None)

    @property
    def pending_count(self):
        with self._lock:
            return len(self._owned)


_ACTIVITY_SUPERVISOR = ActivitySupervisor()


class ActivityFailure(RuntimeError):
    """A trusted adapter's confirmed terminal failure, not an unknown effect."""
    def __init__(self, reason='activity_failed'):
        if type(reason) is not str or not 1 <= len(reason) <= 128:
            raise ValueError('Bounded failure reason required')
        self.code = reason
        super().__init__(reason)


@dataclass
class _Activity:
    id: str
    caller: ActorCaller
    adapter: ActivityAdapter
    tool_name: str
    arguments: dict
    estimate: UsageBound
    request_id: str | None
    priority: str
    state: str = 'queued'
    result: ActivityResult | None = None
    reason: str | None = None
    dispatched: bool = False
    settled: bool = False
    finalizing: bool = False
    task: asyncio.Task | None = field(default=None, repr=False)
    run_task: asyncio.Task | None = field(default=None, repr=False)
    cancel_task: asyncio.Task | None = field(default=None, repr=False)
    quarantined: bool = False
    pending_error: BaseException | None = field(default=None, repr=False)
    pending_result: ActivityResult | None = field(default=None, repr=False)
    cancel_requested: bool = False


class ActivityManager:
    # Qualification ABI: result checks cover finalization, reconciliation and
    # every cached view; teardown does not depend on ordinary read authority.
    result_authority_version = 1
    activity_supervision_version = 1

    def __init__(self, service, adapters: dict[str, ActivityAdapter], *, allowed_tools_by_role: dict[str, set[str]],
                 max_queued=32, cleanup_timeout=5, supervisor=None):
        if service.policy.delivery_mode != 'background_jobs':
            raise CoordinationError('unsupported_capability', 'Activities require admitted background mode')
        if service._activity_manager is not None:
            raise CoordinationError('activity_owner_conflict', 'The service already owns an activity manager')
        if type(max_queued) is not int or not 1 <= max_queued <= 2048 or not adapters:
            raise CoordinationError('invalid_config', 'Activity registry and finite queue bound are required')
        if type(cleanup_timeout) not in (int, float) or not math.isfinite(cleanup_timeout) or not 0 < cleanup_timeout <= 30:
            raise CoordinationError('invalid_config', 'Finite activity cleanup timeout required')
        if supervisor is not None and not isinstance(supervisor, ActivitySupervisor):
            raise CoordinationError('invalid_config', 'Trusted activity supervisor required')
        reserve_builtin_names(adapters)
        for name, adapter in adapters.items():
            if not isinstance(name, str) or not 1 <= len(name) <= 128 or not isinstance(adapter, ActivityAdapter):
                raise CoordinationError('invalid_config', 'Invalid job adapter registration')
            if not inspect.iscoroutinefunction(adapter.run) or (adapter.cancel is not None and not inspect.iscoroutinefunction(adapter.cancel)):
                raise CoordinationError('unsupported_capability', 'Activities require explicit asynchronous adapters')
            if adapter.validate_result is not None and (not callable(adapter.validate_result)
                    or inspect.iscoroutinefunction(adapter.validate_result)):
                raise CoordinationError('invalid_config', 'Activity result authority must be synchronous')
            if adapter.lifetime_profile != 'cooperative-v1':
                raise CoordinationError('unsupported_capability', 'Activity lifetime profile is not qualified')
        if not isinstance(allowed_tools_by_role, dict) or set(allowed_tools_by_role) != set(service._roles):
            raise CoordinationError('invalid_config', 'Explicit activity grants are required for every participant role')
        grants = {}
        for role, names in allowed_tools_by_role.items():
            if not isinstance(names, (list, tuple, set, frozenset)) or any(type(name) is not str or name not in adapters for name in names):
                raise CoordinationError('invalid_config', 'Activity grants must name registered adapters')
            grants[role] = frozenset(names)
        self._allowed_tools_by_role = MappingProxyType(grants)
        self.service, self.adapters, self.max_queued = service, dict(adapters), max_queued
        self._jobs, self._operations, self._cleanup = {}, {}, set()
        self.cleanup_timeout = cleanup_timeout
        self.supervisor = supervisor or _ACTIVITY_SUPERVISOR
        self._closed = False
        self._stopping = False
        service._activity_manager = self

    def bind(self, caller, *, existing_names=()):
        return ActivityTools(self, caller, existing_names=existing_names)

    def _allowed_tools(self, caller):
        actor = self.service._access(caller)
        return self._allowed_tools_by_role[actor.config.role]

    def _authorize_tool(self, caller, name):
        if name not in self._allowed_tools(caller):
            raise CoordinationError('tool_not_authorized', 'Activity tool is not granted to this actor')

    def _job(self, caller, job_id):
        self.service._access(caller)
        job = self._jobs.get(job_id)
        if job is None or job.caller.actor_id != caller.actor_id:
            raise CoordinationError('unknown_job', 'Job is not visible to this actor')
        return job

    @staticmethod
    def _validate_result(job, result):
        if result is None or job.adapter.validate_result is None:
            return
        decision = job.adapter.validate_result(result, job.id)
        if inspect.isawaitable(decision):
            if inspect.iscoroutine(decision): decision.close()
            raise CoordinationError('activity_result_unavailable', 'Activity result authority must be synchronous')
        if decision is not None:
            raise CoordinationError('activity_result_unavailable', 'Activity result authority must return None or raise')

    def _view(self, job):
        self._validate_result(job, job.result)
        result = {'ok': True, 'jobId': job.id, 'state': job.state}
        if job.request_id: result['requestId'] = job.request_id
        if job.result:
            result['resultRef'] = job.result.result_ref
            if job.result.provider_operation_id: result['providerOperationId'] = job.result.provider_operation_id
        if job.reason: result['reason'] = job.reason
        return result

    async def start(self, caller, tool_name, arguments, *, key, request_id=None, priority='normal'):
        if type(key) is not str or not 1 <= len(key) <= 256 or type(arguments) is not dict:
            raise CoordinationError('invalid_arguments', 'Bounded job key and JSON object arguments are required')
        raw = _json(dict(tool=tool_name, arguments=arguments, request=request_id, priority=priority),
                    max_bytes=self.service.limits.max_inline_message_bytes)
        digest = hashlib.sha256(raw).digest()
        async with self.service._condition:
            actor = self.service._access(caller)
            prior = self._operations.get((actor.id, key))
            if prior is not None:
                if prior[0] != digest:
                    raise CoordinationError('idempotency_conflict', 'Job key was used with different arguments')
                return {**copy.deepcopy(prior[2]), 'currentDisposition': self._view(self._jobs[prior[1]])}
            self.service.admit_tool_locked(caller)
            if self._closed: raise CoordinationError('epoch_closed', 'Activity manager is closed')
            self._authorize_tool(caller, tool_name)
            adapter = self.adapters.get(tool_name)
            if adapter is None:
                raise CoordinationError('unsupported_tool', 'Tool is not enabled for background work')
            if adapter.lifetime_profile != 'cooperative-v1':
                raise CoordinationError('unsupported_capability', 'Activity lifetime profile is not qualified')
            if priority not in ('normal', 'peer_request'):
                raise CoordinationError('invalid_priority', 'Invalid activity priority')
            if request_id is not None:
                request = self.service._messages.get(self.service._requests.get(request_id, ''))
                if request is None or request.target != actor.id or request.state not in ('accepted', 'consumed'):
                    raise CoordinationError('unknown_request', 'Activity request is not addressed to this actor')
            elif priority == 'peer_request':
                raise CoordinationError('invalid_priority', 'Peer priority requires an owned incoming request')
            validated = adapter.validate(copy.deepcopy(arguments))
            if type(validated) is not dict:
                raise CoordinationError('invalid_arguments', 'Job validator must return detached JSON arguments')
            _json(validated, max_bytes=self.service.limits.max_inline_message_bytes)
            bound = adapter.estimate(copy.deepcopy(validated))
            if not isinstance(bound, UsageBound) or bound.tool_calls < 1 or bound.messages or bound.wakeups:
                raise CoordinationError('usage_bound_unavailable', 'Activity needs a complete tool/job bound')
            job_id = _id('job')
            job = _Activity(job_id, caller, adapter, tool_name, copy.deepcopy(validated), bound, request_id, priority)
            self.supervisor.reserve(self, job)
            try:
                self.service.admit_activity_locked(caller, job_id)
                self.service.budget.queue_job_locked(job_id, bound, max_queued=self.max_queued, priority=priority)
                try:
                    self.service.admit_activity_locked(caller, job_id, commit=True)
                except BaseException:
                    self.service.budget.rollback_unaccepted_job_locked(job_id)
                    raise
            except BaseException:
                self.supervisor.release(job)
                raise
            self._jobs[job_id] = job
            self._operations[(actor.id, key)] = (digest, job_id, self._view(job))
            job.task = asyncio.create_task(self._run(job), name='owned-activity-' + job_id)
            job.task.add_done_callback(lambda task: self._done(job, task))
            return self._view(job)

    def _done(self, job, task):
        # A task cancelled before its first bytecode never executes a finally.
        # Keep this cleanup owned and await it from close().
        if task.cancelled() and not job.settled and not job.finalizing:
            self._track_cleanup(job, self._finish(job, None, asyncio.CancelledError()))
        elif not task.cancelled():
            task.exception()  # retrieve an internal failure; _finish retains its state
        if job.settled: self.supervisor.release(job)

    def _track_cleanup(self, job, coroutine):
        task = asyncio.create_task(coroutine)
        self._cleanup.add(task)
        def finished(task):
            self._cleanup.discard(task)
            failed = task.cancelled() or task.exception() is not None
            if failed and not job.settled:
                # Do not discard the last owner merely because settlement or
                # a result commit failed. No automatic retry of partial writes.
                job.state, job.reason = 'reconciling', 'activity_settlement_pending'
                job.result = None
                job.quarantined, job.finalizing = True, False
                self.service.budget._shared.cancelled = True
                notification = asyncio.create_task(self.service.budget.cancel())
                self._cleanup.add(notification)
                def notified(task):
                    self._cleanup.discard(task)
                    if not task.cancelled(): task.exception()
                notification.add_done_callback(notified)
        task.add_done_callback(finished)
        return task

    def _owned_callback_done(self, job, task):
        if task.done() and not task.cancelled(): task.exception()
        if job.quarantined and job.reason == 'local_cleanup_unconfirmed' and not job.finalizing and not job.settled and all(
                task is None or task.done() for task in (job.run_task, job.cancel_task)):
            # A callback finishing later releases ownership only through the
            # original manager's settlement path, without dispatch/retry.
            self._track_cleanup(job, self._finish(job, job.pending_result, job.pending_error))

    async def _quarantine(self, job, result, error):
        job.pending_result, job.pending_error = result, error
        job.state, job.reason, job.quarantined = 'reconciling', 'local_cleanup_unconfirmed', True
        # Retain the active/queued reservation and local ownership slot. No
        # settlement may release a physical permit while callbacks remain live.
        await self.service.budget.cancel()
        async with self.service._condition:
            self.service._expire()
            self.service._event('job_state', self.service._actors[job.caller.actor_id],
                                jobId=job.id, state=job.state, reason=job.reason)
            self.service._condition.notify_all()
        job.finalizing = False
        self._owned_callback_done(job, job.run_task or job.cancel_task)

    @staticmethod
    async def _wait_owned(tasks, deadline):
        """Wait a fixed grace without wait_for's unbounded cancellation join."""
        pending = {task for task in tasks if task is not None and not task.done()}
        while pending:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0: break
            try:
                _, pending = await asyncio.wait(pending, timeout=remaining)
            except asyncio.CancelledError:
                # Repeated cancellation does not renew grace or abandon owners.
                pending = {task for task in pending if not task.done()}
        return pending

    async def _run(self, job):
        result, error = None, None
        try:
            await self.service.budget.start_queued_job(job.id, guard=lambda: self.service.check_owner_open(job.caller))
            async with self.service._condition:
                self.service.admit_tool_locked(job.caller)
                self._authorize_tool(job.caller, job.tool_name)
                validated = job.adapter.validate(copy.deepcopy(job.arguments))
                if _json(validated, max_bytes=self.service.limits.max_inline_message_bytes) != _json(job.arguments, max_bytes=self.service.limits.max_inline_message_bytes):
                    raise CoordinationError('activity_arguments_changed', 'Queued activity no longer matches its admitted input')
                self.supervisor.assert_dispatch()
                job.state = 'running'
                self.service._event('job_state', self.service._actors[job.caller.actor_id], jobId=job.id, state='running')
                self.service._condition.notify_all()
            remaining = max(0, self.service.deadline - self.service.clock())
            if not remaining: raise TimeoutError()
            async def dispatch():
                # Creating an owned task is an await boundary; capture and
                # recheck the original actor immediately before actual entry.
                if self._stopping or job.cancel_requested:
                    raise CoordinationError('cancelled', 'Activity dispatch was closed')
                self.service.check_owner_open(job.caller)
                self.supervisor.assert_dispatch()
                job.dispatched = True
                return await job.adapter.run(copy.deepcopy(job.arguments), job.id)
            job.run_task = asyncio.create_task(dispatch(), name='activity-effect-' + job.id)
            job.run_task.add_done_callback(lambda task: self._owned_callback_done(job, task))
            done, _ = await asyncio.wait({job.run_task}, timeout=remaining)
            if not done: raise TimeoutError()
            result = job.run_task.result()
            if not isinstance(result, ActivityResult):
                raise ValueError('Activity adapter must return an owned ActivityResult')
        except BaseException as exc:
            error = exc
        cleanup = self._track_cleanup(job, self._finish(job, result, error))
        # The state commit remains owned even if shutdown cancels this worker
        # again while it is acquiring the short lineage transaction.
        while not cleanup.done():
            try: await asyncio.shield(cleanup)
            except asyncio.CancelledError: continue
        cleanup.result()

    async def _finish(self, job, result, error):
        if job.settled or job.finalizing: return
        job.finalizing = True
        cancelled = isinstance(error, (asyncio.CancelledError, TimeoutError))
        if cancelled: job.cancel_requested = True
        if cancelled and job.run_task is not None and not job.run_task.done():
            job.run_task.cancel()
        if cancelled and job.dispatched and job.adapter.cancel is not None and job.cancel_task is None:
            job.cancel_task = asyncio.create_task(job.adapter.cancel(job.id), name='activity-cancel-' + job.id)
            job.cancel_task.add_done_callback(lambda task: self._owned_callback_done(job, task))
        pending = await self._wait_owned((job.run_task, job.cancel_task),
                                        asyncio.get_running_loop().time() + self.cleanup_timeout)
        if pending:
            for task in pending: task.cancel()
            await self._quarantine(job, result, error)
            return
        if job.run_task is not None and not job.run_task.cancelled():
            try:
                late_result = job.run_task.result()
                if isinstance(late_result, ActivityResult): result = late_result
            except BaseException:
                pass  # Preserve the original dispatch/cancellation failure.
        billed_result = result
        try:
            self._validate_result(job, result)
        except BaseException as exc:
            # Keep the private result only for authoritative billing. It must
            # never enter the job journal or a model-visible completion notice.
            result, error = None, exc
        confirmed = not job.dispatched
        if cancelled and job.cancel_task is not None:
            try: confirmed = job.cancel_task.result() is True
            except BaseException: confirmed = False
        if error is not None:
            # A cancellation-suppressing callback may return a private result
            # late; accounting may use it, but the cancelled actor cannot.
            result = None
        actual = UsageBound() if not job.dispatched else None
        if job.dispatched:
            try: actual = job.adapter.usage(billed_result, error)
            except Exception: actual = None
        if actual is not None and not isinstance(actual, UsageBound): actual = None
        if actual is not None and job.dispatched:
            actual = replace(actual, tool_calls=max(1, actual.tool_calls))
        unknown = error is not None and job.dispatched and not confirmed and not isinstance(error, ActivityFailure)
        try:
            await self.service.budget.settle(job.id, actual, uncertain=actual is None or unknown)
        except BudgetError as exc:
            error = exc
        async with self.service._condition:
            self.service._expire()
            try:
                # Settlement may have waited for the lineage lock while the
                # asset permission changed. Recheck before committing a view.
                self._validate_result(job, result)
            except BaseException as exc:
                result, error, unknown = None, exc, job.dispatched
            job.result = result
            if error is None:
                job.state = 'completed'
            elif isinstance(error, (asyncio.CancelledError, TimeoutError)) and confirmed:
                job.state, job.reason = 'cancelled', 'cancel_confirmed' if job.dispatched else 'cancelled_before_dispatch'
            elif unknown:
                job.state, job.reason = 'reconciling', 'external_outcome_unknown'
            else:
                job.state, job.reason = 'failed', str(getattr(error, 'code', 'activity_failed'))[:128]
            if job.state == 'reconciling':
                self.service._event('job_state', self.service._actors[job.caller.actor_id],
                                    jobId=job.id, state=job.state, reason=job.reason)
            if job.state != 'reconciling' or self.service._state != 'open' or self.service._actors[job.caller.actor_id].state in ('failed', 'bypassed', 'timed_out', 'cancelled', 'budget_exhausted'):
                self.service.activity_finished_locked(job.caller, job.id, state=job.state,
                    result_ref=result.result_ref if result else None, reason=job.reason)
            # All callback tasks are done and the original state transition
            # committed. Only now may supervision release its local owner.
            job.settled, job.finalizing, job.quarantined = True, False, False
            job.pending_result = job.pending_error = None
            self.service._condition.notify_all()
        if job.task.done(): self.supervisor.release(job)

    async def inspect(self, caller, job_id):
        async with self.service._condition:
            return self._view(self._job(caller, job_id))

    async def wait(self, caller, job_id, *, timeout):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= self.service.limits.max_single_wait_seconds:
            raise CoordinationError('invalid_timeout', 'Activity wait requires a finite admitted timeout')
        end = min(self.service.deadline, self.service.clock() + timeout)
        async with self.service._condition:
            while True:
                job = self._job(caller, job_id)
                self.service._expire()
                if job.state in ('completed', 'failed', 'cancelled'):
                    return {**self._view(job), 'outcome': job.state}
                self.service.check_owner_open(caller)
                if self.service._pending(self.service._actors[caller.actor_id]):
                    return {'ok': True, 'outcome': 'inbox_ready', 'jobId': job.id}
                remaining = end - self.service.clock()
                if remaining <= 0: return {'ok': True, 'outcome': 'timeout', 'jobId': job.id}
                try: await asyncio.wait_for(self.service._condition.wait(), timeout=remaining)
                except TimeoutError: continue

    async def cancel(self, caller, job_id):
        async with self.service._condition:
            job = self._job(caller, job_id)
            if job.state in ('completed', 'failed', 'cancelled'):
                return {'ok': True, 'outcome': 'already_terminal'}
            self.service.admit_tool_locked(caller)
            if job.state == 'reconciling':
                return {'ok': True, 'outcome': 'cancel_requested'}
            job.cancel_requested = True
            job.task.cancel()
            if job.run_task is not None and not job.run_task.done(): job.run_task.cancel()
            # A request does not claim that cleanup or remote cancellation has
            # committed. The authoritative state is observable through inspect.
            return {'ok': True, 'outcome': 'cancel_requested'}

    def cancel_actor_locked(self, actor_id):
        for job in self._jobs.values():
            if job.caller.actor_id == actor_id:
                if not job.settled and job.task is not asyncio.current_task():
                    job.cancel_requested = True
                    job.task.cancel()
                    if job.run_task is not None and not job.run_task.done(): job.run_task.cancel()
                elif job.state == 'reconciling':
                    self.service.activity_finished_locked(job.caller, job.id, state=job.state, reason=job.reason)

    async def reconcile(self, job_id, *, actual: UsageBound, result: ActivityResult | None = None, reason='activity_failed'):
        """Trusted provider reconciliation, never a model-authored tool."""
        self.service.authorize()
        if not isinstance(actual, UsageBound) or actual.tool_calls < 1:
            raise CoordinationError('invalid_usage', 'Dispatched work retains its physical tool operation charge')
        if result is not None and not isinstance(result, ActivityResult):
            raise CoordinationError('invalid_result', 'Reconciliation requires an owned result reference')
        if type(reason) is not str or not 1 <= len(reason) <= 128:
            raise CoordinationError('invalid_result', 'Reconciliation reason must be bounded')
        job = self._jobs[job_id]
        if job.state != 'reconciling':
            raise CoordinationError('reconciliation_conflict', 'Only unknown external work needs reconciliation')
        if not job.settled:
            raise CoordinationError('activity_cleanup_pending', 'Local effect callbacks have not finished')
        self._validate_result(job, result)
        await self.service.budget.reconcile(job_id, actual)
        async with self.service._condition:
            self.service.authorize()
            self._validate_result(job, result)
            # Keep the obligation and already-incurred settlement on a failed
            # final authority check; a same-total reconciliation can try later.
            if job.state != 'reconciling':
                raise CoordinationError('reconciliation_conflict', 'Activity was already reconciled')
            job.result, job.state, job.reason = result, 'completed' if result else 'failed', None if result else reason
            self.service.activity_finished_locked(job.caller, job.id, state=job.state,
                result_ref=result.result_ref if result else None, reason=job.reason)
            self.service._condition.notify_all()

    async def snapshot(self, caller):
        async with self.service._condition:
            self.service._access(caller)
            return {'schemaVersion': 1, 'jobs': [self._view(j) for j in self._jobs.values() if j.caller.actor_id == caller.actor_id]}

    async def close(self, *, cancelled=False):
        self._closed = True
        if cancelled:
            self._stopping = True
            # Stop queued coroutine entry before yielding to the teardown task.
            for job in self._jobs.values():
                if not job.task.done(): job.task.cancel()
                if job.run_task is not None and not job.run_task.done(): job.run_task.cancel()
        async def drain():
            try:
                if cancelled:
                    # This is runtime-owned teardown, not new work or a read.
                    # Ordinary resource authorization may already be revoked.
                    await self.service.budget.cancel()
                    async with self.service._condition:
                        self.service._expire()
            finally:
                tasks = [j.task for j in self._jobs.values() if not j.task.done()]
                if cancelled:
                    for task in tasks: task.cancel()
                if tasks: await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(0)  # deliver pre-start cancellation callbacks
                while self._cleanup:
                    await asyncio.gather(*tuple(self._cleanup), return_exceptions=True)
                if any(job.quarantined for job in self._jobs.values()):
                    raise CoordinationError('activity_cleanup_pending',
                                            'Local callbacks remain owned and quarantined; join is unconfirmed')
        cleanup = asyncio.create_task(drain())
        interrupted = False
        while not cleanup.done():
            try: await asyncio.shield(cleanup)
            except asyncio.CancelledError: interrupted = True
        cleanup.result()
        if interrupted: raise asyncio.CancelledError()


class _StartOptions(_Arguments):
    requestId: Identifier | None = None
    idempotencyKey: Identifier | None = None
    priority: Literal['normal', 'peer_request'] = 'normal'


class _Start(_Arguments):
    toolName: Identifier
    arguments: dict[str, JsonValue]
    options: _StartOptions = Field(default_factory=_StartOptions)


class _Job(_Arguments):
    jobId: Identifier


class _WaitOptions(_Arguments):
    timeoutSeconds: float = Field(gt=0)


class _Wait(_Job):
    options: _WaitOptions


_MODELS = dict(startToolJob=_Start, inspectWork=_Job, waitForWork=_Wait, cancelWork=_Job)


class ActivityTools:
    def __init__(self, manager, caller, *, existing_names=()):
        reserve_builtin_names(existing_names)
        manager.service._owner(caller)
        self.manager, self.caller = manager, caller
        allowed = sorted(manager._allowed_tools(caller))
        self.tool_functions = {name: self._callable(name) for name in JOB_TOOL_NAMES} if allowed else {}
        self.tools = []
        for name in self.tool_functions:
            model = _MODELS[name]
            schema = model.model_json_schema()
            if name == 'startToolJob':
                schema['properties']['toolName']['enum'] = allowed
                schema['properties']['arguments'] = {'type': 'object', 'additionalProperties': True}
                schema.get('$defs', {}).pop('JsonValue', None)
            self.tools.append({'type': 'function', 'function': {'name': name,
                'description': 'Operate an owned admitted background activity. Completion arrives as new runtime input; waits are finite and inbox-aware.',
                'parameters': schema}})

    def configure_executor(self, executor):
        if not self.tool_functions: return
        binding = (id(self.manager.service), self.caller.actor_id)
        reserve_builtin_names(name for name in executor.registered_names()
            if getattr(executor._registry[name], '_coordination_binding', None) != binding)
        if any(executor.content_limit(name) < 8192 for name in JOB_TOOL_NAMES):
            raise CoordinationError('tool_output_limit', 'Activity tools need room for complete owned handles and references')
        executor.exclude_from_dedup(*JOB_TOOL_NAMES)
        executor.require_complete_output(*JOB_TOOL_NAMES)
        executor.serialize_tools('startToolJob', 'cancelWork')

    def _callable(self, name):
        async def invoke(**kwargs):
            try:
                key = _operation_key(None)
                await self.manager.service.budget.charge_tool_call(self.caller.actor_id + ':' + key,
                    guard=lambda: self.manager.service._access(self.caller),
                    new_guard=lambda: self.manager.service.admit_tool_locked(self.caller))
                args = _MODELS[name].model_validate(kwargs)
                if name == 'startToolJob':
                    return await self.manager.start(self.caller, args.toolName, args.arguments,
                        key=_operation_key(args.options.idempotencyKey), request_id=args.options.requestId, priority=args.options.priority)
                if name == 'inspectWork': return await self.manager.inspect(self.caller, args.jobId)
                if name == 'waitForWork': return await self.manager.wait(self.caller, args.jobId, timeout=args.options.timeoutSeconds)
                return await self.manager.cancel(self.caller, args.jobId)
            except CoordinationError as exc: return exc.as_result()
            except BudgetError as exc: return {'ok': False, 'error': {'code': exc.code, 'message': str(exc)}}
            except (ValueError, ValidationError):
                return {'ok': False, 'error': {'code': 'invalid_arguments', 'message': 'Invalid activity arguments'}}
        invoke._disable_dedup = invoke._require_complete_output = True
        invoke._coordination_binding = (id(self.manager.service), self.caller.actor_id)
        return invoke
