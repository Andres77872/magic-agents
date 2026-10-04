"""Owned fake activities only: no network, database, or paid provider calls."""
import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
import pytest_asyncio

from magic_agents.coordination.activities import ActivityAdapter, ActivityFailure, ActivityManager, ActivityResult, ActivitySupervisor
from magic_agents.coordination.budget import UsageBound
from magic_agents.coordination.service import CoordinationError, CoordinationService
from magic_agents.coordination.tools import CoordinationTools
from magic_agents.coordination.control import ActorLoopControl, BudgetAttemptControl
from magic_agents.models.coordination import CoordinationLimits, CoordinationPolicy, MessagingConfig
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall
from magic_llm.agent.async_agent_loop import AsyncAgentLoop
from magic_llm.agent.types import AgentBudget
from magic_llm.engine.base_chat import BaseChat, RetryConfig
from magic_llm.engine.tooling import map_request_tools
from magic_llm.model import ModelChatResponse
from magic_llm.model.ModelChatStream import UsageModel

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def factory():
    managers = []
    def build(*, slots=1, queue=4, pending=8, messages=16, checkpoint=200000, run=None, cancel=None, usage=None,
              validate_result=None, cleanup_timeout=5, supervisor=None, lifetime_profile='cooperative-v1'):
        limits = CoordinationLimits(maxModelTurns=30, maxInputTokens=10000, maxOutputTokens=10000,
            maxToolCalls=100, maxImageJobs=20, maxCost={'amount': '20', 'currency': 'USD'},
            maxConcurrentJobs=slots, maxPendingMessagesPerActor=pending, maxAcceptedMessages=messages)
        configs = {role: ((role,), MessagingConfig(enabled=True, role=role, peers=[peer]))
                   for role, peer in [('research', 'images'), ('images', 'research')]}
        service = CoordinationService(CoordinationPolicy(enabled=True, allowedParticipants=list(configs),
            deliveryMode='background_jobs', limits=limits), configs, server_limits=limits,
            authorize=lambda: None, background_capable=True, checkpoint_bytes=checkpoint)
        async def default_run(arguments, effect_id): return ActivityResult('asset:' + effect_id)
        adapter = ActivityAdapter(validate=lambda a: a, estimate=lambda a: UsageBound(
            tool_calls=1, image_jobs=1, cost=Decimal('1')), run=run or default_run,
            usage=usage or (lambda result, error: None if error else UsageBound(tool_calls=1, image_jobs=1, cost=Decimal('.5'))),
            cancel=cancel, validate_result=validate_result, lifetime_profile=lifetime_profile)
        manager = ActivityManager(service, {'generate_image': adapter},
            allowed_tools_by_role={'research': set(), 'images': {'generate_image'}}, max_queued=queue,
            cleanup_timeout=cleanup_timeout, supervisor=supervisor)
        managers.append(manager)
        return service, manager
    yield build
    for manager in managers: await manager.close(cancelled=True)


async def callers(service):
    return await service.activate('research'), await service.activate('images')


async def settled(manager, job_id):
    async with manager.service._condition:
        while not manager._jobs[job_id].settled:
            await asyncio.wait_for(manager.service._condition.wait(), timeout=1)
    await asyncio.gather(manager._jobs[job_id].task, return_exceptions=True)
    await asyncio.sleep(0)  # owner release follows controller task completion


async def consume(service, caller):
    items = await service.inbox(caller)
    await service.checkpoint(caller, {}, [m['messageId'] for m in items])
    return items


async def test_handle_returns_before_owned_dispatch_and_domain_replay_never_repeats_effect(factory):
    assert ActivityManager.result_authority_version == 1
    assert ActivityManager.activity_supervision_version == 1
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []
    async def run(args, identity):
        effects.append(identity); entered.set(); await release.wait()
        return ActivityResult('asset:one', 'remote:one')
    service, manager = factory(run=run); a, b = await callers(service)
    first = await manager.start(b, 'generate_image', {'prompt': 'generic'}, key='effect')
    assert first['state'] == 'queued' and not effects
    await entered.wait()
    replay = await manager.start(b, 'generate_image', {'prompt': 'generic'}, key='effect')
    assert replay['jobId'] == first['jobId'] and replay['state'] == 'queued'
    assert replay['currentDisposition']['state'] == 'running'
    release.set(); await settled(manager, first['jobId'])
    view = await manager.inspect(b, first['jobId'])
    assert view['resultRef'] == 'asset:one' and effects == [first['jobId']]
    assert json.loads(json.dumps(await manager.snapshot(b)))['jobs'][0]['state'] == 'completed'
    notice = (await consume(service, b))[0]
    assert notice['senderRole'] == 'runtime' and notice['payload']['jobId'] == first['jobId']
    await service.checkpoint(a, {}, []); await service.finish(a, 'a'); await service.finish(b, 'b'); await service.seal()
    assert (await manager.start(b, 'generate_image', {'prompt': 'generic'}, key='effect'))['jobId'] == first['jobId']
    assert effects == [first['jobId']]


async def test_queued_jobs_reserve_exposure_but_not_slots_and_queue_is_finite(factory):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def run(args, identity):
        calls.append(identity); entered.set(); await release.wait(); return ActivityResult('asset:' + identity)
    service, manager = factory(run=run, queue=1); a, b = await callers(service)
    first = await manager.start(b, 'generate_image', {}, key='one'); await entered.wait()
    second = await manager.start(b, 'generate_image', {}, key='two')
    before = await service.budget.snapshot()
    with pytest.raises(Exception, match='queue is full'):
        await manager.start(b, 'generate_image', {}, key='three')
    after = await service.budget.snapshot()
    assert before == after and after['activeJobs'] == 1 and after['reserved']['image_jobs'] == 2
    assert calls == [first['jobId']]
    release.set(); await settled(manager, first['jobId']); await settled(manager, second['jobId'])
    assert len(calls) == 2 and (await service.budget.snapshot())['activeJobs'] == 0


async def test_completion_reserved_before_receipt_survives_ordinary_message_saturation(factory):
    started, release = asyncio.Event(), asyncio.Event()
    async def run(args, identity): started.set(); await release.wait(); return ActivityResult('asset:ready')
    service, manager = factory(run=run, pending=2, messages=2); a, b = await callers(service)
    job = await manager.start(b, 'generate_image', {}, key='reserved'); await started.wait()
    await service.send(a, 'images', 'fills general allowance', key='traffic', kind='event', expect_reply=False)
    with pytest.raises(CoordinationError):
        await service.send(a, 'images', 'overflow', key='extra', kind='event', expect_reply=False)
    release.set(); await settled(manager, job['jobId'])
    items = await service.inbox(b)
    assert len(items) == 2 and items[-1]['payload']['resultRef'] == 'asset:ready'
    snap = await service.budget.snapshot()
    assert snap['reserved']['messages'] == 0 and snap['spent']['messages'] == 2


async def test_checkpoint_bytes_for_worst_case_terminal_ref_cannot_be_spent_by_messages(factory):
    started, release = asyncio.Event(), asyncio.Event()
    async def run(args, identity): started.set(); await release.wait(); return ActivityResult('\x00' * 1024)
    service, manager = factory(run=run, checkpoint=18000); a, b = await callers(service)
    job = await manager.start(b, 'generate_image', {}, key='reserve-bytes'); await started.wait()
    with pytest.raises(CoordinationError) as error:
        await service.send(a, 'images', 'x' * 16000, key='steal-bytes', kind='event', expect_reply=False)
    assert error.value.code == 'checkpoint_capacity_exceeded'
    release.set(); await settled(manager, job['jobId'])
    assert (await service.inbox(b))[0]['payload']['resultRef'] == '\x00' * 1024


async def test_inadequate_completion_capacity_rejects_before_effect_or_budget_reservation(factory):
    service, manager = factory(checkpoint=1000); a, b = await callers(service)
    before = await service.budget.snapshot()
    with pytest.raises(CoordinationError, match='checkpoint'):
        await manager.start(b, 'generate_image', {}, key='cannot-deliver')
    assert await service.budget.snapshot() == before
    assert manager._jobs == {} and not service._actors[b.actor_id].activities


async def test_cancel_before_worker_coroutine_starts_settles_exactly_once(factory):
    calls = []
    async def run(args, identity): calls.append(identity); return ActivityResult('must-not-run')
    service, manager = factory(run=run); a, b = await callers(service)
    job = await manager.start(b, 'generate_image', {}, key='cancel-before-start')
    await manager.cancel(b, job['jobId'])
    await settled(manager, job['jobId'])
    assert calls == []
    assert (await manager.inspect(b, job['jobId']))['state'] == 'cancelled'
    assert (await manager.cancel(b, job['jobId']))['outcome'] == 'already_terminal'
    budget = await service.budget.snapshot()
    assert budget['activeJobs'] == 0 and budget['reserved']['image_jobs'] == 0 and budget['spent']['image_jobs'] == 0
    assert len(await service.inbox(b)) == 1


async def test_unknown_external_outcome_retains_capacity_and_exposure_until_reconciled(factory):
    async def run(args, identity): raise ConnectionError('provider acknowledgement lost')
    service, manager = factory(run=run, pending=1, messages=1); a, b = await callers(service)
    job = await manager.start(b, 'generate_image', {}, key='unknown'); await settled(manager, job['jobId'])
    assert (await manager.inspect(b, job['jobId']))['state'] == 'reconciling'
    assert job['jobId'] in service._actors[b.actor_id].activities
    budget = await service.budget.snapshot()
    assert budget['activeJobs'] == 0 and budget['uncertain']['image_jobs'] == 1 and budget['reserved']['messages'] == 1
    assert await service.seal() is None
    with pytest.raises(CoordinationError):
        await service.send(a, 'images', 'cannot steal completion', key='traffic', kind='event', expect_reply=False)
    await manager.reconcile(job['jobId'], actual=UsageBound(tool_calls=1, image_jobs=1, cost=Decimal('.5')),
        result=ActivityResult('asset:reconciled'))
    assert (await service.inbox(b))[0]['payload']['resultRef'] == 'asset:reconciled'
    assert not service._actors[b.actor_id].activities
    assert (await service.budget.snapshot())['uncertain']['image_jobs'] == 0


@pytest.mark.parametrize('view', ['inspect', 'wait', 'snapshot', 'historical_start'])
async def test_completed_result_authority_is_fresh_for_every_cached_view(factory, view):
    allowed, checked = True, []
    def validate(result, identity):
        checked.append((result.result_ref, identity))
        if not allowed: raise CoordinationError('result_denied', 'Owned asset permission revoked')
    service, manager = factory(validate_result=validate); _, caller = await callers(service)
    job = await manager.start(caller, 'generate_image', {}, key='same-effect')
    await settled(manager, job['jobId'])
    assert (await manager.inspect(caller, job['jobId']))['resultRef'].startswith('asset:')
    before = await service.budget.snapshot()
    allowed = False
    with pytest.raises(CoordinationError) as error:
        if view == 'inspect': await manager.inspect(caller, job['jobId'])
        elif view == 'wait': await manager.wait(caller, job['jobId'], timeout=.1)
        elif view == 'snapshot': await manager.snapshot(caller)
        else: await manager.start(caller, 'generate_image', {}, key='same-effect')
    assert error.value.code == 'result_denied'
    assert checked and {identity for _, identity in checked} == {job['jobId']}
    assert await service.budget.snapshot() == before


async def test_result_denial_before_finish_retains_billing_and_never_publishes_ref(factory):
    billed = []
    def validate(result, identity): raise CoordinationError('result_denied', 'Asset denied')
    def usage(result, error):
        billed.append((result, error))
        return UsageBound(tool_calls=1, image_jobs=1, cost=Decimal('.25'))
    service, manager = factory(validate_result=validate, usage=usage); _, caller = await callers(service)
    job = await manager.start(caller, 'generate_image', {}, key='denied-result')
    await settled(manager, job['jobId'])
    assert isinstance(billed[0][0], ActivityResult) and billed[0][1].code == 'result_denied'
    view = await manager.inspect(caller, job['jobId'])
    assert view['state'] == 'reconciling' and 'resultRef' not in view
    assert manager._jobs[job['jobId']].result is None and await service.inbox(caller) == []
    totals = await service.budget.snapshot()
    assert Decimal(totals['spent']['cost']) == Decimal('.25')
    assert Decimal(totals['uncertain']['cost']) == Decimal('.75')
    assert totals['reserved']['messages'] == 1


async def test_finish_rechecks_result_after_awaited_settlement(factory, monkeypatch):
    allowed = True
    def validate(result, identity):
        if not allowed: raise CoordinationError('result_denied', 'Asset denied')
    service, manager = factory(validate_result=validate); _, caller = await callers(service)
    original = service.budget.settle
    async def settle_and_revoke(*args, **kwargs):
        nonlocal allowed
        await original(*args, **kwargs)
        allowed = False
    monkeypatch.setattr(service.budget, 'settle', settle_and_revoke)
    job = await manager.start(caller, 'generate_image', {}, key='commit-race')
    await settled(manager, job['jobId'])
    assert manager._jobs[job['jobId']].state == 'reconciling'
    assert manager._jobs[job['jobId']].result is None and await service.inbox(caller) == []
    assert Decimal((await service.budget.snapshot())['spent']['cost']) == Decimal('.5')
    allowed = True
    await manager.reconcile(job['jobId'], actual=UsageBound(tool_calls=1, image_jobs=1, cost=Decimal('.5')),
                            result=ActivityResult('asset:later-authorized'))
    assert (await service.inbox(caller))[0]['payload']['resultRef'] == 'asset:later-authorized'


async def test_reconcile_validates_before_settlement_and_commit_without_losing_obligation(factory, monkeypatch):
    allowed = False
    async def run(args, identity): raise ConnectionError('Lost acknowledgement')
    def validate(result, identity):
        if not allowed: raise CoordinationError('result_denied', 'Asset denied')
    service, manager = factory(run=run, validate_result=validate); _, caller = await callers(service)
    job = await manager.start(caller, 'generate_image', {}, key='reconcile-authority')
    await settled(manager, job['jobId'])
    usage = UsageBound(tool_calls=1, image_jobs=1, cost=Decimal('.5'))
    before = await service.budget.snapshot()
    with pytest.raises(CoordinationError, match='Asset denied'):
        await manager.reconcile(job['jobId'], actual=usage, result=ActivityResult('asset:late'))
    assert await service.budget.snapshot() == before
    allowed = True
    original = service.budget.reconcile
    async def settle_and_revoke(*args, **kwargs):
        nonlocal allowed
        await original(*args, **kwargs)
        allowed = False
    monkeypatch.setattr(service.budget, 'reconcile', settle_and_revoke)
    with pytest.raises(CoordinationError, match='Asset denied'):
        await manager.reconcile(job['jobId'], actual=usage, result=ActivityResult('asset:late'))
    assert manager._jobs[job['jobId']].state == 'reconciling'
    assert manager._jobs[job['jobId']].result is None and await service.inbox(caller) == []
    assert job['jobId'] in service._actors[caller.actor_id].activities
    assert Decimal((await service.budget.snapshot())['spent']['cost']) == Decimal('.5')
    monkeypatch.setattr(service.budget, 'reconcile', original)
    allowed = True
    await manager.reconcile(job['jobId'], actual=usage, result=ActivityResult('asset:late'))
    assert (await manager.inspect(caller, job['jobId']))['resultRef'] == 'asset:late'


@pytest.mark.parametrize('decision', [True, False, 'accepted'])
async def test_result_validator_cannot_accept_a_non_none_return(factory, decision):
    service, manager = factory(validate_result=lambda *_: decision); _, caller = await callers(service)
    job = await manager.start(caller, 'generate_image', {}, key='invalid-guard')
    await settled(manager, job['jobId'])
    assert manager._jobs[job['jobId']].result is None
    assert (await manager.inspect(caller, job['jobId']))['state'] == 'reconciling'
    assert await service.inbox(caller) == []


async def test_result_validator_closes_accidental_coroutine_without_exposing_result(factory):
    created = []
    async def asynchronous(): raise AssertionError('Must never run')
    def validate(*_):
        value = asynchronous(); created.append(value); return value
    service, manager = factory(validate_result=validate); _, caller = await callers(service)
    job = await manager.start(caller, 'generate_image', {}, key='async-guard')
    await settled(manager, job['jobId'])
    assert all(value.cr_frame is None for value in created)
    assert manager._jobs[job['jobId']].result is None


@pytest.mark.parametrize('confirmed', [True, False])
async def test_close_cancels_and_joins_owned_work_after_ordinary_authorization_revoked(factory, confirmed):
    started, stopped = asyncio.Event(), asyncio.Event()
    cleanup = []
    async def run(args, identity):
        started.set()
        try: await asyncio.Future()
        finally: stopped.set()
    async def cancel(identity): cleanup.append(identity); return confirmed
    service, manager = factory(run=run, cancel=cancel); _, caller = await callers(service)
    first = await manager.start(caller, 'generate_image', {}, key='running')
    await started.wait()
    second = await manager.start(caller, 'generate_image', {}, key='queued')
    def denied(): raise CoordinationError('unauthorized', 'Graph access revoked')
    service.authorize = denied
    await asyncio.wait_for(manager.close(cancelled=True), timeout=1)
    assert stopped.is_set() and cleanup == [first['jobId']]
    assert all(job.task.done() and job.settled for job in manager._jobs.values())
    assert manager._jobs[first['jobId']].state == ('cancelled' if confirmed else 'reconciling')
    assert manager._jobs[second['jobId']].state == 'cancelled'
    assert not manager._cleanup and not service._actors[caller.actor_id].activities
    totals = await service.budget.snapshot()
    assert totals['cancelled'] and totals['activeJobs'] == 0 and totals['reserved']['image_jobs'] == 0
    assert totals['uncertain']['image_jobs'] == 1


async def test_interrupted_close_still_joins_adapter_cancellation_and_settlement(factory):
    started, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def run(args, identity): started.set(); await asyncio.Future()
    async def cancel(identity): cancelling.set(); await release.wait(); return False
    service, manager = factory(run=run, cancel=cancel); _, caller = await callers(service)
    job = await manager.start(caller, 'generate_image', {}, key='cancel-close')
    await started.wait()
    closer = asyncio.create_task(manager.close(cancelled=True))
    await cancelling.wait()
    closer.cancel()
    await asyncio.sleep(0)
    assert not closer.done() and not manager._jobs[job['jobId']].settled
    release.set()
    with pytest.raises(asyncio.CancelledError): await closer
    assert manager._jobs[job['jobId']].task.done() and manager._jobs[job['jobId']].settled
    assert not manager._cleanup and (await service.budget.snapshot())['activeJobs'] == 0


async def test_uncooperative_run_is_quarantined_with_owned_task_permit_and_exposure(factory):
    started, release, ignored = asyncio.Event(), asyncio.Event(), asyncio.Event()
    supervisor = ActivitySupervisor(capacity=2)
    calls = []
    async def run(args, identity):
        calls.append(identity); started.set()
        while not release.is_set():
            try: await release.wait()
            except asyncio.CancelledError: ignored.set()
        return ActivityResult('asset:must-not-publish')
    service, manager = factory(run=run, cleanup_timeout=.02, supervisor=supervisor)
    _, caller = await callers(service)
    job = await manager.start(caller, 'generate_image', {}, key='uncooperative')
    await started.wait()
    try:
        with pytest.raises(CoordinationError) as error:
            await asyncio.wait_for(manager.close(cancelled=True), timeout=.5)
        assert error.value.code == 'activity_cleanup_pending' and ignored.is_set()
        owned = manager._jobs[job['jobId']]
        assert owned.quarantined and not owned.run_task.done() and not owned.settled
        assert supervisor.pending_count == 1 and supervisor._owned[job['jobId']][0] is manager
        before = await service.budget.snapshot()
        assert before['cancelled'] and before['activeJobs'] == 1
        assert Decimal(before['reserved']['cost']) == 1 and before['reserved']['messages'] == 1
        assert owned.result is None and not service._messages
        with pytest.raises(CoordinationError): await service.inbox(caller)
        with pytest.raises(CoordinationError): await service.seal()
        with pytest.raises(CoordinationError, match='Local effect callbacks'):
            await manager.reconcile(job['jobId'], actual=UsageBound(tool_calls=1))
        other_service, other = factory(supervisor=supervisor)
        _, other_caller = await callers(other_service)
        other_before = await other_service.budget.snapshot()
        with pytest.raises(CoordinationError) as denied:
            await other.start(other_caller, 'generate_image', {}, key='cannot-bypass-quarantine')
        assert denied.value.code == 'activity_cleanup_pending'
        assert await other_service.budget.snapshot() == other_before
        assert calls == [job['jobId']]
    finally:
        release.set()
        await settled(manager, job['jobId'])
    assert supervisor.pending_count == 0 and manager._jobs[job['jobId']].result is None
    after = await service.budget.snapshot()
    assert after['activeJobs'] == 0 and Decimal(after['uncertain']['cost']) == 1
    assert calls == [job['jobId']]


async def test_stalled_cancel_callback_and_repeated_caller_cancel_have_one_fixed_grace(factory):
    started, cancelling, release, ignored = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    supervisor = ActivitySupervisor(capacity=1)
    cancels = []
    async def run(args, identity): started.set(); await asyncio.Future()
    async def cancel(identity):
        cancels.append(identity); cancelling.set()
        while not release.is_set():
            try: await release.wait()
            except asyncio.CancelledError: ignored.set()
        return True
    service, manager = factory(run=run, cancel=cancel, cleanup_timeout=.025, supervisor=supervisor)
    _, caller = await callers(service)
    job = await manager.start(caller, 'generate_image', {}, key='stalled-cleanup')
    await started.wait()
    closer = asyncio.create_task(manager.close(cancelled=True))
    await cancelling.wait()
    try:
        for _ in range(3):
            closer.cancel(); await asyncio.sleep(.002)
        with pytest.raises(CoordinationError) as error:
            await asyncio.wait_for(closer, timeout=.5)
        assert error.value.code == 'activity_cleanup_pending'
        assert ignored.is_set() and supervisor.pending_count == 1
        assert manager._jobs[job['jobId']].state == 'reconciling'
        assert (await service.budget.snapshot())['activeJobs'] == 1
        assert cancels == [job['jobId']]
    finally:
        release.set()
        await settled(manager, job['jobId'])
    assert manager._jobs[job['jobId']].state == 'cancelled'
    assert supervisor.pending_count == 0 and cancels == [job['jobId']]


async def test_supervision_slot_is_reserved_before_acceptance_and_released_after_finite_drain(factory):
    release, started = asyncio.Event(), asyncio.Event()
    supervisor = ActivitySupervisor(capacity=1)
    async def run(args, identity): started.set(); await release.wait(); return ActivityResult('asset:ok')
    service, manager = factory(run=run, supervisor=supervisor); _, caller = await callers(service)
    job = await manager.start(caller, 'generate_image', {}, key='owns-slot')
    await started.wait()
    before = await service.budget.snapshot()
    with pytest.raises(CoordinationError) as error:
        await manager.start(caller, 'generate_image', {}, key='no-owner-slot')
    assert error.value.code == 'activity_supervision_capacity'
    assert await service.budget.snapshot() == before and len(manager._jobs) == 1
    release.set(); await settled(manager, job['jobId'])
    assert supervisor.pending_count == 0
    await manager.close()
    assert all(job.task.done() and job.run_task.done() for job in manager._jobs.values())


@pytest.mark.parametrize('profile', [None, '', 'unqualified-stronger-supervisor'])
async def test_missing_or_unqualified_lifetime_profile_rejects_before_jobs(factory, profile):
    with pytest.raises(CoordinationError) as error:
        factory(lifetime_profile=profile)
    assert error.value.code == 'unsupported_capability'


async def test_failed_finisher_retains_owner_exposure_and_reports_pending_cleanup(factory, monkeypatch):
    supervisor = ActivitySupervisor(capacity=1)
    service, manager = factory(supervisor=supervisor); _, caller = await callers(service)
    original = service.budget.settle
    async def fail(*args, **kwargs): raise RuntimeError('Settlement storage unavailable')
    monkeypatch.setattr(service.budget, 'settle', fail)
    loop = asyncio.get_running_loop()
    previous, unhandled = loop.get_exception_handler(), []
    loop.set_exception_handler(lambda loop, context: unhandled.append(context))
    job = await manager.start(caller, 'generate_image', {}, key='settlement-fault')
    try:
        await asyncio.gather(manager._jobs[job['jobId']].task, return_exceptions=True)
        await asyncio.sleep(0)
        owned = manager._jobs[job['jobId']]
        assert owned.quarantined and owned.reason == 'activity_settlement_pending'
        assert not owned.settled and supervisor.pending_count == 1
        assert (await service.budget.snapshot())['activeJobs'] == 1
        # A late notification must not retry a partially failed settlement.
        manager._owned_callback_done(owned, owned.run_task)
        await asyncio.sleep(0)
        assert owned.reason == 'activity_settlement_pending' and not owned.finalizing
        with pytest.raises(CoordinationError) as error:
            await manager.close(cancelled=True)
        assert error.value.code == 'activity_cleanup_pending' and not unhandled
    finally:
        # This injected failure happened before the local transaction mutated.
        # Repair that test fault and explicitly settle the same observed result;
        # production does not automatically retry an uncertain state commit.
        monkeypatch.setattr(service.budget, 'settle', original)
        await manager._finish(manager._jobs[job['jobId']], ActivityResult('asset:' + job['jobId']), None)
        loop.set_exception_handler(previous)
    assert supervisor.pending_count == 0


@pytest.mark.parametrize('confirmed', [False, True])
async def test_running_cancellation_claims_remote_stop_only_when_confirmed(factory, confirmed):
    started = asyncio.Event()
    async def run(args, identity): started.set(); await asyncio.Future()
    async def cancel(identity): return confirmed
    service, manager = factory(run=run, cancel=cancel); a, b = await callers(service)
    job = await manager.start(b, 'generate_image', {}, key='cancel-running'); await started.wait()
    assert (await manager.cancel(b, job['jobId']))['outcome'] == 'cancel_requested'
    await settled(manager, job['jobId'])
    assert (await manager.inspect(b, job['jobId']))['state'] == ('cancelled' if confirmed else 'reconciling')
    assert bool(service._actors[b.actor_id].activities) is not confirmed


async def test_dispatch_rechecks_owner_and_unknown_tool_or_priority_never_accepts(factory):
    calls = []
    async def run(args, identity): calls.append(identity); return ActivityResult('asset:bad')
    service, manager = factory(run=run); a, b = await callers(service)
    for name, options in [('foreign_tool', {}), ('generate_image', {'priority': 'peer_request'})]:
        with pytest.raises(CoordinationError): await manager.start(b, name, {}, key='denied', **options)
    job = await manager.start(b, 'generate_image', {}, key='accepted')
    service._actors[b.actor_id].owner_token = object()
    await settled(manager, job['jobId'])
    assert calls == [] and (await service.budget.snapshot())['spent']['image_jobs'] == 0


@pytest.mark.parametrize('provider', ['openai', 'anthropic', 'google'])
async def test_bound_tools_are_fresh_strict_scoped_and_map_all_provider_schemas(factory, provider):
    service, manager = factory(); a, b = await callers(service)
    peer, jobs = CoordinationTools(b), manager.bind(b)
    executor = ToolExecutor(enable_dedup=True)
    peer.configure_executor(executor); jobs.configure_executor(executor)
    for name, function in {**peer.tool_functions, **jobs.tool_functions}.items(): executor.register(name, function)
    peer.configure_executor(executor); jobs.configure_executor(executor)
    assert len(executor.registered_names()) == 10
    mapped = map_request_tools(provider, jobs.tools, 'auto')
    assert 'JsonValue' not in json.dumps(mapped.tools)
    async def call(identifier, name, args):
        result = await executor.execute_async(CanonicalToolCall(identifier, name, args))
        assert not result.is_deduplicated
        return json.loads(result.content)
    one = await call('one', 'startToolJob', {'toolName': 'generate_image', 'arguments': {}, 'options': {'idempotencyKey': 'domain'}})
    await settled(manager, one['jobId'])
    two = await call('two', 'startToolJob', {'toolName': 'generate_image', 'arguments': {}, 'options': {'idempotencyKey': 'domain'}})
    assert one['jobId'] == two['jobId'] and len(manager._jobs) == 1
    assert (await service.budget.snapshot())['spent']['tool_calls'] == 3  # two physical calls + one effect
    with pytest.raises(CoordinationError): await manager.inspect(a, one['jobId'])
    with pytest.raises(CoordinationError): CoordinationTools(b, existing_names=['startToolJob'])
    with pytest.raises(CoordinationError): manager.bind(b, existing_names=['sendMessageToAgent'])


async def test_real_native_loop_plans_targeted_job_while_generic_live_and_preserves_tool_protocol(factory):
    generic_started, release_generic = asyncio.Event(), asyncio.Event()
    dispatched = []
    service = None
    sender = None
    async def run(args, identity):
        dispatched.append(args['kind'])
        if args['kind'] == 'generic':
            await service.send(sender, 'images', 'targeted image for slide 4', key='targeted', expect_reply=False)
            generic_started.set()
            await release_generic.wait()
        return ActivityResult('asset:' + args['kind'])
    service, manager = factory(run=run)
    sender, actor = await callers(service)
    bundle = manager.bind(actor)
    executor = ToolExecutor()
    bundle.configure_executor(executor)

    class Provider:
        engine, model = 'openai', 'fake'
        kwargs, fallback, retry_config = {}, None, RetryConfig(1, 0)
        def __init__(self): self.calls = []; self.targeted = False
        async def _execute_callback(self, *args): pass
        @BaseChat.async_intercept_generate
        async def async_generate(self, chat, **kwargs):
            self.calls.append(json.loads(json.dumps(chat.messages)))
            content = repr(chat.messages)
            kind = None
            if len(self.calls) == 1: kind = 'generic'
            elif 'targeted image for slide 4' in content and not self.targeted:
                assert generic_started.is_set() and not release_generic.is_set()
                self.targeted = True; kind = 'targeted'
            elif not self.targeted:
                await generic_started.wait()
            message = {'role': 'assistant', 'content': 'final manifest' if 'asset:generic' in content and 'asset:targeted' in content else 'planning while jobs run'}
            if kind:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': kind, 'type': 'function',
                    'function': {'name': 'startToolJob', 'arguments': json.dumps({'toolName': 'generate_image', 'arguments': {'kind': kind}})}}]}
            return ModelChatResponse(id='fake', model='fake', object='chat.completion', created=0,
                choices=[{'index': 0, 'message': message, 'finish_reason': 'tool_calls' if kind else 'stop'}],
                usage=UsageModel(prompt_tokens=3, completion_tokens=2, total_tokens=5))
    provider = Provider()
    admission = BudgetAttemptControl(service.budget, estimate=lambda a: UsageBound(input_tokens=10, output_tokens=10),
        usage=lambda a, outcome: UsageBound(input_tokens=3, output_tokens=2), authorize=lambda a: service.check_owner_open(actor))
    loop = AsyncAgentLoop(SimpleNamespace(llm=provider), tools=bundle.tools, tool_functions=bundle.tool_functions,
        task_executor=executor, control=ActorLoopControl(actor), provider_attempt_control=admission,
        builtin_todo_tools=False, budget=AgentBudget(max_iterations=15))
    task = asyncio.create_task(loop.run('Make images'))
    async with service._condition:
        while len(manager._jobs) < 2:
            await asyncio.wait_for(service._condition.wait(), timeout=2)
    assert provider.targeted and dispatched == ['generic']
    assert not task.done() and (await service.budget.snapshot())['activeJobs'] == 1
    release_generic.set()
    result = await asyncio.wait_for(task, 2)
    assert result.content == 'final manifest' and dispatched == ['generic', 'targeted']
    canonical = loop.checkpoint.messages
    for index, message in enumerate(canonical):
        if message.get('tool_calls'):
            ids = [item['id'] for item in message['tool_calls']]
            assert [item.get('tool_call_id') for item in canonical[index + 1:index + 1 + len(ids)]] == ids
    assert sum(m.get('role') == 'tool' for m in canonical) == 2
    assert not service._actors[actor.actor_id].activities


async def test_adapter_rechecks_resource_authority_at_real_dispatch(factory):
    service, manager = factory(); a, b = await callers(service)
    allowed = True
    calls = []
    def validate(args):
        if not allowed: raise PermissionError('resource access revoked')
        return args
    async def run(args, identity): calls.append(identity); return ActivityResult('must-not-run')
    old = manager.adapters['generate_image']
    manager.adapters['generate_image'] = ActivityAdapter(validate, old.estimate, run, old.usage,
                                                         lifetime_profile='cooperative-v1')
    job = await manager.start(b, 'generate_image', {}, key='resource-check')
    allowed = False
    await settled(manager, job['jobId'])
    assert calls == [] and (await manager.inspect(b, job['jobId']))['state'] == 'failed'
    assert (await service.budget.snapshot())['spent']['image_jobs'] == 0


async def test_pre_receipt_admission_failure_rolls_back_queued_exposure_atomically(factory, monkeypatch):
    service, manager = factory(); a, b = await callers(service)
    original = service.admit_activity_locked
    def fail_commit(caller, job_id, *, commit=False):
        if commit: raise CoordinationError('not_authorized', 'Permission changed before acceptance')
        return original(caller, job_id, commit=commit)
    monkeypatch.setattr(service, 'admit_activity_locked', fail_commit)
    before = await service.budget.snapshot()
    with pytest.raises(CoordinationError):
        await manager.start(b, 'generate_image', {}, key='not-accepted')
    assert await service.budget.snapshot() == before
    assert not service.budget._shared.journal and not manager._jobs and not manager._operations
    assert not service._actors[b.actor_id].activities


async def test_targeted_priority_is_fifo_with_bounded_burst_so_generic_work_cannot_starve(factory):
    started, release = asyncio.Event(), asyncio.Event()
    order = []
    async def run(args, identity):
        order.append(args['name'])
        if args['name'] == 'held': started.set(); await release.wait()
        return ActivityResult('asset:' + args['name'])
    service, manager = factory(run=run, pending=16, messages=32, queue=8); a, b = await callers(service)
    request = await service.send(a, 'images', 'targeted work', key='priority-request')
    held = await manager.start(b, 'generate_image', {'name': 'held'}, key='held'); await started.wait()
    jobs = []
    for name in ['normal1', 'normal2', 'target1', 'target2', 'target3']:
        options = {'request_id': request['receipt']['requestId'], 'priority': 'peer_request'} if name.startswith('target') else {}
        jobs.append(await manager.start(b, 'generate_image', {'name': name}, key=name, **options))
    release.set()
    for job in [held, *jobs]: await settled(manager, job['jobId'])
    assert order == ['held', 'target1', 'target2', 'normal1', 'target3', 'normal2']


async def test_actor_job_grants_are_explicit_narrowed_and_enforced_before_acceptance(factory):
    service, manager = factory(); research, images = await callers(service)
    denied_bundle = manager.bind(research)
    assert denied_bundle.tools == [] and denied_bundle.tool_functions == {}
    images_bundle = manager.bind(images)
    start_schema = next(spec['function']['parameters'] for spec in images_bundle.tools if spec['function']['name'] == 'startToolJob')
    assert start_schema['properties']['toolName']['enum'] == ['generate_image']
    before = await service.budget.snapshot()
    with pytest.raises(CoordinationError) as error:
        await manager.start(research, 'generate_image', {'prompt': 'steal image capability'}, key='forbidden')
    assert error.value.code == 'tool_not_authorized'
    assert await service.budget.snapshot() == before and not manager._jobs
    with pytest.raises(TypeError):
        manager._allowed_tools_by_role['research'] = frozenset({'generate_image'})


async def test_repeated_worker_cancellation_during_settlement_does_not_strand_permit(factory):
    entered = asyncio.Event()
    async def run(args, identity): entered.set(); await asyncio.Future()
    service, manager = factory(run=run); a, b = await callers(service)
    job = await manager.start(b, 'generate_image', {}, key='double-cancel'); await entered.wait()
    worker = manager._jobs[job['jobId']].task
    async with service._condition:
        worker.cancel()
        await asyncio.sleep(0)
        worker.cancel()
        await asyncio.sleep(0)
    await settled(manager, job['jobId'])
    snapshot = await service.budget.snapshot()
    assert snapshot['activeJobs'] == 0 and snapshot['reserved']['image_jobs'] == 0
    assert snapshot['uncertain']['image_jobs'] == 1
