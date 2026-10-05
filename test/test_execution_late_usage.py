"""Accounting-only Stop race; supplied usage reports, no provider/billing IO."""
import asyncio
import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from magic_llm.engine.attempt_control import ProviderAttempt, ProviderAttemptOutcome
from magic_agents.execution.llm_storage import CoreAttemptControl
from magic_agents.execution.recorder import CoreRecorder, CoreScope, NodeInvocation
from magic_agents.execution.storage import ExecutionStorageError, ExecutionUsageObservation, ExecutionUsageReceipt
from test.test_execution_recorder import Store, config


class StopStore(Store):
    def __init__(self, *, accounting_failure=False, write_failure=False):
        super().__init__()
        self.accounting_failure, self.write_failure = accounting_failure, write_failure
        self.usage_calls, self.facts = [], {}

    async def load(self, identity):
        return self.record

    async def commit(self, lease, transition):
        if self.record.state.status == 'cancelled':
            raise ExecutionStorageError('stale_lease', 'Stopped execution')
        if self.write_failure and transition.events[0].kind == 'llm_end':
            raise ExecutionStorageError('write_failed', 'Ordinary storage failure')
        return await super().commit(lease, transition)

    def stop(self):
        effects = tuple(effect.model_copy(update={'status': 'unknown'}) for effect in self.record.state.effects)
        self.record = self.record.model_copy(update={'state': self.record.state.model_copy(update={
            'status': 'cancelled', 'effects': effects, 'cancel_generation': 1}),
            'version': self.record.version + 1, 'fence': self.record.fence + 1})

    async def record_usage(self, identity, observation):
        self.usage_calls.append(observation)
        assert self.record.state.status == 'cancelled'
        effect = self.record.state.effects[0]
        assert effect.status == 'unknown' and effect.request_digest == observation.request_digest
        assert effect.attempt_id == observation.attempt_id
        if self.accounting_failure:
            raise RuntimeError('Accounting unavailable')
        prior = self.facts.setdefault(observation.attempt_id, observation)
        assert prior == observation
        return ExecutionUsageReceipt(identity=identity, attempt_id=observation.attempt_id,
            request_digest=observation.request_digest, usage_digest=observation.usage_digest)


@pytest.mark.asyncio
@pytest.mark.parametrize('report', ['final', 'uncertain', 'partial', 'accounting_failure', 'storage_failure'])
async def test_stop_race_accounts_only_final_report_without_reviving_execution(report):
    store = StopStore(accounting_failure=report == 'accounting_failure', write_failure=report == 'storage_failure')
    started, settled = asyncio.Event(), asyncio.Event()
    attempt = ProviderAttempt(request_operation_id='logical-call', provider_attempt_id='physical-call',
        parent_attempt_id=None, provider='openai', model='example', stream=False,
        attempt_index=0, retry_index=0, is_fallback=False,
        messages=({'role': 'user', 'content': 'request'},), generation_options={'max_tokens': 10})
    report_data = {'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10}
    if report == 'partial': report_data.pop('total_tokens')
    async def producer():
        snapshot = config(store).execution_snapshot
        core = CoreRecorder(store, snapshot)
        await core.start()
        scope = CoreScope(core, SimpleNamespace(nodes={}), (), snapshot.identity.run_id,
                          snapshot.identity.root_execution_id, 'scope', True)
        node = NodeInvocation('llm', 'node-span', snapshot.cause)
        control = CoreAttemptControl(scope, node)
        try:
            await scope.record('graph_start', {'status': 'running'}, status='running')
            await control.before_attempt(attempt)
            started.set()
            await settled.wait()
            await control.after_attempt(attempt, ProviderAttemptOutcome(status='completed', semantic_output=True,
                usage=report_data, usage_uncertain=report == 'uncertain'))
        finally:
            await core.close()

    task = asyncio.create_task(producer())
    await started.wait()
    if report != 'storage_failure': store.stop()
    before = copy.deepcopy(store.record)
    settled.set()
    error_type = ExecutionStorageError if report == 'storage_failure' else asyncio.CancelledError
    with pytest.raises(error_type) as stopped:
        await task
    assert store.record == before  # status, effects, cursor, checkpoint, revision and fence unchanged
    expected_calls = int(report in {'final', 'accounting_failure'})
    assert len(store.usage_calls) == expected_calls
    assert len(store.facts) == int(report == 'final')
    if report == 'final':
        observation = store.usage_calls[0]
        receipt = await store.record_usage(before.snapshot.identity, observation)
        assert receipt.usage_digest == observation.usage_digest and len(store.facts) == 1
        assert store.record == before
    if report == 'accounting_failure':
        assert stopped.value.__notes__ == ['Late usage reconciliation failed: RuntimeError']
    if report == 'storage_failure':
        assert stopped.value.code == 'write_failed'


@pytest.mark.parametrize('value', [None, True, 1.5, -1])
def test_usage_observation_requires_original_final_integer_counts(value):
    with pytest.raises(ValidationError):
        ExecutionUsageObservation(attempt_id='attempt', request_digest='a' * 64,
            provider='openai', model='example', usage={'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': value})
