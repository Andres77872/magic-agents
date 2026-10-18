"""Cancellation must not discard the acknowledgement of an owned commit."""
import asyncio
from types import SimpleNamespace

import pytest
from anyio import CancelScope

from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.execution.recorder import CoreRecorder, CoreScope
from magic_agents.execution.storage import ExecutionStorageError
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from test.test_execution_recorder import Store, Value, config


class DelayedAcknowledgementStore(Store):
    def __init__(self, outcome='success'):
        super().__init__()
        self.outcome = outcome
        self.committed = asyncio.Event()
        self.release = asyncio.Event()
        self.interrupted = False
        self.finished = asyncio.Event()

    async def commit(self, lease, transition):
        receipt = await super().commit(lease, transition)
        if any(event.kind == 'node_output' for event in transition.events):
            self.committed.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.interrupted = True
                # The API SQL adapter joins its owned transaction before
                # propagating cancellation, after the write is durable.
                await self.release.wait()
                raise
            finally:
                self.finished.set()
            if self.outcome == 'failure':
                raise ExecutionStorageError('commit_unknown', 'Acknowledgement unavailable')
            if self.outcome == 'invalid_receipt':
                return receipt.model_copy(update={'operation_id': 'different-operation'})
        return receipt


@pytest.mark.asyncio
@pytest.mark.parametrize('stopped', [False, True])
async def test_cancelled_graph_keeps_committed_version_for_cleanup(stopped):
    store = DelayedAcknowledgementStore()
    graph = AgentFlowModel(nodes={'a': Value('a', 'durable output')}, edges=[])
    runtime = config(store)
    runtime.execution_stop_requested = lambda: stopped

    async def run():
        async for _ in execute_graph_reactive(graph, runtime_config=runtime):
            pass

    task = asyncio.create_task(run())
    try:
        async with asyncio.timeout(3):
            await store.committed.wait()
            task.cancel('client disconnected')
            await asyncio.sleep(0)
            store.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        store.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert store.finished.is_set()
    assert not store.interrupted
    assert store.record.state.status == ('cancelled' if stopped else 'blocked')
    assert [event.kind for event in store.events][-2:] == ['node_error', 'graph_end']
    assert sum(event.kind == 'node_output' for event in store.events) == 1


@pytest.mark.asyncio
async def test_repeated_cancellation_joins_commit_before_releasing_recorder_lock():
    store = DelayedAcknowledgementStore()

    async def run():
        snapshot = config(store).execution_snapshot
        core = CoreRecorder(store, snapshot)
        await core.start()
        scope = CoreScope(core, SimpleNamespace(nodes={}), (), snapshot.identity.run_id,
                          snapshot.identity.root_execution_id, 'scope', True)
        try:
            with pytest.raises(asyncio.CancelledError, match='disconnect'):
                await scope.record('node_output', {'handle': 'out'}, node_id='a', status='running')
            assert core.lease.version == store.record.version
            assert core.state == store.record.state
            assert core.failure is None
            await scope.record('graph_end', {'status': 'blocked'}, status='blocked')
        finally:
            await core.close()

    task = asyncio.create_task(run())
    try:
        async with asyncio.timeout(3):
            await store.committed.wait()
            task.cancel('disconnect')
            await asyncio.sleep(0)
            task.cancel('disconnect repeated')
            await asyncio.sleep(0)
            assert not task.done() and not store.finished.is_set()
            store.release.set()
            await task
    finally:
        store.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not store.interrupted
    assert store.record.state.status == 'blocked'
    assert [event.sequence for event in store.events] == [1, 2]


@pytest.mark.asyncio
async def test_anyio_level_cancellation_preserves_commit_and_graph_cleanup():
    store = DelayedAcknowledgementStore()
    entered = asyncio.Future()

    async def run():
        with CancelScope() as cancellation:
            entered.set_result(cancellation)
            graph = AgentFlowModel(nodes={'a': Value('a', 'durable output')}, edges=[])
            async for _ in execute_graph_reactive(graph, runtime_config=config(store)):
                pass

    task = asyncio.create_task(run())
    try:
        async with asyncio.timeout(3):
            cancellation = await entered
            await store.committed.wait()
            cancellation.cancel()
            await asyncio.sleep(0)
            store.release.set()
            await task
    finally:
        store.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not store.interrupted
    assert store.finished.is_set()
    assert store.record.state.status == 'blocked'
    assert [event.kind for event in store.events][-2:] == ['node_error', 'graph_end']


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome,code', [
    ('failure', 'commit_unknown'), ('invalid_receipt', 'execution_receipt_invalid')])
async def test_cancellation_does_not_hide_storage_failure_or_invalid_receipt(outcome, code):
    store = DelayedAcknowledgementStore(outcome)

    async def run():
        graph = AgentFlowModel(nodes={'a': Value('a', 'durable output')}, edges=[])
        async for _ in execute_graph_reactive(graph, runtime_config=config(store)):
            pass

    task = asyncio.create_task(run())
    try:
        async with asyncio.timeout(3):
            await store.committed.wait()
            task.cancel('disconnect')
            await asyncio.sleep(0)
            store.release.set()
            with pytest.raises(ExecutionStorageError) as caught:
                await task
            assert caught.value.code == code
    finally:
        store.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert store.finished.is_set()
    assert store.record.state.status == 'running'
    assert store.events[-1].kind == 'node_output'
