"""Mandatory native boundaries with an in-process storage port, no database or providers."""
import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.execution.recorder import current_scope
from magic_agents.execution.storage import (ExecutionSnapshot, ExecutionIdentity, ExecutionCause,
    ExecutionRecord, ExecutionState, ExecutionLease, CommitReceipt, ExecutionStorageError, canonical_bytes)
from magic_agents.hooks.runtime_config import RuntimeConfig
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.node_system.Node import Node


class Store:
    def __init__(self, fail_kind=None):
        self.fail_kind = fail_kind
        self.events = []
        self.transitions = []
        self.created = 0
        self.output = None

    async def create(self, snapshot):
        self.created += 1
        self.record = ExecutionRecord(snapshot=snapshot, state=ExecutionState(), version=0, fence=0)
        return self.record

    async def claim(self, identity, *, expected_version, owner_id, lease_seconds):
        assert expected_version == self.record.version
        self.record = self.record.model_copy(update={'version': expected_version + 1, 'fence': 1})
        return ExecutionLease(identity=identity, owner_id=owner_id, version=self.record.version,
                              fence=1, lease_until_ms=9999999999999)

    async def commit(self, lease, transition):
        assert lease.version == self.record.version == transition.expected_version
        if any(event.kind == self.fail_kind for event in transition.events):
            raise ExecutionStorageError('write_failed', 'test storage failure')
        self.record = self.record.model_copy(update={'state': transition.state, 'version': self.record.version + 1})
        self.events.extend(transition.events)
        self.transitions.append(transition)
        if transition.visible_output:
            self.output = transition.visible_output
        return CommitReceipt(identity=lease.identity, operation_id=transition.operation_id,
            content_digest=transition.content_digest, version=self.record.version, fence=1,
            status=transition.state.status)


class Value(Node):
    def __init__(self, name, value):
        super().__init__(node_id=name, node_type='parser')
        self.value = value
        self.called = 0

    async def process(self, chat_log):
        self.called += 1
        yield {'type': 'out', 'content': self.inputs.get('in', self.value)}


class Nested(Value):
    async def process(self, chat_log):
        child = AgentFlowModel(nodes={'leaf': Value('leaf', 'child')}, edges=[])
        async for _ in execute_graph_reactive(child, run_id='child-run'):
            pass
        yield {'type': 'out', 'content': 'nested done'}


def config(store):
    raw = {'nodes': [{'id': 'a'}, {'id': 'b'}], 'edges': []}
    snapshot = ExecutionSnapshot(identity=ExecutionIdentity(run_id='existing-run',
        root_execution_id='existing-root', conversation_id=None), graph_definition=raw,
        graph_digest=hashlib.sha256(canonical_bytes(raw)).hexdigest(),
        runtime_revision='test-native', cause=ExecutionCause(kind='normal'))
    return RuntimeConfig(execution_store=store, execution_snapshot=snapshot)


@pytest.mark.asyncio
async def test_actual_executor_records_without_observer_and_preserves_existing_ids():
    a, b = Value('a', {'answer': 42}), Value('b', None)
    graph = AgentFlowModel(nodes={'a': a, 'b': b}, edges=[EdgeNodeModel(
        id='a-b', source='a', sourceHandle='out', target='b', targetHandle='in')], persistence_enabled=False)
    store = Store()
    result = {}
    async for _ in execute_graph_reactive(graph, runtime_config=config(store), result=result):
        pass
    assert store.created == 1 and store.record.state.status == 'completed'
    assert store.record.snapshot.identity.run_id == 'existing-run'
    assert a.called == b.called == 1 and b.outputs['out'] == {'answer': 42}
    assert {event.kind for event in store.events} >= {'graph_start', 'node_start', 'node_output', 'node_end', 'delivery', 'graph_end'}
    assert not result['has_errors'] and current_scope() is None


@pytest.mark.asyncio
async def test_failed_mandatory_write_propagates_before_node_work():
    node = Value('a', 'never')
    store = Store(fail_kind='node_start')
    with pytest.raises(ExecutionStorageError, match='test storage failure'):
        async for _ in execute_graph_reactive(AgentFlowModel(nodes={'a': node}, edges=[]), runtime_config=config(store)):
            pass
    assert node.called == 0 and store.record.state.status == 'running'
    assert current_scope() is None


@pytest.mark.asyncio
async def test_nested_graph_shares_snapshot_and_has_distinct_scope_path():
    store = Store()
    graph = AgentFlowModel(nodes={'inner': Nested('inner', None)}, edges=[])
    async for _ in execute_graph_reactive(graph, runtime_config=config(store)):
        pass
    assert store.created == 1 and store.record.state.status == 'completed'
    graphs = store.record.state.checkpoint['graphs'].values()
    assert {tuple(item['node_path']) for item in graphs} == {(), ('inner',)}
    assert len({event.data['scope_instance_id'] for event in store.events}) == 2


@pytest.mark.asyncio
async def test_visible_prefix_written_only_in_typed_message_command():
    store = Store()
    graph = AgentFlowModel(nodes={'a': Value('a', 'result')}, edges=[])
    committed = False
    async for _ in execute_graph_reactive(graph, runtime_config=config(store)):
        if not committed:
            assert await current_scope().commit_visible_output(assistant_message_id='message',
                assistant_text='private visible prefix', reasoning='reason') == 1
            committed = True
    assert store.record.state.output_cursor == 1
    assert store.output.assistant_text == 'private visible prefix'
    assert 'private visible prefix' not in canonical_bytes(store.record.state.model_dump(mode='json')).decode()
    assert all('private visible prefix' not in event.model_dump_json() for event in store.events)
