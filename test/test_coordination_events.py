import asyncio
import json

import pytest

from magic_agents.coordination.events import PrivateEventJournal, EventJournalError
from magic_agents.coordination.service import CoordinationError
from test.test_coordination_service import service, actors, consume


def journal(s, *, events=None, run='run-child', root='run-root'):
    events = events or PrivateEventJournal()
    events.bind_scope(root_run_id=root, run_id=run, workgroup_id='group-shared', epoch_id=s.epoch_id,
                      scope_id=s.scope_id, terminal_slots=64)
    s.events = events
    return events


@pytest.mark.asyncio
async def test_committed_private_events_capture_wait_reply_wake_and_original_identity():
    s = service(); events = journal(s); a, b = await actors(s)
    first = await s.send(a, 'images', 'PRIVATE request secret', {'credential': 'PRIVATE value'}, key='one')
    request = first['receipt']['requestId']
    await consume(s, b)
    waiting = asyncio.create_task(s.wait_message(a, request_id=request, timeout=1))
    while not any(e['type'] == 'wait_started' for e in events.read()):
        await events.wait_after(events.sequence)
    await s.reply(b, request, 'PRIVATE reply secret', key='reply')
    assert (await waiting)['outcome'] == 'reply'
    await consume(s, a)
    await s.finish(b, 'PRIVATE output')
    with pytest.raises(CoordinationError):
        await s.send(a, 'images', 'denied', key='denied', expect_reply=False)
    await s.send(a, 'images', 'wake', key='wake', expect_reply=False, wake=True)
    await s.send(a, 'images', 'coalesced', key='coalesced', expect_reply=False, wake=True)
    b2 = await s.activate('images'); assert b2.activation_id != b.activation_id
    await consume(s, b2); await s.finish(b2, 'PRIVATE new output')
    await s.finish(a, 'PRIVATE final'); await s.seal()
    frames = events.read()
    kinds = {e['type'] for e in frames}
    assert {'message_accepted', 'message_consumed', 'request_resolved', 'wait_started', 'wait_finished',
            'wake_denied', 'wake_scheduled', 'wake_coalesced', 'activation_finished', 'epoch_sealed'} <= kinds
    assert [e['eventSequence'] for e in frames] == list(range(1, len(frames)+1))
    assert 'PRIVATE' not in json.dumps(frames) and 'credential' not in json.dumps(frames)
    image_starts = [e for e in frames if e['type'] == 'activation_started' and e['actorId'] == b.actor_id]
    assert [e['activationId'] for e in image_starts] == [b.activation_id, b2.activation_id]
    assert {e['runId'] for e in frames} == {'run-child'}
    assert {e['rootRunId'] for e in frames} == {'run-root'}
    frames[0]['data']['changed'] = True
    assert events.read()[0]['data'].get('changed') is None
    assert events.read()[-1]['type'] == 'epoch_sealed'


@pytest.mark.asyncio
async def test_scopes_with_identical_node_paths_share_sequence_not_actors_or_runs():
    s, t = service(), service(); events = journal(s)
    journal(t, events=events, run='run-another-child')
    await actors(s); await actors(t)
    frames = events.read()
    starts = [e for e in frames if e['type'] == 'activation_started']
    assert len({e['actorId'] for e in starts}) == 4
    assert len({e['scopeInstanceId'] for e in starts}) == 2
    assert {e['runId'] for e in starts} == {'run-child', 'run-another-child'}
    assert len({tuple(e['nodePath']) for e in starts}) == 2
    assert [e['eventSequence'] for e in frames] == list(range(1, len(frames)+1))


def test_ordinary_pressure_cannot_consume_reserved_terminals_and_replay_gap_is_explicit():
    events = PrivateEventJournal(max_events=6, max_bytes=6000, record_bytes=1000)
    events.bind_scope(root_run_id='r', run_id='r', workgroup_id='g', epoch_id='e', scope_id='s', terminal_slots=2)
    events.append('s', 'epoch_sealed', {}, terminal=True)
    for _ in range(20): events.append('s', 'message_rejected', {'reason': 'queue_full'})
    events.append('s', 'budget_stopped', {'reason': 'budget_exhausted'}, terminal=True)
    with pytest.raises(EventJournalError, match='event_retention_unavailable'): events.read(after=0)
    assert len(events._terminal) == 2
    assert events.read(after=21)[0]['type'] == 'budget_stopped'
    assert len(events._ordinary) + len(events._terminal) <= events.max_events
    assert events._ordinary_bytes + sum(len(v) for _, v in events._terminal) <= events.max_bytes
    with pytest.raises(EventJournalError, match='event_terminal_capacity'):
        events.bind_scope(root_run_id='r', run_id='other', workgroup_id='g', epoch_id='e', scope_id='t', terminal_slots=4)


@pytest.mark.asyncio
async def test_replayed_mutation_does_not_emit_new_acceptance_or_charge_again():
    s = service(); events = journal(s); a, _ = await actors(s)
    result = await s.send(a, 'images', 'private', key='repeat', expect_reply=False)
    cursor = events.sequence
    replay = await s.send(a, 'images', 'private', key='repeat', expect_reply=False)
    assert replay['receipt'] == result['receipt'] and events.read(after=cursor) == ()
    await s.cancel()
    cancelled = [e for e in events.read() if e['type'] == 'activation_cancelled']
    assert len(cancelled) == 2
    cursor = events.sequence
    await s.cancel()
    assert events.read(after=cursor) == ()


@pytest.mark.asyncio
async def test_actual_scheduler_binds_run_before_first_actor_event():
    from test.test_coordination_graph import runtime, node, graph, collect
    rt = runtime(public_source=('author',))
    async def peer(*args): return {'content': 'PRIVATE peer output'}
    async def author(*args): return {'content': 'Public output'}
    a, _ = node('research', peer, peer='images')
    b, _ = node('images', peer, peer='research')
    c, _ = node('author', author)
    result, messages = await collect(graph({'research': a, 'images': b, 'author': c}), rt)
    assert not result['has_errors'], messages
    frames = rt.events.read()
    assert frames and {e['runId'] for e in frames} == {rt._root.run_id}
    assert {e['rootRunId'] for e in frames} == {rt._root.run_id}
    assert frames[-1]['type'] == 'epoch_sealed'
    assert 'PRIVATE' not in json.dumps(frames)


@pytest.mark.asyncio
async def test_actual_nested_scheduler_preserves_distinct_run_bindings(monkeypatch):
    from test import test_coordination_graph as graphs
    create = graphs.runtime
    captured = []
    def remember(*args, **kwargs):
        result = create(*args, **kwargs); captured.append(result); return result
    monkeypatch.setattr(graphs, 'runtime', remember)
    await graphs.test_two_real_inner_scopes_share_lineage_and_publish_the_exact_nested_author()
    rt = captured[0]
    assert len({scope.run_id for scope in rt.scopes}) == 3
    frames = rt.events.read()
    assert {e['rootRunId'] for e in frames} == {rt._root.run_id}
    for scope in rt.scopes[1:]:
        scoped = [e for e in frames if e['scopeInstanceId'] == scope.scope_id]
        assert scoped and {e['runId'] for e in scoped} == {scope.run_id}
        assert scope.run_id != rt._root.run_id


@pytest.mark.asyncio
async def test_actual_persistence_start_binds_its_run_to_scheduler_events():
    from unittest.mock import AsyncMock
    from test.test_coordination_graph import runtime, node, graph
    from magic_agents.execution.reactive_executor import execute_graph_reactive
    from magic_agents.hooks import GraphPersistenceHook
    from magic_agents.hooks.hook_registry import HookRegistry
    from magic_agents.hooks.runtime_config import RuntimeConfig
    rt = runtime()
    async def reply(*args): return {'content': 'done'}
    a, _ = node('research', reply, peer='images'); b, _ = node('images', reply, peer='research')
    c, _ = node('author', reply)
    sink = AsyncMock()
    sink.begin_run.return_value = 'persisted-run'
    sink.begin_execution.return_value = 'persisted-execution'
    hook = GraphPersistenceHook(sink=sink, id_chat='chat', id_thread='thread', id_user='user')
    hooks = HookRegistry(); hooks.register_graph(hook)
    result = {}
    async for _ in execute_graph_reactive(graph({'research': a, 'images': b, 'author': c}),
        runtime_config=RuntimeConfig(coordination=rt), hooks=hooks, result=result): pass
    assert not result['has_errors']
    sink.begin_run.assert_awaited_once()
    assert rt._root.run_id == hooks.run_id == hook.run_id == 'persisted-run'
    assert {e['runId'] for e in rt.events.read()} == {'persisted-run'}
