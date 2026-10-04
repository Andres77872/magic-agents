"""Existing Hook factory: shared Python and built-in messages bindings."""
import asyncio
from copy import deepcopy
import pytest
from magic_agents.agt_flow import build, run_agent, validate_graph
from magic_agents.models.factory.Nodes.HookNodeModel import HookNodeModel
from magic_agents.node_system.NodeHook import NodeHook
from magic_agents.hooks.messages_hook import effective_messages_nodes
from magic_agents.util.coordination_validation import require_valid_coordination


def definition():
    return {'nodes': [{'id': 'input', 'type': 'user_input'},
        {'id': 'a', 'type': 'llm', 'data': {}}, {'id': 'b', 'type': 'llm', 'data': {}},
        {'id': 'shared', 'type': 'hook', 'data': {'hook_mode': 'messages', 'target_node_ids': ['a', 'b'],
            'messaging_by_target': {'a': {'enabled': False}, 'b': {'enabled': False}}}}], 'edges': []}


def test_real_hook_factory_binds_distinct_target_configs_and_fresh_invocations():
    raw = definition(); raw['nodes'][-1]['data']['messaging_by_target']['a']['description'] = 'A only'
    before = deepcopy(raw); graph = build(raw, message='test')
    assert raw == before and isinstance(graph.nodes['shared'], NodeHook)
    assert graph.nodes['shared'].hook_mode == 'messages'
    assert graph.nodes['a'].messaging.description == 'A only'
    assert graph.nodes['b'].messaging.description != 'A only'
    assert graph.nodes['a']._invocation_factory().messaging == graph.nodes['a'].messaging
    assert 'messaging' not in raw['nodes'][1]['data']
    from magic_agents.hooks.invocation_control import InvocationControl
    controller = InvocationControl(graph.nodes, graph.edges)
    assert 'shared' in controller.hook_ids and not controller.bindings


@pytest.mark.parametrize('change', [
    lambda g: g['nodes'][-1]['data'].update(target_node_ids=[]),
    lambda g: g['nodes'][-1]['data'].update(target_node_ids=None),
    lambda g: g['nodes'][-1]['data'].update(target_node_ids=['a', 'a']),
    lambda g: g['nodes'][-1]['data'].update(target_node_ids=['a', 'absent']),
    lambda g: g['nodes'][-1]['data'].update(target_node_ids=['a', 'input']),
    lambda g: g['nodes'][-1]['data'].update(target_node_id='a'),
    lambda g: g['nodes'][-1]['data']['messaging_by_target'].pop('b'),
    lambda g: g['nodes'][-1]['data']['messaging_by_target'].update(extra={'enabled': False}),
    lambda g: g['nodes'][-1]['data'].update(function_template='async def f(c,l): return None'),
    lambda g: g['nodes'][-1]['data'].update(lifecycle_event='onStart'),
    lambda g: g['nodes'][1]['data'].update(messaging=None),
    lambda g: g['nodes'].append({'id': 'second', 'type': 'hook', 'data': {
        'hook_mode': 'messages', 'target_node_ids': ['b'], 'messaging_by_target': {'b': {'enabled': False}}}}),
    lambda g: g['edges'].append({'id': 'normal', 'source': 'shared', 'target': 'a'}),
    lambda g: g['edges'].append({'id': 'normal', 'source': 'input', 'target': 'shared'}),
    lambda g: g['edges'].append({'id': 'intercept', 'source': 'input', 'target': 'a',
        'hooks': {'enabled': True, 'hook_node_id': 'shared'}}),
])
def test_invalid_binding_rejects_before_any_node_construction(change, monkeypatch):
    raw = definition(); change(raw)
    import magic_agents.agt_flow as flow
    monkeypatch.setattr(flow, 'create_node', lambda *a, **k: pytest.fail('node construction reached'))
    with pytest.raises(ValueError): build(raw, message='')
    assert not validate_graph(raw['nodes'], raw['edges'])['valid']


def test_scalar_python_legacy_and_explicit_python_are_preserved():
    old = HookNodeModel(target_node_id='a', lifecycle_event='onFinish', function_template='async def f(c,l): return None')
    assert old.hook_mode == 'python' and old.target_ids == ('a',)
    assert HookNodeModel(hook_mode='python', target_node_ids=['a', 'b']).target_ids == ('a', 'b')
    with pytest.raises(ValueError): HookNodeModel(target_node_id='a', target_node_ids=['a'])


def test_cross_scope_target_cannot_borrow_inner_identity():
    raw = definition(); raw['nodes'][2] = {'id': 'inner', 'type': 'inner', 'data': {'magic_flow': {
        'nodes': [{'id': 'b', 'type': 'llm'}], 'edges': []}}}
    with pytest.raises(ValueError): require_valid_coordination(raw)


@pytest.mark.asyncio
async def test_one_python_hook_runs_concurrently_with_target_local_contexts():
    template = '''async def shared(context, chat_log):
    import asyncio
    target = context.event['node_id']
    value = context.request['content']['value']
    await asyncio.sleep(0)
    context.emit.debug({'target': target, 'value': value})
    return {'action': 'outcome', 'outcome': {'status': 'success', 'content': {'handle_parser_output': target + ':' + value}}}
'''
    raw = {'nodes': [{'id': 'input', 'type': 'user_input'},
        *[{'id': name, 'type': 'parser', 'data': {'text': 'original'}} for name in ('a', 'b')],
        {'id': 'shared', 'type': 'hook', 'data': {'target_node_ids': ['a', 'b'],
            'lifecycle_event': 'onStart', 'function_template': template}}, {'id': 'end', 'type': 'end'}],
        'edges': [*({'id': 'in-'+name, 'source': 'input', 'target': name,
            'sourceHandle': 'handle_user_message', 'targetHandle': 'value'} for name in ('a', 'b')),
            *({'id': 'out-'+name, 'source': name, 'target': 'end',
            'sourceHandle': 'handle_parser_output', 'targetHandle': 'handle_flow_input'} for name in ('a', 'b'))]}
    graph = build(raw, message='original')
    assert not graph._validation_errors
    await asyncio.wait_for(_collect(graph), 3)
    records = graph.nodes['a']._invocation_control.records
    targets = [record for record in records if record['kind'] == 'node' and record['node_id'] in ('a','b')]
    assert {record['node_id'] for record in targets} == {'a', 'b'}
    for record in targets:
        assert not record['executed']
        assert record['outcome']['content'] == {'handle_parser_output': record['node_id'] + ':original'}
        assert record['child'][0]['node_id'] == 'shared'
        assert record['child'][0]['side_events'] == [{'type': 'debug', 'content': {'target': record['node_id'], 'value': 'original'}}]
    assert targets[0]['child'][0]['id'] != targets[1]['child'][0]['id']


async def _collect(graph):
    return [event async for event in run_agent(graph)]
