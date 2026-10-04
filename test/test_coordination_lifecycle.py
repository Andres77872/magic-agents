"""Actual lifecycle templates/child nodes retain execution-owned authority."""
import asyncio

import pytest

from magic_agents.coordination.budget import BudgetError, UsageBound
from magic_agents.coordination.service import CoordinationError
from magic_agents.hooks.invocation_control import InvocationControl, OperationFailure
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes.HookNodeModel import HookNodeModel
from magic_agents.models.factory.Nodes.ParserNodeModel import ParserNodeModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.NodeHook import NodeHook
from magic_agents.node_system.NodeParser import NodeParser
from magic_llm.agent.control import AgentControlError
from magic_llm.agent.types import AgentBudgetExceeded
from magic_llm.engine.attempt_control import ProviderAttemptControlError
from magic_llm.exception.ChatException import RequestValidationError

from test.test_coordination_graph import collect, limits, node, runtime


pytestmark = pytest.mark.asyncio


def parser(name):
    return NodeParser(ParserNodeModel(text='core:{{ value }}'), node_id=name, node_type='parser')


def hook(name, template, *, phase='onStart', target='target', failure_policy='preserve'):
    return NodeHook(HookNodeModel(function_template=template, lifecycle_event=phase,
        target_node_id=target, failure_policy=failure_policy), node_id=name, node_type='hook')


def connection(source, target, handle):
    return EdgeNodeModel(id=source + '-' + target, source=source, target=target,
        sourceHandle='handle-child-call', targetHandle=handle)


def setup(nodes, edges=(), *, rt=None, timeout=2):
    rt = rt or runtime(tool_estimate=lambda *args: UsageBound(tool_calls=1),
                       tool_usage=lambda *args: UsageBound(tool_calls=1))
    g = AgentFlowModel(type='graph', nodes={n.node_id: n for n in nodes}, edges=list(edges), timeout=timeout)
    scope = rt.enter_graph(g)
    control = InvocationControl(g.nodes, g.edges, timeout=timeout)
    log = ModelAgentRunLog(coordination=scope, flow_state={'ordinary': {'value': 'original'}})
    return rt, scope, control, log, g


async def response(provider, chat):
    return {'content': 'child answer'}


@pytest.mark.parametrize('stream', [False, True])
async def test_real_llm_child_and_hook_snapshot_share_one_authority(stream, monkeypatch):
    child, provider = node('child', response, stream=stream)
    control_hook = hook('control', '''async def control(context, chat_log):
    chat_log.flow_state['ordinary']['value'] = 'isolated'
    context.emit.debug({'budgetId': chat_log.coordination.budget.id})
    child = await context.call('control-child', 'child prompt')
    return {'action': 'outcome', 'outcome': {'status': 'success', 'content':
        {'handle_parser_output': child['outcome']['content']['handle_generated_content']}}}
''')
    rt, scope, control, log, _ = setup([parser('target'), child, control_hook],
        [connection('control', 'child', child.INPUT_HANDLER_USER_MESSAGE)])
    seen = []
    original = type(child).process
    async def observe(self, child_log):
        seen.append(child_log.coordination)
        child_log.flow_state['ordinary']['value'] = 'child isolated'
        async for event in original(self, child_log):
            yield event
    monkeypatch.setattr(type(child), 'process', observe)
    record = await control.invoke('target', {'value': 'original'}, log)
    assert record['outcome']['content'] == {'handle_parser_output': 'child answer'}
    assert not record['executed'] and seen == [scope]
    assert log.flow_state == {'ordinary': {'value': 'original'}}
    assert record['child'][0]['side_events'][0]['content'] == {'budgetId': rt.budget.id}
    assert len(provider.calls) == 1
    assert (await rt.budget.snapshot())['modelTurns'] == 1
    assert (await rt.budget.snapshot())['spent']['tool_calls'] == 0
    assert [entry['state'] for entry in scope.child_operations.values()] == ['completed']


async def test_real_graph_lifecycle_can_replace_input_and_effective_output():
    rt = runtime(tool_estimate=lambda *args: UsageBound(tool_calls=1),
                 tool_usage=lambda *args: UsageBound(tool_calls=1))
    target, child = parser('target'), parser('child')
    target.inputs['value'] = 'original'
    start = hook('start', "async def control(context, chat_log): return {'action':'input','content':{'value':'replaced'}}")
    finish = hook('finish', '''async def finish(context, chat_log):
    child = await context.call('finish-child', 'useful')
    return {'action':'outcome','outcome':{'status':'success','content':{
        'handle_parser_output': context.outcome['content']['handle_parser_output'] + '+' +
            child['outcome']['content']['handle_parser_output']}}}
''', phase='onFinish')
    g = AgentFlowModel(type='graph', nodes={n.node_id:n for n in [target, child, start, finish]},
        edges=[connection('finish', 'child', 'value')], timeout=2)
    outcome, events = await collect(g, rt)
    assert outcome['has_errors'] is False, events
    assert 'core:replaced+core:useful' in repr(target.outputs)
    assert (await rt.budget.snapshot())['spent']['tool_calls'] == 1
    assert len(rt.scopes[0].child_operations) == 1


async def test_ordinary_failure_recovers_through_admitted_real_provider(monkeypatch):
    child, provider = node('child', response)
    recovery = hook('recover', '''async def recover(context, chat_log):
    child = await context.call('recover-child', 'recover prompt')
    return {'action':'outcome','outcome':{'status':'success','content':{
        'handle_parser_output':child['outcome']['content']['handle_generated_content']}}}
''', phase='onError')
    rt, scope, control, log, _ = setup([parser('target'), child, recovery],
        [connection('recover', 'child', child.INPUT_HANDLER_USER_MESSAGE)])
    async def failed(self, log):
        raise RuntimeError('ordinary parser failure')
        yield
    monkeypatch.setattr(NodeParser, 'process', failed)
    record = await control.invoke('target', {'value':'original'}, log)
    assert record['original_outcome']['error']['code'] == 'NODE_EXCEPTION'
    assert record['outcome']['content'] == {'handle_parser_output':'child answer'}
    assert record['recovered'] and len(provider.calls) == 1
    assert (await rt.budget.snapshot())['modelTurns'] == 1


@pytest.mark.parametrize('error', [BudgetError('budget_exhausted', 'no allowance'),
    CoordinationError('stale_owner', 'fence lost'), AgentControlError('closed', 'epoch_closed'),
    ProviderAttemptControlError('denied', error_code='authorization_denied'), PermissionError('denied'),
    AgentBudgetExceeded('max_iterations', 1, 2), RequestValidationError(PermissionError('request guard'))])
@pytest.mark.parametrize('failure_policy', ['preserve', 'fail'])
async def test_protected_operation_failure_cannot_use_recovery_or_finish(error, failure_policy, monkeypatch):
    recovery = hook('recover', "async def recover(context, chat_log): return {'action':'outcome','outcome':{'status':'success','content':{'handle_parser_output':'unauthorized'}}}",
                    phase='onError', failure_policy=failure_policy)
    finish = hook('finish', "async def finish(context, chat_log): return {'action':'outcome','outcome':{'status':'success','content':{'handle_parser_output':'unauthorized'}}}",
                  phase='onFinish', failure_policy=failure_policy)
    _, _, control, log, _ = setup([parser('target'), recovery, finish])
    async def denied(self, log):
        raise error
        yield
    monkeypatch.setattr(NodeParser, 'process', denied)
    with pytest.raises(type(error)):
        await control.invoke('target', {'value':'original'}, log)
    root = control.records[0]
    assert root['protected'] and not root['recovered']
    assert root['outcome']['status'] == 'error' and not root['child']


@pytest.mark.parametrize('stream', [False, True])
async def test_caught_child_budget_denial_is_latched_and_cannot_launch_more_work(stream):
    child, provider = node('child', response, stream=stream)
    recovery = hook('control', '''async def control(context, chat_log):
    await context.call('control-child', 'first')
    try:
        await context.call('control-child', 'second')
    except Exception:
        try:
            await context.call('control-child', 'third')
        except Exception:
            pass
    return {'action':'outcome','outcome':{'status':'success','content':{'handle_parser_output':'fake recovery'}}}
''')
    rt, scope, control, log, _ = setup([parser('target'), child, recovery],
        [connection('control', 'child', child.INPUT_HANDLER_USER_MESSAGE)], rt=runtime(limits(maxModelTurns=1)))
    with pytest.raises(ProviderAttemptControlError):
        await control.invoke('target', {'value':'original'}, log)
    assert len(provider.calls) == 1
    assert (await rt.budget.snapshot())['modelTurns'] == 1
    assert [entry['state'] for entry in scope.child_operations.values()] == ['completed', 'unknown']
    assert control.records[0]['outcome']['status'] == 'error'


async def test_missing_child_bounds_denies_before_effect_even_when_hook_preserves(monkeypatch):
    control_hook = hook('control', "async def control(context, chat_log): return {'action':'redirect','connection':'control-child','content':'effect'}")
    _, scope, control, log, _ = setup([parser('target'), parser('child'), control_hook],
        [connection('control', 'child', 'value')], rt=runtime())
    calls = []
    original = NodeParser.process
    async def observed(self, log):
        calls.append(self.node_id)
        async for event in original(self, log): yield event
    monkeypatch.setattr(NodeParser, 'process', observed)
    with pytest.raises(CoordinationError, match='trusted cost'):
        await control.invoke('target', {'value':'original'}, log)
    assert not calls and not scope.child_operations


async def test_actor_child_rejects_before_hook_or_clone():
    control_hook = hook('control', "async def control(context, chat_log): return {'action':'redirect','connection':'control-child','content':'x'}")
    _, scope, control, log, _ = setup([parser('target'), parser('child'), control_hook],
        [connection('control', 'child', 'value')])
    # Simulate an actor discovered after a caller was assembled. Runtime checks
    # remain required in addition to authored topology validation.
    scope.roles['child'] = 'child'
    with pytest.raises(CoordinationError) as caught:
        await control.invoke('target', {'value':'original'}, log)
    assert caught.value.code == 'unsupported_coordination_topology'
    assert not scope.child_operations


async def test_fence_revocation_during_hook_prevents_output_override():
    control_hook = hook('control', '''async def control(context, chat_log):
    from magic_agents.coordination.service import CoordinationError
    def revoked(): raise CoordinationError('stale_owner', 'lost authority')
    chat_log.coordination.runtime.authorize = revoked
    return {'action':'outcome','outcome':{'status':'success','content':{'handle_parser_output':'denied'}}}
''')
    _, _, control, log, _ = setup([parser('target'), control_hook])
    with pytest.raises(CoordinationError) as caught:
        await control.invoke('target', {'value':'original'}, log)
    assert caught.value.code == 'stale_owner'
    assert control.records[0]['outcome']['status'] == 'error'


async def test_external_cancel_joins_child_provider_and_cannot_be_recovered():
    started, joined = asyncio.Event(), asyncio.Event()
    async def held(provider, chat):
        started.set()
        try: await asyncio.Event().wait()
        finally: joined.set()
    child, provider = node('child', held)
    control_hook = hook('control', '''async def control(context, chat_log):
    import asyncio
    try: await context.call('control-child', 'held')
    except asyncio.CancelledError: pass
    return {'action':'outcome','outcome':{'status':'success','content':{'handle_parser_output':'denied'}}}
''')
    rt, scope, control, log, _ = setup([parser('target'), child, control_hook],
        [connection('control', 'child', child.INPUT_HANDLER_USER_MESSAGE)])
    task = asyncio.create_task(control.invoke('target', {'value':'original'}, log))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert joined.is_set() and len(provider.calls) == 1
    assert control.records[0]['outcome'] == {'status':'cancelled'}
    assert [entry['state'] for entry in scope.child_operations.values()] == ['unknown']
    assert (await rt.budget.snapshot())['modelTurns'] == 1


async def test_cumulative_child_tool_limit_blocks_second_effect_and_success_override():
    control_hook = hook('control', '''async def control(context, chat_log):
    await context.call('control-child', 'one')
    await context.call('control-child', 'two')
    return {'action':'pass'}
''')
    rt, scope, control, log, _ = setup([parser('target'), parser('child'), control_hook],
        [connection('control', 'child', 'value')], rt=runtime(limits(maxToolCalls=1),
            tool_estimate=lambda *args: UsageBound(tool_calls=1), tool_usage=lambda *args: UsageBound(tool_calls=1)))
    with pytest.raises(BudgetError):
        await control.invoke('target', {'value':'original'}, log)
    assert (await rt.budget.snapshot())['spent']['tool_calls'] == 1
    assert [entry['state'] for entry in scope.child_operations.values()] == ['completed']


async def test_sync_hook_rejects_before_thread_or_body_but_legacy_still_works(monkeypatch):
    sync = hook('control', "def control(context, chat_log):\n    context.emit.debug({'effect':'sync ran'})\n    return {'action':'pass'}")
    _, _, control, log, _ = setup([parser('target'), sync])
    entered = []
    original = NodeHook._execute_function
    async def observe(self, function, context, child_log):
        entered.append(self.node_id)
        return await original(self, function, context, child_log)
    monkeypatch.setattr(NodeHook, '_execute_function', observe)
    with pytest.raises(CoordinationError, match='require an async'):
        await control.invoke('target', {'value':'original'}, log)
    assert entered == [] and control.records[0]['child'][0]['side_events'] == []
    # Verify the legacy thread-dispatch seam without depending on the host
    # interpreter's executor shutdown behavior (no thread in coordinated mode).
    dispatches = []
    async def thread_dispatch(function, *args, **kwargs):
        dispatches.append(function.__name__)
        return function(*args, **kwargs)
    monkeypatch.setattr(asyncio, 'to_thread', thread_dispatch)
    record = await control.invoke('target', {'value':'legacy'}, ModelAgentRunLog())
    assert record['outcome']['content']['handle_parser_output'] == 'core:legacy'
    assert entered == ['control']
    assert dispatches == ['control']
    assert record['child'][0]['side_events'] == [{'type':'debug','content':{'effect':'sync ran'}}]


async def test_sync_raw_child_tool_rejects_before_thread_submission():
    _, _, control, log, _ = setup([parser('target')])
    effects = []
    def sync_tool(**kwargs):
        effects.append(kwargs)
        return 'effect'
    with pytest.raises(CoordinationError, match='outstanding-work reconciliation'):
        await control.invoke('target', {'value':'original'}, log, tool=sync_tool)
    assert not effects


async def test_invocation_deadline_cannot_be_replaced_by_finish_success(monkeypatch):
    finish = hook('finish', "async def finish(context, chat_log): return {'action':'outcome','outcome':{'status':'success','content':{'handle_parser_output':'late'}}}", phase='onFinish')
    _, _, control, log, _ = setup([parser('target'), finish])
    control.timeout = 0.01
    async def held(self, log):
        await asyncio.Event().wait()
        yield
    monkeypatch.setattr(NodeParser, 'process', held)
    with pytest.raises(CoordinationError) as caught:
        await control.invoke('target', {'value':'original'}, log)
    assert caught.value.code == 'deadline_exceeded'
    assert not control.records[0]['child']


async def test_typed_skills_context_limit_is_terminal_while_unrelated_operation_failure_recovers(monkeypatch):
    recovery = hook('recover', "async def recover(context, chat_log): return {'action':'outcome','outcome':{'status':'success','content':{'handle_parser_output':'fallback'}}}", phase='onError')
    _, _, control, log, _ = setup([parser('target'), recovery])
    code = 'SKILLS_CONTEXT_LIMIT'
    async def failed(self, log):
        raise OperationFailure(code, 'typed limit')
        yield
    monkeypatch.setattr(NodeParser, 'process', failed)
    with pytest.raises(AgentControlError) as caught:
        await control.invoke('target', {'value':'original'}, log)
    assert caught.value.code == code
    assert control.records[0]['protected'] and not control.records[0]['child']
    code = 'ORDINARY_INPUT_ERROR'
    record = await control.invoke('target', {'value':'original'}, log)
    assert record['recovered'] and record['outcome']['content'] == {'handle_parser_output':'fallback'}


async def test_ordinary_operation_timeout_can_recover_without_expiring_invocation(monkeypatch):
    recovery = hook('recover', "async def recover(context, chat_log): return {'action':'outcome','outcome':{'status':'success','content':{'handle_parser_output':'fallback'}}}", phase='onError')
    _, _, control, log, _ = setup([parser('target'), recovery])
    async def failed(self, log):
        raise TimeoutError('ordinary transport timeout')
        yield
    monkeypatch.setattr(NodeParser, 'process', failed)
    record = await control.invoke('target', {'value':'original'}, log)
    assert record['recovered'] and record['outcome']['content'] == {'handle_parser_output':'fallback'}


async def test_ordinary_hook_timeout_exception_keeps_declared_preserve_policy():
    control_hook = hook('control', "async def control(context, chat_log): raise TimeoutError('ordinary Hook request timeout')")
    _, _, control, log, _ = setup([parser('target'), control_hook])
    record = await control.invoke('target', {'value':'original'}, log)
    assert record['outcome']['content']['handle_parser_output'] == 'core:original'
    assert record['child'][0]['diagnostic']['code'] == 'HOOK_TIMEOUT'


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('denial', ['fence', 'skills', 'ordinary'])
async def test_native_controlled_tool_denial_stops_paid_turns_and_ordinary_recovery_remains(stream, denial):
    from magic_llm.agent.async_agent_loop import AsyncAgentLoop
    async def scripted(provider, chat):
        if len(provider.calls) == 1:
            return {'content':None, 'tool_calls':[{'id':'owned-effect','type':'function',
                'function':{'name':'effect','arguments':'{}'}}]}
        return {'content':'ordinary recovery complete'}
    llm, provider = node('llm', scripted, stream=stream)
    recovery = hook('recover', "async def recover(context, chat_log): return {'action':'outcome','outcome':{'status':'success','content':'declared fallback'}}", phase='onError', target='operation')
    edge = EdgeNodeModel(id='operation-llm', source='operation', target='llm',
        sourceHandle='handle-tool-definition', targetHandle='handle-tool-definition-0')
    rt, scope, control, log, _ = setup([llm, parser('operation'), recovery], [edge])
    async def effect():
        if denial == 'fence': raise CoordinationError('stale_owner', 'fence lost')
        if denial == 'skills': raise OperationFailure('SKILLS_CONTEXT_LIMIT', 'limit')
        raise ValueError('ordinary operation failure')
    controlled = control.wrap_tool(effect, llm, edge.targetHandle, log)
    functions = {'effect':controlled}
    schemas = [{'type':'function','function':{'name':'effect','description':'owned effect',
        'parameters':{'type':'object','properties':{}}}}]
    options, executor = await scope.configure_native(llm, schemas, functions)
    executor.propagate_errors(BudgetError, CoordinationError, AgentControlError,
        ProviderAttemptControlError, AgentBudgetExceeded, RequestValidationError, PermissionError)
    loop = AsyncAgentLoop(llm.inputs[llm.INPUT_HANDLER_CLIENT_PROVIDER], tools=schemas,
        tool_functions=functions, tool_executor=executor, **options)
    async def run():
        if stream: return [item async for item in loop.stream(user_input='task')]
        return await loop.run(user_input='task')
    if denial == 'ordinary':
        await run()
        assert len(provider.calls) == 2 and control.records[0]['recovered']
    else:
        with pytest.raises((CoordinationError, AgentControlError)):
            await run()
        assert len(provider.calls) == 1 and control.records[0]['protected']
        assert not control.records[0]['child']
    assert (await rt.budget.snapshot())['modelTurns'] == len(provider.calls)
