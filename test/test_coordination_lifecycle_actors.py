"""Controlled participants retain ownership through final callbacks and children."""
import asyncio
import copy
import json

import pytest

from magic_agents.coordination.budget import UsageBound
from magic_agents.coordination.context import _Binding, _OwnerGuard, _OWNERS
from magic_agents.coordination.service import CoordinationError, LIFECYCLE_CHECKPOINT_KEY
from magic_agents.models.coordination import CoordinationPolicy
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.util.coordination_validation import validate_coordination_definition
from magic_llm.engine.base_chat import RetryConfig
from magic_llm.model import ModelChat

from test.test_coordination_graph import collect, graph, limits, node, runtime, send_call
from test.test_coordination_lifecycle import hook, connection
from test.test_coordination_skills import skills, skill_call, PRIVATE


pytestmark = pytest.mark.asyncio


def actor(scope, role='images'):
    return scope.service._actors[scope.service._roles[role]]


def add_hooks(g, *callbacks, edges=()):
    g.nodes.update({item.node_id:item for item in callbacks})
    g.edges.extend(edges)
    return g


@pytest.mark.parametrize('stream', [False, True])
async def test_finish_child_and_concurrent_arrival_commit_effective_state_before_seal(stream):
    started, release = asyncio.Event(), asyncio.Event()
    rt = runtime()
    async def research(provider, chat):
        if len(provider.calls) == 1:
            await started.wait()
            a = actor(rt.scopes[0])
            assert a.owner_token is not None and a.state == 'running'
            assert rt.scopes[0].service._state == 'open'
            return send_call()
        assert rt.scopes[0].service._pending(actor(rt.scopes[0]))
        release.set()
        return {'content':'research complete'}
    async def images(provider, chat):
        if len(provider.calls) == 1: return {'content':'generic'}
        assert any(m.get('role') == 'assistant' and m.get('content') == 'generic' for m in chat.messages)
        assert 'approved:generic' not in repr(chat.messages)
        assert 'targeted image for slide 4' in chat.messages[-1]['content']
        return {'content':'generic + targeted'}
    async def child(provider, chat):
        started.set()
        await release.wait()
        return {'content':'owned child result'}
    async def author(provider, chat):
        assert rt.scopes[0].service._state == 'sealed'
        assert 'approved:generic + targeted' in repr(chat.messages)
        saved = actor(rt.scopes[0]).checkpoint
        assert saved[LIFECYCLE_CHECKPOINT_KEY]['effectiveOutput']['handle_generated_content']['content'] == 'approved:generic + targeted'
        return {'content':'public final'}
    a, pa = node('research', research, peer='images', stream=stream)
    b, pb = node('images', images, peer='research', stream=stream)
    c, pc = node('author', author, stream=stream)
    d, pd = node('child', child, stream=stream)
    finish = hook('finish', '''async def finish(context, chat_log):
    value = context.outcome['content']['handle_generated_content']
    if value == 'generic': await context.call('finish-child', 'postprocess generic')
    content = dict(context.outcome['content'])
    content['handle_generated_content'] = 'approved:' + value
    return {'action':'outcome','outcome':{'status':'success','content':content}}
''', phase='onFinish', target='images')
    g = add_hooks(graph({'research':a,'images':b,'author':c,'child':d}), finish,
                  edges=[connection('finish','child',d.INPUT_HANDLER_USER_MESSAGE)])
    result, events = await collect(g, rt)
    assert not result['has_errors'], events
    assert (len(pa.calls),len(pb.calls),len(pc.calls),len(pd.calls)) == (2,2,1,1)
    saved = actor(rt.scopes[0]).checkpoint
    assert saved['output_candidate'] == 'generic + targeted'
    assert len(saved['consumed_message_ids']) == 1
    lifecycle = saved[LIFECYCLE_CHECKPOINT_KEY]
    assert lifecycle['invocation']['original_outcome']['content']['handle_generated_content'] == 'generic + targeted'
    assert len(lifecycle['childOperations']) == 1 and lifecycle['childOperations'][0]['state'] == 'completed'
    assert lifecycle['childOperations'][0]['ownerActorId'] == actor(rt.scopes[0]).id
    assert (await rt.budget.snapshot())['modelTurns'] == 6


@pytest.mark.parametrize('stream', [False, True])
async def test_controlled_skills_wake_preserves_both_private_envelopes(stream):
    rt = runtime(authorize_skills=lambda *args:None)
    async def research(provider, chat):
        if len(provider.calls) == 1:
            scope=rt.scopes[0]
            async with scope.service._condition:
                while actor(scope).state != 'quiescent': await scope.service._condition.wait()
            assert LIFECYCLE_CHECKPOINT_KEY in actor(scope).checkpoint
            assert 'coordination_skills' in actor(scope).checkpoint
            return send_call(wake=True)
        return {'content':'research done'}
    async def images(provider, chat):
        if len(provider.calls) == 1: return skill_call()
        assert PRIVATE in repr(chat.messages)
        assert 'approved:' not in repr(chat.messages)
        return {'content':'generic' if len(provider.calls) == 2 else 'targeted'}
    async def author(provider, chat):
        assert PRIVATE not in repr(chat.messages)
        assert 'approved:targeted' in repr(chat.messages)
        return {'content':'public final'}
    a, pa=node('research',research,peer='images',stream=stream)
    b, pb=node('images',images,peer='research',stream=stream)
    b.inputs[b.INPUT_HANDLER_SKILLS]=skills(); b._skills_source_node_ids=('skills',)
    c, pc=node('author',author,stream=stream)
    finish=hook('finish', '''async def finish(context, chat_log):
    content=dict(context.outcome['content'])
    content['handle_generated_content']='approved:' + content['handle_generated_content']
    return {'action':'outcome','outcome':{'status':'success','content':content}}
''', phase='onFinish', target='images')
    result, events=await collect(add_hooks(graph({'research':a,'images':b,'author':c}),finish),rt)
    assert not result['has_errors'],events
    saved=actor(rt.scopes[0]).checkpoint
    assert saved['output_candidate']=='targeted' and 'coordination_skills' in saved and LIFECYCLE_CHECKPOINT_KEY in saved
    assert (len(pa.calls),len(pb.calls),len(pc.calls))==(2,3,1)


@pytest.mark.parametrize('policy', ['fail_group','publish_partial'])
async def test_preloop_short_circuit_settles_actor_without_fabricated_checkpoint(policy):
    rt=runtime()
    async def peer(provider,chat): return {'content':'peer done'}
    async def forbidden(provider,chat): raise AssertionError('Short circuited actor must not dispatch')
    async def author(provider,chat):
        assert 'lifecycle_short_circuit' in repr(chat.messages)
        return {'content':'partial final'}
    a,pa=node('research',peer,peer='images');b,pb=node('images',forbidden,peer='research');c,pc=node('author',author)
    start=hook('start', "async def start(context, chat_log): return {'action':'outcome','outcome':{'status':'success','content':{'handle_generated_content':'skipped'}}}",target='images')
    g=add_hooks(graph({'research':a,'images':b,'author':c}),start)
    g.coordination=CoordinationPolicy(enabled=True,allowedParticipants=['research','images'],limits=limits(),failurePolicy=policy)
    result,events=await collect(g,rt)
    assert not pb.calls and actor(rt.scopes[0]).checkpoint is None
    assert actor(rt.scopes[0]).failure_reason=='lifecycle_short_circuit'
    if policy=='fail_group': assert result['has_errors'] and not pc.calls
    else: assert not result['has_errors'] and len(pc.calls)==1,events


@pytest.mark.parametrize('fallback', [False, True])
async def test_original_owner_is_checked_before_child_retry_or_fallback(fallback):
    rt=runtime()
    async def basic(provider,chat): return {'content':'ok'}
    a,_=node('research',basic,peer='images');b,_=node('images',basic,peer='research')
    g=graph({'research':a,'images':b,'author':node('author',basic)[0]})
    scope=rt.enter_graph(g)
    caller=await scope.service.activate('images')
    scope.bindings['images']=_Binding(caller)
    owner_token=_OWNERS.set((_OwnerGuard(scope.service,caller),))
    try:
        control=scope.attempt_control('child')
    finally:
        _OWNERS.reset(owner_token)
    async def fail_and_replace(provider,chat):
        # Replace only the mutable binding with a newly issued owner. Old child
        # authority must still be rejected on its next physical dispatch.
        current=actor(scope)
        current.state='queued';current.owner_token=None
        newer=await scope.service.activate('images')
        scope.bindings['images']=_Binding(newer)
        raise RuntimeError('retryable fake transport error')
    child,provider=node('child',fail_and_replace)
    provider.retry_config=RetryConfig(1 if fallback else 2,0)
    alternate,backup=node('backup',basic)
    if fallback: provider.fallback=alternate.inputs[alternate.INPUT_HANDLER_CLIENT_PROVIDER]
    with pytest.raises(CoordinationError):
        await provider.async_generate(ModelChat(),provider_attempt_control=control)
    assert len(provider.calls)==1 and not backup.calls
    assert (await rt.budget.snapshot())['modelTurns']==1


async def test_runtime_rejects_unsupported_participant_delivery_before_provider_work():
    rt=runtime()
    async def basic(provider,chat): return {'content':'ok'}
    a,pa=node('research',basic,peer='images');b,pb=node('images',basic,peer='research');c,pc=node('author',basic)
    delivery=hook('deliver',"async def deliver(context, chat_log): return {'action':'pass'}",phase='onDeliver',target='images')
    with pytest.raises(CoordinationError,match='onDeliver'):
        await collect(add_hooks(graph({'research':a,'images':b,'author':c}),delivery),rt)
    assert not pa.calls and not pb.calls and not pc.calls


@pytest.mark.parametrize('stream', [False, True])
async def test_input_replacement_and_parallel_children_keep_distinct_actor_authority(stream):
    both_started=asyncio.Event(); children=[]
    rt=runtime(limits(maxConcurrentModelTurns=2))
    async def peer(provider,chat):
        assert chat.messages[-1]['content']=='replaced'
        return {'content':'candidate'}
    async def child(provider,chat):
        label=chat.messages[-1]['content'];children.append(label)
        if len(children)==2:
            assert (await rt.budget.snapshot())['activeModels']==2
            assert all(actor(rt.scopes[0],role).owner_token is not None for role in ['images','research'])
            both_started.set()
        await both_started.wait()
        return {'content':label+' result'}
    async def author(provider,chat):
        assert 'images result' in repr(chat.messages) and 'research result' in repr(chat.messages)
        return {'content':'final'}
    a,pa=node('research',peer,peer='images',stream=stream);b,pb=node('images',peer,peer='research',stream=stream)
    c,pc=node('author',author,stream=stream);d,pd=node('child',child,stream=stream)
    g=graph({'research':a,'images':b,'author':c,'child':d})
    for role in ['research','images']:
        start=hook('start-'+role,"async def start(context, chat_log): return {'action':'input','content':{'handle_user_message':'replaced'}}",target=role)
        finish=hook('finish-'+role,'''async def finish(context, chat_log):
    role=context.event['node_id']
    child=await context.call('finish-'+role+'-child',role)
    content=dict(context.outcome['content'])
    content['handle_generated_content']=child['outcome']['content']['handle_generated_content']
    return {'action':'outcome','outcome':{'status':'success','content':content}}
''',phase='onFinish',target=role)
        add_hooks(g,start,finish,edges=[connection(finish.node_id,'child',d.INPUT_HANDLER_USER_MESSAGE)])
    result,events=await collect(g,rt)
    assert not result['has_errors'],events
    assert sorted(children)==['images','research']
    assert (len(pa.calls),len(pb.calls),len(pd.calls),len(pc.calls))==(1,1,2,1)
    for role in ['research','images']:
        current=actor(rt.scopes[0],role)
        journal=current.checkpoint[LIFECYCLE_CHECKPOINT_KEY]['childOperations']
        assert len(journal)==1 and journal[0]['ownerActorId']==current.id
        assert role+' result' in repr(journal[0]['result'])


async def test_cancelling_held_finish_joins_provider_and_never_publishes_candidate():
    started,joined=asyncio.Event(),asyncio.Event()
    rt=runtime()
    async def peer(provider,chat):return {'content':'tentative'}
    async def child(provider,chat):
        started.set()
        try:await asyncio.Event().wait()
        finally:joined.set()
    a,_=node('research',peer,peer='images');b,_=node('images',peer,peer='research')
    c,pc=node('author',peer);d,pd=node('child',child)
    finish=hook('finish',"async def finish(context, chat_log):\n    await context.call('finish-child','hold')\n    return {'action':'pass'}",phase='onFinish',target='images')
    g=add_hooks(graph({'research':a,'images':b,'author':c,'child':d}),finish,
        edges=[connection('finish','child',d.INPUT_HANDLER_USER_MESSAGE)])
    task=asyncio.create_task(collect(g,rt))
    await asyncio.wait_for(started.wait(),1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert joined.is_set() and len(pd.calls)==1 and not pc.calls
    assert actor(rt.scopes[0]).output is None
    assert (await rt.budget.snapshot())['cancelled']


async def test_finish_child_budget_denial_is_terminal_through_preserve_hook():
    rt=runtime(limits(maxModelTurns=2))
    peer_ready=asyncio.Event()
    async def research(provider,chat):
        peer_ready.set();return {'content':'done'}
    async def images(provider,chat):
        await peer_ready.wait();return {'content':'candidate'}
    async def forbidden(provider,chat):raise AssertionError('Budget denial must precede child/author work')
    a,_=node('research',research,peer='images');b,_=node('images',images,peer='research')
    c,pc=node('author',forbidden);d,pd=node('child',forbidden)
    finish=hook('finish',"async def finish(context, chat_log):\n    await context.call('finish-child','denied')\n    return {'action':'pass'}",phase='onFinish',target='images')
    g=add_hooks(graph({'research':a,'images':b,'author':c,'child':d}),finish,
        edges=[connection('finish','child',d.INPUT_HANDLER_USER_MESSAGE)])
    result,_=await collect(g,rt)
    assert result['has_errors'] and not pd.calls and not pc.calls
    assert actor(rt.scopes[0]).output is None
    assert (await rt.budget.snapshot())['modelTurns']==2


async def test_child_scope_captured_owner_survives_task_context_exit():
    rt=runtime()
    async def basic(provider,chat):return {'content':'ok'}
    a,_=node('research',basic,peer='images');b,_=node('images',basic,peer='research');c,_=node('author',basic)
    scope=rt.enter_graph(graph({'research':a,'images':b,'author':c}))
    caller=await scope.service.activate('images');scope.bindings['images']=_Binding(caller)
    token=_OWNERS.set((_OwnerGuard(scope.service,caller),))
    try:
        from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
        child,provider=node('child',basic)
        inner=rt.enter_graph(AgentFlowModel(type='graph',nodes={'child':child},edges=[]),parent=scope,path=('nested',))
    finally:_OWNERS.reset(token)
    actor(scope).owner_token=object()
    with pytest.raises(CoordinationError):
        await provider.async_generate(ModelChat(),provider_attempt_control=inner.attempt_control('child'))
    assert not provider.calls


async def test_authored_validation_enables_only_proven_participant_callback_cells():
    from test.test_coordination_models import graph as definition_graph
    definition=definition_graph()
    definition['nodes'].append({'id':'control','type':'hook','data':{
        'lifecycle_event':'onFinish','target_node_id':'a',
        'function_template':"async def finish(context, chat_log): return {'action':'pass'}"}})
    assert validate_coordination_definition(definition)==[]
    definition['nodes'][-1]['data']['lifecycle_event']='onDeliver'
    assert validate_coordination_definition(definition)[0].code=='unsupported_coordination_topology'
    definition['nodes'][-1]['data']['lifecycle_event']='onFinish'
    definition['nodes'][-1]['data']['function_template']="def finish(context, chat_log): return {'action':'pass'}"
    assert validate_coordination_definition(definition)[0].code=='unsupported_coordination_topology'


@pytest.mark.parametrize('fallback', [False,True])
async def test_real_hook_child_cannot_dispatch_retry_after_initiating_owner_is_replaced(fallback):
    rt=runtime()
    async def basic(provider,chat):return {'content':'candidate'}
    async def child(provider,chat):
        scope=rt.scopes[0];current=actor(scope)
        current.state='queued';current.owner_token=None
        replacement=await scope.service.activate('images')
        scope.bindings['images']=_Binding(replacement)
        raise RuntimeError('transient child failure')
    a,_=node('research',basic,peer='images');b,_=node('images',basic,peer='research');c,pc=node('author',basic)
    d,pd=node('child',child);alternative,pf=node('fallback',basic)
    pd.retry_config=RetryConfig(1 if fallback else 2,0)
    if fallback:pd.fallback=alternative.inputs[alternative.INPUT_HANDLER_CLIENT_PROVIDER]
    finish=hook('finish',"async def finish(context, chat_log):\n    await context.call('finish-child','owned')\n    return {'action':'pass'}",phase='onFinish',target='images')
    g=add_hooks(graph({'research':a,'images':b,'author':c,'child':d}),finish,
        edges=[connection('finish','child',d.INPUT_HANDLER_USER_MESSAGE)])
    result,_=await collect(g,rt)
    assert result['has_errors'] and len(pd.calls)==1 and not pf.calls and not pc.calls
    assert (await rt.budget.snapshot())['modelTurns']==3


async def test_lifecycle_checkpoint_capacity_failure_settles_accepted_request():
    started,release=asyncio.Event(),asyncio.Event();rt=runtime()
    async def research(provider,chat):
        if len(provider.calls)==1:
            await started.wait()
            message=send_call()
            message['tool_calls'][0]['function']['arguments']=json.dumps({'agentRef':'images','message':'bounded accepted request','options':{'expectReply':True}})
            return message
        release.set();return {'content':'waiting for required reply'}
    async def images(provider,chat):return {'content':'candidate'}
    async def child(provider,chat):
        scope=rt.scopes[0]
        saved=actor(scope).checkpoint
        scope.service._checkpoint_bytes=len(json.dumps(saved,ensure_ascii=False).encode())+4000
        started.set();await release.wait()
        return {'content':'owned completion '+('x'*10000)}
    a,_=node('research',research,peer='images');b,_=node('images',images,peer='research');c,pc=node('author',images);d,_=node('child',child)
    finish=hook('finish',"async def finish(context, chat_log):\n    await context.call('finish-child','bounded child')\n    return {'action':'pass'}",phase='onFinish',target='images')
    g=add_hooks(graph({'research':a,'images':b,'author':c,'child':d}),finish,
        edges=[connection('finish','child',d.INPUT_HANDLER_USER_MESSAGE)])
    result,_=await collect(g,rt)
    assert result['has_errors'] and not pc.calls
    scope=rt.scopes[0]
    assert actor(scope).failure_reason=='checkpoint_capacity_exceeded'
    assert any(item.request_id for item in scope.service._messages.values())
    assert all(not member.requests for member in scope.service._actors.values())
    assert LIFECYCLE_CHECKPOINT_KEY not in actor(scope).checkpoint


@pytest.mark.parametrize('corruption', ['boolean_version', 'canonical_digest', 'unknown_child'])
async def test_incompatible_lifecycle_checkpoint_rejects_wake_before_dispatch(corruption):
    rt = runtime()
    async def research(provider, chat):
        if len(provider.calls) == 1:
            scope = rt.scopes[0]
            async with scope.service._condition:
                while actor(scope).state != 'quiescent':
                    await scope.service._condition.wait()
                saved = actor(scope).checkpoint[LIFECYCLE_CHECKPOINT_KEY]
                if corruption == 'boolean_version': saved['schemaVersion'] = True
                elif corruption == 'canonical_digest': saved['canonicalDigest'] = '0' * 64
                else:
                    saved['childOperations'].append({
                        'state': 'completed', 'node_id': 'missing', 'operation_id': 'owned-child',
                        'digest': '1' * 64, 'ownerActorId': actor(scope).id,
                        'ownerActivationId': saved['activationId'], 'result': 'private child result',
                    })
            return send_call(wake=True)
        return {'content': 'research done'}
    async def basic(provider, chat): return {'content': 'candidate'}
    a, _ = node('research', research, peer='images')
    b, pb = node('images', basic, peer='research')
    c, pc = node('author', basic)
    finish = hook('finish', "async def finish(context, chat_log): return {'action':'pass'}",
                  phase='onFinish', target='images')
    result, _ = await collect(add_hooks(graph({'research': a, 'images': b, 'author': c}), finish), rt)
    assert result['has_errors'] and len(pb.calls) == 1 and not pc.calls
    assert not rt.scopes[0].child_operations


async def test_completed_child_identity_replays_only_for_its_initiating_actor():
    rt = runtime()
    async def basic(provider, chat): return {'content': 'candidate'}
    a, _ = node('research', basic, peer='images')
    b, _ = node('images', basic, peer='research')
    c, _ = node('author', basic)
    d, _ = node('child', basic)
    scope = rt.enter_graph(graph({'research': a, 'images': b, 'author': c, 'child': d}))
    one = await scope.service.activate('images')
    two = await scope.service.activate('research')
    calls = []
    async def operation():
        calls.append('effect')
        return {'content': 'owned result'}
    token = _OWNERS.set((_OwnerGuard(scope.service, one),))
    try:
        first = await scope.run_child_operation('child', 'stable-child', 'input', operation)
        replay = await scope.run_child_operation('child', 'stable-child', 'input', operation)
        assert replay == first and len(calls) == 1
    finally: _OWNERS.reset(token)
    token = _OWNERS.set((_OwnerGuard(scope.service, two),))
    try:
        with pytest.raises(CoordinationError) as denied:
            await scope.run_child_operation('child', 'stable-child', 'input', operation)
        assert denied.value.code == 'operation_owner_mismatch' and len(calls) == 1
        await scope.run_child_operation('child', 'different-child', 'input', operation)
        assert len(calls) == 2
    finally: _OWNERS.reset(token)
