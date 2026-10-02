"""Real graph fan-in/fanout never exposes another consumer's private skills."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from magic_llm import MagicLLM
from magic_llm.agent.request import AgentRequestContext
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall, ToolResult
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatResponse import ModelChatResponse
from magic_agents.agt_flow import build, run_agent
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from magic_agents.hooks.hook_relay import HookRelay
from magic_agents.hooks.invocation_control import OperationFailure, error_code
from magic_agents.models.factory.Nodes import LlmNodeModel, SkillsNodeModel
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.skills import (SkillPromptBundle, SkillsCatalogError, create_skills_loader,
    create_request_guard, merge_skill_bundles, safe_loader_result, strip_ephemeral_skills_history)

X_PROMPT='PRIVATE_SHARED_X_PROMPT'
Y_PROMPT='PRIVATE_B_ONLY_Y_PROMPT'
X_PROPS='PRIVATE_X_PROPS'
Y_PROPS='PRIVATE_Y_PROPS'


def entry(identifier,prompt,props,**changes):
    return {'id':identifier,'name':identifier,'description':f'Apply {identifier} to supplied input.',
            'prompt':prompt,'props':{'value':props},'enabled':True,**changes}


def source_bundle(source,entries):
    return SkillPromptBundle.from_model(source,SkillsNodeModel(schema_version=1,skills=entries))


def xy_graph(*, y_enabled=True, y_id='release-notes'):
    nodes=[
        {'id':'input','type':'user_input','data':{}},
        {'id':'client','type':'client','data':{'engine':'openai','model':'fake','api_info':{'private_key':'dummy'}}},
        {'id':'skills-x','type':'skills','data':{'schema_version':1,'skills':[entry('shared-summary',X_PROMPT,X_PROPS)]}},
        {'id':'skills-y','type':'skills','data':{'schema_version':1,'skills':[entry(y_id,Y_PROMPT,Y_PROPS,enabled=y_enabled)]}},
        {'id':'system-a','type':'text','data':{'text':'WORKER_A: write a short summary.'}},
        {'id':'system-b','type':'text','data':{'text':'WORKER_B: write detailed release notes.'}},
        {'id':'llm-a','type':'llm','data':{'stream':False,'json_output':False}},
        {'id':'llm-b','type':'llm','data':{'stream':False,'json_output':False}},
        {'id':'end-a','type':'end','data':{}}, {'id':'end-b','type':'end','data':{}}]
    edges=[]
    def wire(id,source,target,output,input):
        edges.append({'id':id,'source':source,'target':target,'sourceHandle':output,'targetHandle':input})
    # Expected source order is this declared edge order, independent of arrival.
    wire('x-a','skills-x','llm-a','handle-skills','handle-skills')
    wire('x-b','skills-x','llm-b','handle-skills','handle-skills')
    wire('y-b','skills-y','llm-b','handle-skills','handle-skills')
    for name in ('a','b'):
        wire(f'client-{name}','client',f'llm-{name}','handle-client-provider','handle-client-provider')
        wire(f'input-{name}','input',f'llm-{name}','handle_user_message','handle_user_message')
        wire(f'system-{name}',f'system-{name}',f'llm-{name}','handle_text_output','handle-system-context')
        wire(f'output-{name}',f'llm-{name}',f'end-{name}','handle_generated_content','handle_flow_input')
    return {'nodes':nodes,'edges':edges,'contract_config':{'mode':'strict'},'timeout':3.0}


def response(worker,calls=None):
    return ModelChatResponse(id=f'response-{worker}',object='chat.completion',created=0,model='fake',choices=[{
        'index':0,'message':{'role':'assistant','content':None if calls else f'{worker} complete','tool_calls':calls},
        'finish_reason':'tool_calls' if calls else 'stop'}]).model_dump(exclude_none=True)


def load(ids,call_id='same-load'):
    return {'id':call_id,'type':'function','function':{'name':'skills_load','arguments':json.dumps({'skill_ids':ids})}}


def catalog(wire):
    text=next(message['content'] for message in wire['messages'] if message['role']=='system')
    marker='Available skills (metadata only):\n'
    return json.loads(text.split(marker,1)[1].split('\n',1)[0]) if marker in text else []


@pytest.mark.asyncio
async def test_real_shared_graph_concurrent_consumers_and_private_batch_scope(monkeypatch):
    import magic_llm.engine.engine_openai as engine_module
    wire={'A':[],'B':[]}; callbacks=[]; both_started=asyncio.Event()
    class FakeHTTP:
        async def __aenter__(self): return self
        async def __aexit__(self,*args): return False
        async def post_json(self,*,data,**kwargs):
            payload=json.loads(data)
            system=next(message['content'] for message in payload['messages'] if message['role']=='system')
            worker='A' if 'WORKER_A:' in system else 'B'
            wire[worker].append(payload);iteration=len(wire[worker])
            if iteration==1:
                if wire['A'] and wire['B']: both_started.set()
                await asyncio.wait_for(both_started.wait(),1.0)
            if worker=='A':
                calls=[load(['release-notes'])] if iteration==1 else [load(['shared-summary'],'a-valid')] if iteration==2 else None
            else:
                calls=[load(['release-notes','shared-summary'])] if iteration==1 else None
            return response(worker,calls)
    monkeypatch.setattr(engine_module,'AsyncHttpClient',FakeHTTP)
    definition=xy_graph();unchanged=copy.deepcopy(definition)
    graph=build(definition,message='Summarize the supplied changes.')
    async def callback(chat,*args): callbacks.append(copy.deepcopy(chat.messages))
    graph.nodes['client'].client.llm.callback=callback
    events=[event async for event in run_agent(graph)]
    assert definition==unchanged and not graph._contract_report.has_errors()
    assert graph.nodes['llm-a'].inputs['handle-client-provider'] is graph.nodes['llm-b'].inputs['handle-client-provider']
    assert len(wire['A'])==3 and len(wire['B'])==2
    assert [item['id'] for item in catalog(wire['A'][0])]==['shared-summary']
    assert [item['id'] for item in catalog(wire['B'][0])]==['shared-summary','release-notes']
    assert set(catalog(wire['B'][0])[0])=={'id','name','description'}
    assert X_PROMPT not in json.dumps(wire['A'][0]) and Y_PROMPT not in json.dumps(wire['B'][0])
    assert X_PROPS not in json.dumps(wire['A'][0]) and Y_PROPS not in json.dumps(wire['B'][0])
    assert Y_PROMPT not in json.dumps(wire['A']) and Y_PROPS not in json.dumps(wire['A'])
    rejected=next(message for message in wire['A'][1]['messages'] if message['role']=='tool')
    assert 'SKILLS_LOAD_UNKNOWN_ID' in rejected['content'] and Y_PROMPT not in rejected['content']
    a_result=json.loads([message for message in wire['A'][2]['messages'] if message['role']=='tool'][-1]['content'])
    assert set(a_result)=={'schema_version','source_node_id','skills'} and a_result['source_node_id']=='skills-x'
    b_result=json.loads(next(message for message in wire['B'][1]['messages'] if message['role']=='tool')['content'])
    assert b_result['source_node_ids']==['skills-x','skills-y']
    assert b_result['skill_sources']=={'release-notes':'skills-y','shared-summary':'skills-x'}
    assert [item['id'] for item in b_result['skills']]==['release-notes','shared-summary']
    assert [item['prompt'] for item in b_result['skills']]==[Y_PROMPT,X_PROMPT]
    assert X_PROMPT not in str(events) and Y_PROMPT not in str(events)
    assert X_PROPS not in json.dumps(callbacks) and Y_PROPS not in json.dumps(callbacks)
    assert X_PROMPT not in json.dumps(callbacks) and Y_PROMPT not in json.dumps(callbacks)
    a=graph.nodes['llm-a'];b=graph.nodes['llm-b'];x=graph.nodes['skills-x']._bundle
    assert a._skills_bundle is x and b._skills_bundle.skills[0] is x.skills[0]
    assert a._skills_relay is not b._skills_relay and a._skills_request_guard is not b._skills_request_guard
    assert a._capture_internal_state()['skills']['loaded_ids']==['shared-summary']
    assert b._capture_internal_state()['skills']['source_node_ids']==['skills-x','skills-y']
    assert b._capture_internal_state()['skills']['loaded_ids']==['release-notes','shared-summary']
    results=[event['content']['data'] for event in events if event.get('type')=='debug' and event['content'].get('event_type')=='TOOL_RESULT']
    b_events=[event for event in results if event.get('skills_source_node_ids')]
    assert len(b_events)==1 and b_events[0]['skill_sources']==b_result['skill_sources']
    projected=json.loads(b_events[0]['content'])
    assert projected['skill_sources']==b_result['skill_sources'] and projected['skills_source_node_ids']==['skills-x','skills-y']


@pytest.mark.asyncio
async def test_readiness_all_sources_even_disabled_and_stable_order_after_reverse_arrival():
    graph=build(xy_graph(y_enabled=False),message='request')
    dispatcher=GraphEventDispatcher(graph.nodes,graph.edges)
    a=graph.nodes['llm-a'];b=graph.nodes['llm-b']
    assert b._skills_source_node_ids==('skills-x','skills-y')
    for edge in graph.edges:
        if edge.target=='llm-b' and edge.targetHandle!='handle-skills':
            await dispatcher.dispatch_input('llm-b',edge.targetHandle,'ready',edge_id=edge.id)
    y=graph.nodes['skills-y']._bundle;x=graph.nodes['skills-x']._bundle
    await dispatcher.dispatch_input('llm-b','handle-skills',y,edge_id='y-b')
    assert not dispatcher.get_tracker('llm-b').is_ready
    with pytest.raises(OperationFailure): b._prepare_skills()
    await dispatcher.dispatch_input('llm-b','handle-skills',x,edge_id='x-b')
    assert dispatcher.get_tracker('llm-b').is_ready
    assert b._prepare_skills() and b._skills_bundle.source_node_ids==('skills-x','skills-y')
    assert [entry.id for entry in b._skills_bundle.skills]==['shared-summary']
    fresh=b._invocation_factory()
    assert fresh._skills_source_node_ids==('skills-x','skills-y') and not fresh.inputs
    fresh.inputs['handle-skills']=tuple(reversed(b.inputs['handle-skills']))
    fresh._prepare_skills()
    assert fresh._skills_bundle.source_node_ids==('skills-x','skills-y')
    assert fresh._skills_request_guard is None and fresh._skills_relay is None
    assert not a.inputs


@pytest.mark.asyncio
async def test_source_failure_clears_only_failed_source_and_blocks_b(monkeypatch):
    import magic_llm.engine.engine_openai as engine_module
    wire=[]
    class FakeHTTP:
        async def __aenter__(self): return self
        async def __aexit__(self,*args): return False
        async def post_json(self,*,data,**kwargs):
            payload=json.loads(data);wire.append(payload)
            return response('A')
    monkeypatch.setattr(engine_module,'AsyncHttpClient',FakeHTTP)
    graph=build(xy_graph(),message='request')
    async def fail(chat_log):
        raise OperationFailure('TEST_SOURCE_FAILURE','Skills Y failed')
        yield
    graph.nodes['skills-y'].process=fail
    events=[event async for event in run_agent(graph)]
    assert len(wire)==1 and 'WORKER_A:' in json.dumps(wire[0])
    assert all('WORKER_B:' not in json.dumps(request) for request in wire)
    b=graph.nodes['llm-b']
    with pytest.raises(OperationFailure) as failure: b._prepare_skills()
    assert error_code(failure.value)=='SKILLS_SOURCE_FAILED'
    assert X_PROMPT not in str(events) and Y_PROMPT not in str(events)


@pytest.mark.asyncio
async def test_legacy_loop_delivery_keeps_each_source_and_refreshes_it_without_aliasing():
    graph=build(xy_graph(),message='request');b=graph.nodes['llm-b']
    x=graph.nodes['skills-x'];y=graph.nodes['skills-y']
    b.add_parent({'handle-skills':{'content':y._bundle}},'handle-skills','handle-skills')
    b.add_parent({'handle-skills':{'content':x._bundle}},'handle-skills','handle-skills')
    b._prepare_skills()
    assert [entry.id for entry in b._skills_bundle.skills]==['shared-summary','release-notes']
    updated=source_bundle('skills-x',[entry('shared-summary','NEW_X_PROMPT',X_PROPS)])
    b.add_parent({'handle-skills':{'content':updated}},'handle-skills','handle-skills')
    b._prepare_skills()
    assert b._skills_bundle.skills[0].prompt=='NEW_X_PROMPT' and x._bundle.skills[0].prompt==X_PROMPT
    dispatcher=GraphEventDispatcher(graph.nodes,graph.edges)
    dispatcher._clear_bypassed_skills_input('llm-b','y-b')
    assert b.inputs['handle-skills'].source_node_id=='skills-x'
    with pytest.raises(OperationFailure): b._prepare_skills()


@pytest.mark.parametrize('case',['duplicate_edge','duplicate_id','count','prompt','props'])
def test_consumer_conflicts_and_aggregate_limits_fail_before_provider(case):
    definition=xy_graph()
    x=definition['nodes'][2]['data']['skills'];y=definition['nodes'][3]['data']['skills']
    if case=='duplicate_edge':definition['edges'].append({**definition['edges'][0],'id':'duplicate'})
    if case=='duplicate_id':y[0]['id']='shared-summary'
    if case=='count':
        x[:]=[entry(f'x-{i}',X_PROMPT,X_PROPS) for i in range(9)]
        y[:]=[entry(f'y-{i}',Y_PROMPT,Y_PROPS) for i in range(8)]
    if case=='prompt':
        x[:]=[entry(f'x-{i}','x'*12000,{}) for i in range(2)]
        y[:]=[entry('y-1','y'*10000,{})]
    if case=='props':
        x[:]=[entry(f'x-{i}',X_PROMPT,'x'*3500) for i in range(3)]
        y[:]=[entry(f'y-{i}',Y_PROMPT,'y'*3500) for i in range(2)]
    with pytest.raises(ValueError) as failure:build(definition,message='request')
    code='SKILLS_CONNECTION_INVALID' if case=='duplicate_edge' else 'SKILLS_ID_CONFLICT' if case=='duplicate_id' else 'SKILLS_CATALOG_LIMIT'
    assert code in str(failure.value) and X_PROMPT not in str(failure.value) and Y_PROMPT not in str(failure.value)


@pytest.mark.parametrize('disconnected',[False,True])
def test_disabled_or_disconnected_duplicate_id_is_harmless(disconnected):
    definition=xy_graph(y_enabled=disconnected,y_id='shared-summary')
    if disconnected: definition['edges']=[edge for edge in definition['edges'] if edge['id']!='y-b']
    graph=build(definition,message='request');b=graph.nodes['llm-b']
    x=graph.nodes['skills-x']._bundle;y=graph.nodes['skills-y']._bundle
    b.inputs['handle-skills']=x if disconnected else (x,y)
    b._prepare_skills()
    assert [entry.id for entry in b._skills_bundle.skills]==['shared-summary']
    assert Y_PROMPT not in str(b._skills_bundle.safe_summary())


@pytest.mark.parametrize('case',['missing','duplicate_delivery','unexpected','wrong_type'])
def test_runtime_expected_sources_cannot_be_satisfied_by_partial_or_duplicate_delivery(case):
    graph=build(xy_graph(y_enabled=False),message='request');b=graph.nodes['llm-b']
    x=graph.nodes['skills-x']._bundle;y=graph.nodes['skills-y']._bundle
    values={'missing':(x,), 'duplicate_delivery':(x,x), 'unexpected':(x,source_bundle('other',[entry('other','p',{})])), 'wrong_type':(x,{'summary':y.safe_summary()})}
    b.inputs['handle-skills']=values[case]
    with pytest.raises(OperationFailure) as failure:b._prepare_skills()
    assert error_code(failure.value)=='SKILLS_SOURCE_FAILED' and b._skills_bundle is None


@pytest.mark.asyncio
async def test_multisource_loader_fresh_closures_atomic_scope_and_observer_idempotency():
    x=source_bundle('skills-x',[entry('shared-summary',X_PROMPT,X_PROPS)])
    y=source_bundle('skills-y',[entry('release-notes',Y_PROMPT,Y_PROPS)])
    a=create_skills_loader(merge_skill_bundles((x,)),50000)
    b=create_skills_loader(merge_skill_bundles((x,y)),50000)
    with pytest.raises(ValueError):await a(skill_ids=['release-notes'])
    success=await b(skill_ids=['release-notes','shared-summary'])
    assert success['skill_sources']=={'release-notes':'skills-y','shared-summary':'skills-x'}
    canonical=ToolResult(tool_call_id='same',name='skills_load',content=ToolExecutor.serialize_output(success))
    projected=safe_loader_result(canonical)
    assert safe_loader_result(projected).content==projected.content
    assert Y_PROMPT in canonical.content and Y_PROMPT not in projected.content
    relay=HookRelay(node_id='B',skills_source_node_ids=['skills-x','skills-y'],skills_available_ids=['shared-summary','release-notes'],
                    skills_source_by_id={'shared-summary':'skills-x','release-notes':'skills-y'})
    relay.on_tool_complete(projected,SimpleNamespace(iteration=0))
    event=relay.get_collected_tool_data_for_yield()[0]['data']
    assert event['skill_sources']==success['skill_sources'] and json.loads(event['content'])['skill_sources']==success['skill_sources']
    failed=ToolResult(tool_call_id='bad',name='skills_load',content='{"error":"safe"}',is_error=True)
    assert safe_loader_result(safe_loader_result(failed)).content==safe_loader_result(failed).content
    assert X_PROMPT not in str(event) and Y_PROPS not in str(event)


@pytest.mark.parametrize('native',['openai','anthropic','google'])
def test_multisource_provenance_cleans_native_history_atomically(native):
    marker={'source':'builtin_skills','ephemeral':True,'skills_source_node_ids':['skills-x','skills-y']}
    history=[{'role':'assistant','tool_calls':[load(['shared-summary'],'loaded'),{'id':'keep','function':{'name':'lookup','arguments':'{}'}}],**marker},
             {'role':'tool','tool_call_id':'loaded','content':Y_PROMPT}, {'role':'tool','tool_call_id':'keep','content':'keep result'}]
    if native=='anthropic':history[1]={'role':'user','content':[{'type':'tool_result','tool_use_id':'loaded','content':Y_PROMPT},{'type':'tool_result','tool_use_id':'keep','content':'keep result'}]};history.pop(2)
    if native=='google':
        history[0]['gemini_parts']=[{'functionCall':{'name':'skills_load','args':{}}},{'functionCall':{'name':'lookup','args':{}}}]
        history[1]={'role':'user','content':[{'functionResponse':{'id':'loaded','response':{'result':Y_PROMPT}}},{'functionResponse':{'id':'keep','response':{'result':'keep result'}}}]};history.pop(2)
    original=copy.deepcopy(history);clean=strip_ephemeral_skills_history(history)
    assert history==original and Y_PROMPT not in json.dumps(clean) and 'keep result' in json.dumps(clean)
    assert 'loaded' not in json.dumps(clean)


@pytest.mark.asyncio
async def test_repeated_real_graph_invocation_refreshes_state_and_source_failure(monkeypatch):
    import magic_llm.engine.engine_openai as engine_module
    wire={'A':[],'B':[]}
    class FakeHTTP:
        async def __aenter__(self): return self
        async def __aexit__(self,*args): return False
        async def post_json(self,*,data,**kwargs):
            payload=json.loads(data)
            system=next(message['content'] for message in payload['messages'] if message['role']=='system')
            worker='A' if 'WORKER_A:' in system else 'B';wire[worker].append(payload)
            prior=[message for message in payload['messages'] if message['role']=='tool']
            return response(worker,None if prior else [load(['shared-summary'] if worker=='A' else ['release-notes','shared-summary'])])
    monkeypatch.setattr(engine_module,'AsyncHttpClient',FakeHTTP)
    graph=build(xy_graph(),message='request')
    [event async for event in run_agent(graph)]
    a=graph.nodes['llm-a'];b=graph.nodes['llm-b'];first_guard=a._skills_request_guard;first_relay=a._skills_relay
    first_b_guard=b._skills_request_guard
    # Built graph instances cache completed nodes. Each request builds a fresh
    # graph while reusing the same actual client, as the API invocation does.
    shared_client=graph.nodes['client'].client
    second_graph=build(xy_graph(),message='second request')
    second_graph.nodes['client'].client=shared_client
    second=[event async for event in run_agent(second_graph)]
    a=second_graph.nodes['llm-a'];b=second_graph.nodes['llm-b']
    assert len(wire['A'])==4 and len(wire['B'])==4
    assert all(X_PROMPT not in json.dumps(wire[worker][2]) and Y_PROMPT not in json.dumps(wire[worker][2]) for worker in ('A','B'))
    assert a._skills_request_guard is not first_guard and a._skills_relay is not first_relay
    assert b._skills_request_guard is not first_b_guard
    available=[event['content']['data']['skills'] for event in second if event.get('type')=='debug' and event['content'].get('event_type')=='SKILLS_AVAILABLE']
    assert len(available)==2 and all(summary['loaded_ids']==[] for summary in available)
    async def fail(chat_log):
        raise OperationFailure('TEST_SOURCE_FAILURE','Skills Y failed')
        yield
    failed_graph=build(xy_graph(),message='third request')
    failed_graph.nodes['client'].client=shared_client
    b=failed_graph.nodes['llm-b']
    # A previous delivery cannot satisfy the failed source on a new request.
    b.inputs['handle-skills']=second_graph.nodes['llm-b'].inputs['handle-skills']
    failed_graph.nodes['skills-y'].process=fail
    failed=[event async for event in run_agent(failed_graph)]
    assert len(wire['A'])==6 and len(wire['B'])==4
    assert b._skills_bundle is None and b._skills_request_guard is None
    assert b.inputs['handle-skills'].source_node_id=='skills-x'
    with pytest.raises(OperationFailure) as failure: b._prepare_skills()
    assert error_code(failure.value)=='SKILLS_SOURCE_FAILED'
    assert X_PROMPT not in str(failed) and Y_PROMPT not in str(failed)


@pytest.mark.asyncio
async def test_all_disabled_multiple_sources_need_delivery_but_install_no_loader():
    graph=build(xy_graph(y_enabled=False),message='request')
    x=source_bundle('skills-x',[entry('shared-summary',X_PROMPT,X_PROPS,enabled=False)])
    y=graph.nodes['skills-y']._bundle;b=graph.nodes['llm-b']
    b.inputs['handle-skills']=(y,x)
    assert not b._prepare_skills()
    assert b._skills_bundle.source_node_ids==('skills-x','skills-y') and b._skills_bundle.skills==()
    schemas=[];functions={};chat=ModelChat('base')
    b._install_skills(graph.nodes['client'].client,chat,schemas,functions,SimpleNamespace(tool_schemas=[]))
    assert not schemas and not functions and b._skills_request_guard is None
    assert chat.messages==[{'role':'system','content':'base'}]


def test_multi_source_call_provenance_and_observer_projection_stay_private():
    relay=HookRelay(node_id='B',skills_source_node_ids=['skills-x','skills-y'],skills_available_ids=['shared-summary','release-notes'],
                    skills_source_by_id={'shared-summary':'skills-x','release-notes':'skills-y'})
    relay.on_tool_start('skills_load','load',{'skill_ids':['release-notes','shared-summary']},SimpleNamespace(iteration=0))
    call=relay.get_collected_tool_data_for_yield()[0]['data']
    assert call['skills_source_node_ids']==['skills-x','skills-y']
    assert call['skill_sources']=={'release-notes':'skills-y','shared-summary':'skills-x'}
    chat=ModelChat();chat.add_user_message('request')
    guard=create_request_guard(20000,source_node_ids=('skills-x','skills-y'))
    def context():return AgentRequestContext(chat=chat,tools=[],tool_choice=None,provider='openai',model='fake',generation_options={})
    guard(context())
    chat.messages += [{'role':'assistant','tool_calls':[load(['release-notes'])]}, {'role':'tool','tool_call_id':'same-load','content':Y_PROMPT}]
    canonical=copy.deepcopy(chat.messages);guard(context());projected=chat.observer_projection()
    assert chat.messages==canonical and Y_PROMPT not in json.dumps(projected.messages)
    assert projected.extra_args['skills']['source_node_ids']==['skills-x','skills-y']
