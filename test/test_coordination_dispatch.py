"""Physical dispatches share captured ownership and one finite ledger."""
import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from magic_agents.coordination.budget import BudgetError, UsageBound, WorkgroupBudget
from magic_agents.coordination.dispatch import DispatchSession, invocation_charge
from magic_agents.coordination.service import CoordinationError
from test.test_coordination_graph import limits

pytestmark = pytest.mark.asyncio


def scope(**overrides):
    owner = [True]
    budget = WorkgroupBudget(limits(**overrides))
    def guard():
        budget._open()
        if not owner[0]: raise CoordinationError('stale_owner', 'Original owner was replaced')
    runtime = SimpleNamespace(authorize=lambda:None,
        authorize_external=lambda path, kind, request:None,
        external_estimate=lambda *args:UsageBound(tool_calls=1, cost=Decimal('1')),
        external_usage=lambda *args:UsageBound(tool_calls=1, cost=Decimal('0.5')))
    return SimpleNamespace(path=('inner',), budget=budget, runtime=runtime,
                           capture_guard=lambda node_id:guard, owner=owner)


async def test_full_private_request_reaches_admission_and_journal_only_retains_digest():
    context=scope(); seen=[]
    context.runtime.authorize_external=lambda *args:seen.append(args)
    session=DispatchSession(context,'fetch')
    async def effect():return {'answer':'ok'}
    request={'url':'https://example.invalid', 'headers':{'Authorization':'private-key'},
             'json':{'properties':{'token':{'description':'semantic secret-named field'}}}}
    assert await session.call('http.fetch',request,effect)=={'answer':'ok'}
    assert seen[0][2]==request and 'private-key' not in repr(context._external_operations)
    totals=await context.budget.snapshot()
    assert totals['spent']['tool_calls']==1 and totals['spent']['cost']=='0.5'
    assert totals['activeJobs']==0


async def test_invocation_credit_used_once_across_parallel_siblings_without_held_slot():
    context=scope(maxConcurrentJobs=1);session=DispatchSession(context,'fetch')
    async def effect():await asyncio.sleep(0);return {'ok':True}
    async with invocation_charge(context,'native-call-1',context.capture_guard('fetch')):
        await asyncio.gather(session.call('http.fetch',{'n':1},effect),
                             session.call('http.fetch',{'n':2},effect))
    totals=await context.budget.snapshot()
    assert totals['spent']['tool_calls']==2 and totals['spent']['cost']=='1.0'
    assert totals['activeJobs']==0


async def test_forged_reused_or_foreign_budget_credit_cannot_admit_zero_counter_job():
    context=scope(); budget=context.budget
    await budget.charge_tool_call('valid')
    await budget.reserve('first',UsageBound(),kind='job',charged_tool_operation_id='valid')
    await budget.settle('first',UsageBound())
    for target,key in [(budget,'valid'),(budget,'missing'),(budget.child(limits()),'valid')]:
        with pytest.raises(BudgetError,match='credit'):
            await target.reserve('second',UsageBound(),kind='job',charged_tool_operation_id=key)


async def test_captured_guard_denies_a_queued_dispatch_after_owner_replacement():
    context=scope(maxConcurrentJobs=1);session=DispatchSession(context,'fetch');calls=[]
    await context.budget.reserve('held',UsageBound(tool_calls=1),kind='job')
    async def effect():calls.append(True)
    task=asyncio.create_task(session.call('http.fetch',{},effect))
    await asyncio.sleep(0)
    context.owner[0]=False
    await context.budget.settle('held',UsageBound(tool_calls=1))
    with pytest.raises(CoordinationError,match='Original owner'):await task
    assert not calls and next(iter(context._external_operations.values()))['state']=='not_dispatched'


async def test_cancelled_started_effect_releases_slot_and_retains_uncertain_exposure():
    context=scope();session=DispatchSession(context,'fetch');started=asyncio.Event();joined=asyncio.Event()
    async def effect():
        started.set()
        try:await asyncio.Event().wait()
        finally:joined.set()
    task=asyncio.create_task(session.call('http.fetch',{},effect))
    await started.wait();task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    totals=await context.budget.snapshot()
    assert joined.is_set() and totals['activeJobs']==0
    assert totals['uncertain']['cost']!='0'
    assert next(iter(context._external_operations.values()))['state']=='unknown'


async def test_unknown_remote_failure_is_terminal_and_replay_cannot_redispatch():
    context=scope();session=DispatchSession(context,'fetch');calls=[]
    async def effect():calls.append(True);raise ConnectionError('remote acknowledgement lost')
    with pytest.raises(CoordinationError) as failed:
        await session.call('http.fetch',{},effect,operation_id='stable')
    assert failed.value.code=='external_outcome_unknown'
    with pytest.raises(CoordinationError) as replay:
        await session.call('http.fetch',{},effect,operation_id='stable')
    assert replay.value.code=='operation_unresolved' and len(calls)==1


async def test_missing_adapter_or_exhausted_cap_rejects_before_effect():
    context=scope(maxToolCalls=1);calls=[]
    async def effect():calls.append(True);return None
    await context.budget.charge_tool_call('spent')
    with pytest.raises(BudgetError):await DispatchSession(context,'fetch').call('http.fetch',{},effect)
    context.runtime.authorize_external=None
    with pytest.raises(CoordinationError):DispatchSession(context,'fetch')
    assert not calls


def node_scope(target, **overrides):
    from test.test_coordination_graph import runtime
    from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
    from magic_agents.models.model_agent_run_log import ModelAgentRunLog
    rt=runtime(authorize_external=lambda *args:None,
        external_estimate=lambda *args:UsageBound(tool_calls=1,cost=Decimal('.1')),
        external_usage=lambda *args:UsageBound(tool_calls=1,cost=Decimal('.01')), **overrides)
    context=rt.enter_graph(AgentFlowModel(type='graph',nodes={target.node_id:target},edges=[]))
    return rt,context,ModelAgentRunLog(coordination=context)


@pytest.mark.parametrize('tool_mode',[False,True])
async def test_real_fetch_step_and_callable_use_the_effective_request_port(tool_mode,monkeypatch):
    from magic_agents.node_system.NodeFetch import NodeFetch
    from magic_agents.models.factory.Nodes.FetchNodeModel import FetchNodeModel
    from magic_agents.node_system import fetch_request
    target=NodeFetch(FetchNodeModel(url='https://example.invalid/search',tool_mode=tool_mode,
        tool_name='search',method='GET'),node_id='fetch',node_type='fetch')
    target.inputs['handle_fetch_input']='topic'
    rt,context,log=node_scope(target);calls=[]
    async def send(request,**options):
        calls.append((request.url,options))
        assert (await rt.budget.snapshot())['activeJobs']==1
        return {'results':['bounded result']}
    monkeypatch.setattr(fetch_request,'send_admitted_request',send)
    events=[item async for item in target.process(log)]
    if tool_mode:
        function=next(item['content']['content'] for item in events if item['type']==target.OUTPUT_HANDLE)
        assert await function()=='{"results": ["bounded result"]}'
    else:assert 'bounded result' in repr(events)
    assert len(calls)==1
    assert (await rt.budget.snapshot())['spent']['tool_calls']==1
    assert next(iter(context._external_operations.values()))['kind']=='http.fetch'


async def test_fetch_protected_admission_cannot_become_an_ordinary_tool_error(monkeypatch):
    from magic_agents.node_system.NodeFetch import NodeFetch
    from magic_agents.models.factory.Nodes.FetchNodeModel import FetchNodeModel
    target=NodeFetch(FetchNodeModel(url='https://example.invalid',tool_mode=True,tool_name='search'),
                     node_id='fetch',node_type='fetch')
    rt,context,log=node_scope(target)
    def denied(*args):raise PermissionError('Resource permission revoked')
    rt.authorize_external=denied
    events=[item async for item in target.process(log)]
    function=next(item['content']['content'] for item in events if item['type']==target.OUTPUT_HANDLE)
    with pytest.raises(PermissionError):await function()
    assert not context._external_operations and (await rt.budget.snapshot())['spent']['tool_calls']==0


async def test_real_memory_joins_extraction_and_charges_both_embeddings(monkeypatch):
    from magic_llm.engine.engine_openai import EngineOpenAI
    from magic_llm.engine import engine_openai
    from magic_agents.node_system.NodeMemory import NodeMemory
    from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryNodeModel
    from test.test_coordination_graph import node
    import json
    calls=[]
    class Client:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):return None
        async def post_json(self,**kwargs):
            body=json.loads(kwargs['data']);calls.append(body['input'])
            assert kwargs['allow_redirects'] is False and kwargs['max_response_bytes']>0
            return {'model':body['model'],'data':[{'index':0,'embedding':[.25]*3072}]}
    monkeypatch.setattr(engine_openai,'AsyncHttpClient',Client)
    started,release=asyncio.Event(),asyncio.Event()
    async def extraction(provider,chat):
        started.set();await release.wait()
        return {'content':'{"memories":[{"content":"remembered","trigger":"topic"}]}'}
    llm,provider=node('extractor',extraction)
    embedder=EngineOpenAI(api_key="test", model="text-embedding-3-small")
    target=NodeMemory(MemoryNodeModel(instructions='Remember useful facts'),node_id='memory',node_type='memory',
        client=llm.inputs[llm.INPUT_HANDLER_CLIENT_PROVIDER],embedding_client=SimpleNamespace(llm=embedder),
        background_task_tracker=lambda coro:(_ for _ in ()).throw(AssertionError('No detached tracker')))
    target.inputs[target.INPUT_HANDLE]='new information'
    rt,context,log=node_scope(target)
    async def run():return [item async for item in target.process(log)]
    task=asyncio.create_task(run());await started.wait()
    assert not task.done() and len(calls)==1
    release.set();events=await task
    assert 'new information' in repr(events)
    assert len(calls)==2 and len(provider.calls)==1
    assert len(target._vector_db._entries)==1
    totals=await rt.budget.snapshot()
    assert totals['modelTurns']==1 and totals['spent']['tool_calls']==2
    assert {item['kind'] for item in context._external_operations.values()}=={'memory.embedding.query','memory.embedding.write'}


async def test_memory_cancellation_joins_embedding_and_does_not_start_extraction(monkeypatch):
    from magic_agents.node_system.NodeMemory import NodeMemory
    from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryNodeModel
    started,joined=asyncio.Event(),asyncio.Event()
    from magic_llm.engine.engine_openai import EngineOpenAI
    from magic_llm.engine import engine_openai
    class Client:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):return None
        async def post_json(self,**kwargs):
            started.set()
            try:await asyncio.Event().wait()
            finally:joined.set()
    monkeypatch.setattr(engine_openai,'AsyncHttpClient',Client)
    engine=EngineOpenAI(api_key="test",model="text-embedding-3-small")
    target=NodeMemory(MemoryNodeModel(instructions='Remember'),node_id='memory',node_type='memory',
                      embedding_client=SimpleNamespace(llm=engine))
    target.inputs[target.INPUT_HANDLE]='input'
    rt,context,log=node_scope(target)
    async def run():return [item async for item in target.process(log)]
    task=asyncio.create_task(run());await started.wait();task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert joined.is_set() and (await rt.budget.snapshot())['activeJobs']==0
    assert next(iter(context._external_operations.values()))['state']=='unknown'


@pytest.mark.parametrize('denied',[False,True])
async def test_conditional_real_provider_admission_cannot_default_around_denial(denied):
    from test.test_coordination_graph import node
    from test.test_conditional_llm import conditional
    async def answer(provider,chat):return {'content':'{"answers":{"ok":0.9}}'}
    llm,provider=node('judge',answer)
    target=conditional(questions={'ok':{'type':'noul','instructions':'Allowed?'}},
                       evaluation_error_policy='default',default_handle='no')
    target.inputs={'handle_input':'input','handle-client-provider':llm.inputs[llm.INPUT_HANDLER_CLIENT_PROVIDER]}
    rt,context,log=node_scope(target)
    if denied:
        def fail(*args):raise PermissionError('No judgment permission')
        rt.authorize_attempt=fail
        with pytest.raises(PermissionError):[item async for item in target.process(log)]
        assert not provider.calls and not target.used_default
    else:
        events=[item async for item in target.process(log)]
        assert target.selected_handle=='yes' and len(provider.calls)==1
        assert (await rt.budget.snapshot())['modelTurns']==1


async def test_mcp_unqualified_stdio_rejected_before_spawn_or_logical_charge():
    from magic_agents.mcp.session import MCPSessionManager
    from magic_agents.models.factory.Nodes.McpNodeModel import MCPServerConfig
    context=scope();control=DispatchSession(context,'mcp');calls=[]
    manager=MCPSessionManager(MCPServerConfig(transport='stdio',command='fake-server'),
                              node_id='mcp',dispatch_session=control)
    async def connect():calls.append('spawn')
    manager._connect_uncontrolled=connect
    with pytest.raises(CoordinationError, match='qualified stateless HTTP'):
        await manager.connect()
    assert not calls and (await context.budget.snapshot())['spent']['tool_calls']==0


@pytest.mark.parametrize('denied',[False,True])
async def test_conditional_jev_direct_positive_and_protected_default_guard(denied,monkeypatch):
    from test.test_conditional_jev import node, payload
    from magic_agents.node_system import fetch_request
    target=node(evaluation_error_policy='default',default_handle='no')
    target.inputs['handle_input']='admitted input'
    rt,context,log=node_scope(target);calls=[]
    monkeypatch.setenv('JEV_API_KEY','fake-key')
    async def send(request,**options):
        calls.append(request)
        return payload()
    monkeypatch.setattr(fetch_request,'send_admitted_request',send)
    if denied:
        def fail(*args):raise PermissionError('Jev resource revoked')
        rt.authorize_external=fail
        with pytest.raises(PermissionError):[item async for item in target.process(log)]
        assert not calls and not target.used_default
    else:
        events=[item async for item in target.process(log)]
        assert len(calls)==1 and not target.used_default,events
        assert (await rt.budget.snapshot())['spent']['tool_calls']==1
        assert next(iter(context._external_operations.values()))['kind']=='conditional.jev'


async def test_admitted_http_stops_body_at_bound_without_following_redirects(monkeypatch):
    from magic_agents.node_system import fetch_request
    reads=[];options=[]
    class Response:
        status,reason=200,'OK'
        headers={}
        @property
        def content(self):return self
        async def iter_chunked(self,size):
            reads.append(1);yield b'1234'
            reads.append(2);yield b'not reached'
        async def __aenter__(self):return self
        async def __aexit__(self,*args):return None
    class Session:
        _retry_connection = True
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):return None
        def request(self,**kwargs):options.append(kwargs);return Response()
    monkeypatch.setattr(fetch_request.aiohttp,'ClientSession',Session)
    with pytest.raises(CoordinationError) as failure:
        await fetch_request.send_admitted_request(fetch_request.FetchRequest('GET','https://example.invalid',{}),
                                                  timeout=1,max_response_bytes=3)
    assert failure.value.code=='external_response_limit' and reads==[1]
    assert options[0]['allow_redirects'] is False


@pytest.mark.parametrize('stream',[False,True])
async def test_actual_scheduler_native_fetch_single_job_slot_without_double_charge(stream,monkeypatch):
    from test.test_coordination_graph import node,runtime,collect
    from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
    from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
    from magic_agents.models.factory.Nodes.FetchNodeModel import FetchNodeModel
    from magic_agents.node_system.NodeFetch import NodeFetch
    from magic_agents.node_system import fetch_request
    seen=[]
    async def answer(provider,chat):
        if len(provider.calls)==1:return {'tool_calls':[{'id':'search-1','type':'function',
            'function':{'name':'search','arguments':'{}'}}]}
        assert 'source result' in repr(chat.messages)
        return {'content':'research complete'}
    llm,provider=node('research',answer,stream=stream)
    fetch=NodeFetch(FetchNodeModel(url='https://example.invalid/search',tool_mode=True,tool_name='search'),
                     node_id='fetch',node_type='fetch')
    rt=runtime(server_limits=limits(maxConcurrentJobs=1),authorize_external=lambda *args:None,
        external_estimate=lambda *args:UsageBound(tool_calls=1,cost=Decimal('.1')),
        external_usage=lambda *args:UsageBound(tool_calls=1,cost=Decimal('.01')))
    async def send(request,**options):
        seen.append(request.url)
        assert (await rt.budget.snapshot())['activeJobs']==1
        return {'text':'source result'}
    monkeypatch.setattr(fetch_request,'send_admitted_request',send)
    graph=AgentFlowModel(type='graph',nodes={'research':llm,'fetch':fetch},edges=[EdgeNodeModel(
        id='fetch-research',source='fetch',target='research',sourceHandle=fetch.OUTPUT_HANDLE,
        targetHandle=llm.INPUT_TOOL_PREFIX+'search')])
    result,events=await collect(graph,rt)
    assert not result['has_errors'],events
    assert len(seen)==1 and len(provider.calls)==2
    totals=await rt.budget.snapshot()
    assert totals['spent']['tool_calls']==1 and totals['activeJobs']==0


async def test_actual_hook_memory_child_single_slot_and_completed_identity_replay(monkeypatch):
    from test.test_coordination_lifecycle import parser,hook,connection,setup
    from test.test_coordination_graph import runtime
    from magic_agents.node_system.NodeMemory import NodeMemory
    from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryNodeModel
    from magic_llm.engine.engine_openai import EngineOpenAI
    from magic_llm.engine import engine_openai
    calls=[]
    class Client:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):return None
        async def post_json(self,**kwargs):
            calls.append(kwargs)
            assert (await rt.budget.snapshot())['activeJobs']==1
            return {'model':'text-embedding-3-small','data':[{'index':0,'embedding':[.5]}]}
    monkeypatch.setattr(engine_openai,'AsyncHttpClient',Client)
    memory=NodeMemory(MemoryNodeModel(),node_id='memory',node_type='memory',
        embedding_client=SimpleNamespace(llm=EngineOpenAI(api_key='test',model='text-embedding-3-small')))
    owner=parser('target')
    control_hook=hook('control', '''async def control(context, chat_log):
    child=await context.call('control-memory','topic')
    return {'action':'outcome','outcome':{'status':'success','content':{
        'handle_parser_output':child['outcome']['content']}}}
''')
    rt=runtime(server_limits=limits(maxConcurrentJobs=1),authorize_external=lambda *args:None,
        external_estimate=lambda *args:UsageBound(tool_calls=1,cost=Decimal('.1')),
        external_usage=lambda *args:UsageBound(tool_calls=1,cost=Decimal('.01')))
    rt,context,control,log,_=setup([owner,memory,control_hook],
        [connection('control','memory',memory.INPUT_HANDLE)],rt=rt)
    result=await control.invoke('target',{'value':'original'},log)
    assert result['outcome']['status']=='success' and len(calls)==1
    entry=next(iter(context.child_operations.values()))
    async def forbidden():raise AssertionError('Completed child was repeated')
    assert await context.run_child_operation('memory',entry['operation_id'],'topic',forbidden)==entry['result']
    assert (await rt.budget.snapshot())['spent']['tool_calls']==1


async def test_actual_transport_snapshot_stays_detached_and_rejects_expansion_before_admission(monkeypatch):
    from magic_agents.node_system import fetch_request
    calls=[];context=scope()
    context.dispatch_session=lambda node_id:DispatchSession(context,node_id)
    request=fetch_request.FetchRequest('POST','https://example.invalid',{},json_body={'message':'original'})
    def mutate(*args):request.json_body['message']='mutated'
    context.runtime.authorize_external=mutate
    async def send(request,**options):calls.append(options['prepared']);return {'ok':True}
    monkeypatch.setattr(fetch_request,'send_admitted_request',send)
    await fetch_request.coordinated_fetch(context,'fetch',request)
    import json
    assert json.loads(calls[0]['data'])=={'message':'original'}
    large=fetch_request.FetchRequest('POST','https://example.invalid',{},json_body={'text':'é'*800000})
    with pytest.raises(CoordinationError) as failure:
        await fetch_request.coordinated_fetch(context,'fetch',large)
    assert failure.value.code=='external_request_limit' and len(calls)==1


async def test_queued_credit_cancellation_does_not_dispatch_or_restore_new_allowance():
    context=scope(maxConcurrentJobs=1);session=DispatchSession(context,'fetch');seen=[]
    await context.budget.reserve('held',UsageBound(tool_calls=1),kind='job')
    async def effect():seen.append(True)
    async def run():
        async with invocation_charge(context,'outer',context.capture_guard('fetch')):
            await session.call('http.fetch',{},effect)
    task=asyncio.create_task(run());await asyncio.sleep(0);task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert not seen and (await context.budget.snapshot())['spent']['tool_calls']==1
    await context.budget.settle('held',UsageBound(tool_calls=1))
    assert (await context.budget.snapshot())['activeJobs']==0


async def test_loaded_node_capability_is_class_owned():
    from magic_agents.node_system.NodeFetch import NodeFetch
    from magic_agents.node_system.NodeMemory import NodeMemory
    from magic_agents.node_system.NodeMcp import NodeMcp
    from magic_agents.node_system.NodeConditional import NodeConditional
    for cls in (NodeFetch,NodeMemory,NodeMcp,NodeConditional):
        assert cls.__dict__['coordination_external_dispatch_version']==1


async def test_real_embedding_http_cancellation_joins_reader_and_keeps_exposure(monkeypatch):
    from magic_llm.engine.engine_openai import EngineOpenAI
    from magic_llm.util import http
    started,closed=asyncio.Event(),asyncio.Event();calls=[]
    class Wire:
        _retry_connection = True
        status=200;headers={}
        def request(self,*args,**kwargs):calls.append(kwargs);return self
        async def close(self):closed.set()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):return None
        @property
        def content(self):return self
        async def iter_chunked(self,size):
            started.set()
            await asyncio.Event().wait()
            yield b''
    wire=Wire();monkeypatch.setattr(http.aiohttp,'ClientSession',lambda:wire)
    context=scope();session=DispatchSession(context,'memory')
    engine=EngineOpenAI(api_key='test',model='text-embedding-3-small')
    task=asyncio.create_task(engine.async_embedding('input',external_dispatch=session))
    await started.wait();task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    totals=await context.budget.snapshot()
    assert closed.is_set() and len(calls)==1 and totals['activeJobs']==0
    assert totals['uncertain']['cost']!='0' and next(iter(context._external_operations.values()))['state']=='unknown'


async def test_unqualified_memory_embedding_adapter_is_rejected_before_work():
    from magic_agents.node_system.NodeMemory import NodeMemory
    from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryNodeModel
    calls=[]
    async def raw(text):calls.append(text)
    target=NodeMemory(MemoryNodeModel(),node_id='memory',node_type='memory',
        embedding_client=SimpleNamespace(llm=SimpleNamespace(async_embedding=raw)))
    target.inputs[target.INPUT_HANDLE]='input'
    _,_,log=node_scope(target)
    with pytest.raises(CoordinationError) as failure:[item async for item in target.process(log)]
    assert failure.value.code=='adapter_unsupported' and not calls
