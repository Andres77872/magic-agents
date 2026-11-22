"""Graph-to-agent contracts exercised with real NodeLLM and MagicLLM loops."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.mcp.errors import ToolNameCollisionError
from magic_agents.mcp.bundle import MCPToolBundle
from magic_agents.node_system.utils import apply_windowing
from magic_llm import MagicLLM
from magic_llm.agent.types import AgentBudgetExceeded
from magic_llm.model.ModelChatResponse import Choice, Message, ModelChatResponse, UsageModel
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel


def schema(name):
    return {'type':'function','function':{'name':name,'description':'Double the supplied value',
        'parameters':{'type':'object','properties':{'value':{'type':'integer'}},'required':['value']}}}


class Provider:
    def __init__(self, name='double', function=None):
        self.tool_schema = schema(name)
        self.tool_callable = function or self.execute
        self.calls = []
        self.events = []

    async def execute(self, value):
        self.calls.append(value)
        return value * 2

    def drain_events(self):
        result, self.events = self.events, []
        return result


def node(inputs=None, **data):
    result = NodeLLM(data=LlmNodeModel(**data), node_id='reasoner', node_type='llm', debug=False)
    result.inputs = inputs or {}
    return result


@pytest.mark.asyncio
async def test_collect_many_flattens_providers_bundles_and_deduplicates_repeated_provider():
    first, second = Provider('first'), Provider('second')
    async def third(value): return value
    bundle = MCPToolBundle(tool_schemas=[schema('third')], tool_functions={'third':third}, node_id='mcp')
    schemas, functions, _ = await node({'handle-tool-0':[first,[second,bundle],first]})._collect_tools()
    assert [s['function']['name'] for s in schemas] == ['first','second','third']
    assert list(functions) == ['first','second','third']
    assert await functions['first'](value=3) == 6


@pytest.mark.asyncio
async def test_schema_name_binds_a_bound_method_with_different_implementation_name():
    provider = Provider('public_double')
    _, functions, _ = await node({'handle-tool-0':provider})._collect_tools()
    assert list(functions) == ['public_double']
    assert await functions['public_double'](value=4) == 8


@pytest.mark.asyncio
async def test_different_plain_callables_with_same_name_fail_before_execution():
    async def first(value): return 1
    async def second(value): return 2
    first.__name__ = second.__name__ = 'collision'
    with pytest.raises(ToolNameCollisionError):
        await node({'handle-tool-0':first,'handle-tool-1':second})._collect_tools()


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [42,'not a tool',{'value':1}])
async def test_unsupported_tool_payload_is_not_silently_ignored(value):
    with pytest.raises(ValueError, match='tool'):
        await node({'handle-tool-0':value})._collect_tools()


@pytest.mark.asyncio
async def test_provider_without_executor_remains_schema_only_and_cannot_mix():
    schema_only = Provider('client_only')
    schema_only.tool_callable = None
    with pytest.raises(ValueError, match='mixed callable and schema-only'):
        await node({'handle-tool-0':[schema_only,Provider('server')]})._collect_tools()


@pytest.mark.parametrize('config', [
    {'max_iterations':0}, {'max_iterations':True}, {'max_iterations':1.5},
    {'wall_clock_timeout':0}, {'per_tool_timeout':-1}, {'per_tool_timeout':float('inf')},
    {'max_parallel_tools':0}, {'max_output_chars':0}, {'deduplicate':'false'}, {'typo':3},
])
def test_agent_config_rejects_invalid_limits(config):
    with pytest.raises(ValidationError):
        LlmNodeModel(agent_config=config)


def response(content=None, calls=None):
    return ModelChatResponse(id='request',object='chat.completion',created=0,model='scripted',
        choices=[Choice(index=0,message=Message(role='assistant',content=content,tool_calls=calls),
                        finish_reason='tool_calls' if calls else 'stop')],
        usage=UsageModel(prompt_tokens=2,completion_tokens=1,total_tokens=3))


def call(identifier, value):
    return {'id':identifier,'type':'function','function':{'name':'double','arguments':json.dumps({'value':value})}}


class Engine:
    model = 'scripted'
    engine_name = 'openai'

    def __init__(self, steps):
        self.steps = iter(steps)
        self.requests = []

    async def async_generate(self, chat, **kwargs):
        self.requests.append((copy.deepcopy(chat.messages),kwargs))
        result = next(self.steps)
        if isinstance(result, BaseException): raise result
        return result

    async def async_stream_generate(self, chat, **kwargs):
        result = await self.async_generate(chat, **kwargs)
        message = result.choices[0].message
        calls = [c.model_dump() for c in (message.tool_calls or [])]
        for index, item in enumerate(calls): item['index'] = index
        yield ChatCompletionModel(id=result.id,model=result.model,choices=[ChoiceModel(index=0,
            delta=DeltaModel(content=message.content or '',tool_calls=calls or None),
            finish_reason=result.choices[0].finish_reason)],usage=result.usage)


async def prepared_node(monkeypatch, engine, provider, **data):
    client = MagicLLM.__new__(MagicLLM)
    client.llm = engine
    client._task_executor = None
    result = node({'handle-client-provider':client,'handle_user_message':'compute',
                   'handle-tool-0':provider}, **data)
    async def no_subagents(_): return SimpleNamespace(tool_schemas=[],registered_count=0)
    monkeypatch.setattr(result,'_load_subagents_if_enabled',no_subagents)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
async def test_real_loop_multi_round_tools_keep_results_ids_and_config_out_of_provider(monkeypatch,stream):
    provider = Provider()
    engine = Engine([response(calls=[call('first',3),call('parallel',4)]),
                     response(calls=[call('second',7)]),response('14')])
    result = await prepared_node(monkeypatch,engine,provider,stream=stream,
        agent_config={'max_iterations':3,'max_parallel_tools':1,'per_tool_timeout':1,'max_output_chars':100})
    events = [e async for e in result.process(None)]
    assert result.generated == '14'
    assert provider.calls == [3,4,7]
    second_history = engine.requests[1][0]
    assert [(m['tool_call_id'],json.loads(m['content'])) for m in second_history if m['role']=='tool'] == [('first',6),('parallel',8)]
    assert [m['tool_call_id'] for m in engine.requests[2][0] if m['role']=='tool'] == ['first','parallel','second']
    assert all(not ({'agent_config','tool_executor_options','per_tool_timeout','max_parallel_tools'} & set(kwargs)) for _,kwargs in engine.requests)
    assert any(e.get('type')=='debug' and e.get('content',{}).get('event_type')=='TOOL_RESULT' for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
async def test_config_enforces_iteration_budget_before_next_provider_request(monkeypatch,stream):
    provider=Provider()
    engine=Engine([response(calls=[call('first',3)]),response('should not run')])
    result=await prepared_node(monkeypatch,engine,provider,stream=stream,agent_config={'max_iterations':1})
    events=[]
    with pytest.raises(AgentBudgetExceeded):
        async for event in result.process(None): events.append(event)
    assert len(engine.requests)==1
    assert provider.calls==[3]
    tool_events=[event['content'] for event in events if event.get('type')=='debug'
                 and event.get('content',{}).get('event_type') in {'TOOL_CALL','TOOL_RESULT'}]
    assert [event['event_type'] for event in tool_events]==['TOOL_CALL','TOOL_RESULT']
    assert tool_events[0]['data']['tool_calls'][0]['id']=='first'
    assert tool_events[1]['data']['tool_call_id']=='first'
    assert tool_events[1]['data']['status']=='success'
    assert json.loads(tool_events[1]['data']['content'])==6


@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
async def test_continuation_without_new_user_preserves_complete_tool_history(monkeypatch,stream):
    history=[{'role':'user','content':'compute'},
             {'role':'assistant','content':None,'tool_calls':[call('existing',5)]},
             {'role':'tool','tool_call_id':'existing','name':'double','content':'10'}]
    engine=Engine([response('10')])
    result=await prepared_node(monkeypatch,engine,Provider(),stream=stream,history_messages=history)
    result.inputs['handle_user_message']=''
    async for _ in result.process(None): pass
    seen=[m for m in engine.requests[0][0] if m['role']!='system']
    assert [m['role'] for m in seen]==['user','assistant','tool']
    assert seen[1]['tool_calls'][0]['id']=='existing'
    assert seen[2]['tool_call_id']=='existing'
    assert seen[2]['name']=='double'


@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('fails',[False,True])
async def test_nested_tool_usage_drains_once_on_success_or_provider_failure(monkeypatch,stream,fails):
    class MeteredProvider(Provider):
        @property
        def __name__(self): return 'double'
        async def __call__(self,value):
            from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL
            active=CURRENT_TOOL_CALL.get()
            self.events.append({'type':'tool_usage','data':{'id':'child','model':'nested',
                'provider':'fake','usage':{'prompt_tokens':5,'completion_tokens':2,'total_tokens':7}},
                'source_node':'child-llm','source_node_path':['inner','child-llm'],'tool_call_id':active.id})
            return await self.execute(value)
    provider=MeteredProvider()
    provider.tool_callable=provider
    engine=Engine([response(calls=[call('first',3)]), RuntimeError('provider failed') if fails else response('6')])
    result=await prepared_node(monkeypatch,engine,provider,stream=stream)
    events=[]
    try:
        async for event in result.process(None): events.append(event)
    except RuntimeError:
        assert fails
    else:
        assert not fails
    usage=[e['content']['data'] for e in events if e.get('content',{}).get('event_type')=='TOOL_USAGE']
    tool_events=[event['content'] for event in events if event.get('type')=='debug'
                 and event.get('content',{}).get('event_type') in {'TOOL_CALL','TOOL_RESULT'}]
    assert [event['event_type'] for event in tool_events]==['TOOL_CALL','TOOL_RESULT']
    assert len(usage)==1
    assert usage[0]['tool_call_id']=='first'
    assert usage[0]['data']['usage']['total_tokens']==7
    assert provider.drain_events()==[]


@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('kind',['timeout','truncate','dedup','parallel'])
async def test_real_node_configuration_changes_executor_behavior(monkeypatch,stream,kind):
    state={'active':0,'peak':0,'cancelled':False}
    class ControlledProvider(Provider):
        async def execute(self,value):
            self.calls.append(value)
            if kind=='timeout':
                try: await asyncio.sleep(10)
                finally: state['cancelled']=True
            if kind=='truncate': return 'x'*200
            state['active']+=1
            state['peak']=max(state['peak'],state['active'])
            await asyncio.sleep(.003)
            state['active']-=1
            return value*2
    provider=ControlledProvider()
    tool_calls=[call('first',3),call('second',3)]
    engine=Engine([response(calls=tool_calls),response('done')])
    # Only the timeout case should hit the per-tool limit; the others get room for a loaded CI.
    config={'per_tool_timeout':.05 if kind=='timeout' else 2,'max_output_chars':30,'deduplicate':kind=='dedup','max_parallel_tools':1}
    result=await prepared_node(monkeypatch,engine,provider,stream=stream,agent_config=config)
    async for _ in result.process(None): pass
    outputs=[m for m in engine.requests[1][0] if m['role']=='tool']
    assert [m['tool_call_id'] for m in outputs]==['first','second']
    if kind=='timeout':
        assert state['cancelled']
        assert all(json.loads(m['content'])['type']=='TimeoutError' for m in outputs)
    elif kind=='truncate':
        assert all(m['content'].endswith('[TRUNCATED]') for m in outputs)
        assert all(len(m['content'])==30+len('[TRUNCATED]') for m in outputs)
    elif kind=='dedup':
        assert provider.calls==[3]
        assert all(json.loads(m['content'])==6 for m in outputs)
    else:
        assert state['peak']==1
        assert provider.calls==[3,3]


@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
async def test_real_node_wall_deadline_cancels_provider(monkeypatch,stream):
    cancelled=asyncio.Event()
    class StalledEngine(Engine):
        async def async_generate(self,chat,**kwargs):
            try: await asyncio.Event().wait()
            finally: cancelled.set()
    engine=StalledEngine([])
    result=await prepared_node(monkeypatch,engine,Provider(),stream=stream,agent_config={'wall_clock_timeout':.2})
    with pytest.raises(AgentBudgetExceeded,match='wall_clock_timeout'):
        async for _ in result.process(None): pass
    assert cancelled.is_set()


def test_history_window_keeps_retained_chronology_without_mutating_input():
    history=[{'role':'user','content':'old'},{'role':'assistant','content':'old answer'},
             {'role':'user','content':'current'},
             {'role':'assistant','content':None,'tool_calls':[call('existing',5)]},
             {'role':'tool','tool_call_id':'existing','content':'10'}]
    original=copy.deepcopy(history)
    kept=apply_windowing(history,max_messages=3)
    assert [m['role'] for m in kept]==['user','assistant','tool']
    assert kept[0]['content']=='current'
    assert kept[1]['tool_calls'][0]['id']==kept[2]['tool_call_id']=='existing'
    assert history==original


@pytest.mark.asyncio
@pytest.mark.parametrize("invoke", ["process", "call"])
async def test_closing_node_stream_closes_provider_and_resets_loop_context(monkeypatch, invoke):
    from magic_llm.agent._loop_shared import PARENT_STATE
    closed=asyncio.Event()
    opened=asyncio.Event()
    class StreamingEngine(Engine):
        async def async_stream_generate(self,chat,**kwargs):
            try:
                opened.set()
                yield ChatCompletionModel(id='partial',model='fake',choices=[ChoiceModel(index=0,delta=DeltaModel(content='partial'))])
                await asyncio.Event().wait()
            finally:
                closed.set()
    node_instance=await prepared_node(monkeypatch,StreamingEngine([]),Provider(),stream=True)
    original=PARENT_STATE.get()
    stream=node_instance.process(None) if invoke == "process" else node_instance(None)
    while not opened.is_set():
        await anext(stream)
    await stream.aclose()
    assert closed.is_set()
    assert PARENT_STATE.get() is original


@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
async def test_real_graph_tool_parallel_usage_keeps_call_identity_and_child_text_private(monkeypatch,stream):
    from magic_agents.agt_flow import build
    from magic_llm.agent.tool_executor import CURRENT_TOOL_CALL
    child={'nodes':[
        {'id':'child-input','type':'user_input','data':{}},
        {'id':'calculate','type':'parser','data':{'template':'{{ request.value * 2 }}'}},
        {'id':'child-end','type':'end','data':{}}],
        'edges':[
        {'source':'child-input','target':'calculate','sourceHandle':'handle_user_message','targetHandle':'request'},
        {'source':'calculate','target':'child-end','sourceHandle':'handle_parser_output','targetHandle':'handle_flow_input'}]}
    graph=build({'nodes':[
        {'id':'graph-tool','type':'inner','data':{'magic_flow':child,'tool_mode':True,'tool_name':'double',
            'tool_parameters':schema('double')['function']['parameters']}},
        {'id':'input','type':'user_input','data':{}},{'id':'end','type':'end','data':{}}],
        'edges':[{'source':'input','target':'end','sourceHandle':'handle_user_message','targetHandle':'handle_flow_input'}]},message='parent')
    tool_node=graph.nodes['graph-tool']
    tool=[event async for event in tool_node.process(None) if event['type']=='handle-tool-definition'][0]['content']['content']
    factory=tool_node._tool_graph_factory
    def metered_factory(arguments,extras):
        child_graph=factory(arguments,extras)
        parser=child_graph.nodes['calculate']
        async def metered_process(log):
            # Deliberately interleave child tasks to expose shared call identity.
            await asyncio.sleep(.004 if arguments['value']==3 else .001)
            yield parser.yield_static(ChatCompletionModel(id=f"child-{arguments['value']}",model='fake',
                choices=[ChoiceModel(delta=DeltaModel(content='private child text'))],
                usage={'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}),content_type='content')
            if arguments['value']<0:raise ValueError('negative rejected')
            yield parser.yield_static(str(arguments['value']*2),content_type='handle_parser_output')
        parser.process=metered_process
        return child_graph
    tool_node._tool_graph_factory=metered_factory
    engine=Engine([response(calls=[call('first',3),call('second',-1)]),response('parent final')])
    result=await prepared_node(monkeypatch,engine,tool,stream=stream)
    baseline=CURRENT_TOOL_CALL.get()
    events=[event async for event in result.process(None)]
    usage=[e['content']['data'] for e in events if e.get('content',{}).get('event_type')=='TOOL_USAGE']
    assert {(e['tool_call_id'],e['data']['id']) for e in usage}=={('first','child-3'),('second','child--1')}
    assert all(e['source_node_path']==['graph-tool','calculate'] for e in usage)
    assert CURRENT_TOOL_CALL.get() is baseline
    assert tool.drain_events()==[]
    assert result.generated=='parent final'
    assert 'private child text' not in str(events)
    tool_messages=[message for message in engine.requests[1][0] if message['role']=='tool']
    assert json.loads(tool_messages[0]['content'])=='6'
    assert 'negative rejected' in tool_messages[1]['content']
