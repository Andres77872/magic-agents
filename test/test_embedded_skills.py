"""Prompt definitions stay private until an explicit model-selected load."""
import asyncio
import copy
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from magic_llm import MagicLLM
from magic_llm.agent.request import AgentRequestContext
from magic_llm.agent.tool_executor import ToolExecutor
from magic_llm.agent.types import CanonicalToolCall, ToolResult, TaskManifest
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatResponse import ModelChatResponse
from magic_agents.agt_flow import build, create_node
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from magic_agents.hooks.hook_relay import HookRelay
from magic_agents.hooks.invocation_control import OperationFailure, error_code
from magic_agents.models.factory.Nodes import LlmNodeModel, SkillsNodeModel, ChatNodeModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.NodeChat import NodeChat
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.node_system.NodeSkills import NodeSkills
from magic_agents.skills import (SkillPromptBundle, create_skills_loader, create_request_guard,
    safe_loader_result, strip_ephemeral_skills_history, SKILLS_PROVENANCE, SKILLS_TOOL_SCHEMA)

PRIVATE = 'PRIVATE_SKILL_BODY'
PROPS_PRIVATE = 'PRIVATE_SKILL_PROPERTY'
LOG = ModelAgentRunLog(run_id='test')


def entry(identifier='release-notes', **updates):
    return {'id': identifier, 'name': identifier, 'description': 'Write clear release notes',
            'prompt': PRIVATE, 'props': {'nested': [{'value': PROPS_PRIVATE}]}, **updates}


def model(*entries):
    return SkillsNodeModel(schema_version=1, skills=list(entries) or [entry()])


def bundle(*entries):
    return SkillPromptBundle.from_model('skills', model(*entries))


@pytest.mark.parametrize('change', [
    {'schema_version': True}, {'schema_version': 2}, {'skills': []}, {'unexpected': 'x'},
    {'skills': [entry(id='Bad_ID')]}, {'skills': [entry(enabled='false')]},
    {'skills': [entry(prompt='   ', enabled=False)]}, {'skills': [entry(description='')]},
    {'skills': [entry(props=[]) ]}, {'skills': [entry(props={'n': float('nan')})]},
    {'skills': [entry(props={1: 'value'})]}, {'skills': [entry(props={'s': 'é' * 2048})]},
    {'skills': [entry(props={'a': {'b': {'c': {'d': {'e': {}}}}}})]},
    {'skills': [entry(), entry()]}, {'skills': [entry(str(i)) for i in range(17)]},
    {'skills': [entry(str(i), prompt='x'*12000) for i in range(3)]},
    {'skills': [entry(str(i), props={'x': 'a'*3500}, enabled=False) for i in range(5)]},
])
def test_strict_complete_definition_contract(change):
    raw = {'schema_version': 1, 'skills': [entry()], **change}
    with pytest.raises(ValidationError):
        SkillsNodeModel(**raw)


def test_bundle_deep_immutability_and_fresh_materialization():
    original = model()
    private = SkillPromptBundle.from_model('skills', original)
    original.skills[0].props['nested'][0]['value'] = 'changed'
    assert private.skills[0].definition()['props']['nested'][0]['value'] == PROPS_PRIVATE
    with pytest.raises(FrozenInstanceError):
        private.skills[0].prompt = 'changed'
    with pytest.raises(FrozenInstanceError):
        private.skills[0].props.entries = ()
    first = private.skills[0].definition()
    first['props']['nested'][0]['value'] = 'changed again'
    assert private.skills[0].definition()['props']['nested'][0]['value'] == PROPS_PRIVATE
    assert PRIVATE not in repr(private)
    assert PRIVATE not in private.catalog() and PROPS_PRIVATE not in private.catalog()


def test_safe_factory_validation_errors_and_debug(caplog):
    caplog.set_level('DEBUG')
    with pytest.raises(ValueError, match='SKILL_PROMPT_INVALID') as exc:
        create_node({'id':'skills','type':'skills','data':{'schema_version':1,'skills':[entry(prompt=PRIVATE, enabled='wrong')]}}, load_chat=None)
    assert PRIVATE not in str(exc.value) and PROPS_PRIVATE not in caplog.text
    assert PRIVATE not in caplog.text


@pytest.mark.asyncio
async def test_loader_exact_requested_order_and_no_caller_mutation():
    private = bundle(entry('first'), entry('second'), entry('disabled', enabled=False))
    executor = ToolExecutor(max_content_size=50000)
    executor.register('skills_load', create_skills_loader(private, executor.content_limit('skills_load')))
    result = await executor.execute_async(CanonicalToolCall(id='call',name='skills_load',arguments={'skill_ids':['second','first']}))
    expected = {'schema_version':1,'source_node_id':'skills','skills':[private.skills[1].definition(),private.skills[0].definition()]}
    assert not result.is_error and result.content == ToolExecutor.serialize_output(expected)
    assert json.loads(result.content)['skills'][0]['props']['nested'][0]['value'] == PROPS_PRIVATE


@pytest.mark.asyncio
@pytest.mark.parametrize('arguments', [
    {}, {'skill_ids':[]}, {'skill_ids':'first'}, {'skill_ids':[1]}, {'skill_ids':['first','first']},
    {'skill_ids':['first'],'extra': PRIVATE}, {'skill_ids':['first','missing']}, {'skill_ids':['disabled']},
])
async def test_invalid_load_is_atomic_typed_safe_error(arguments):
    executor = ToolExecutor()
    executor.register('skills_load', create_skills_loader(bundle(entry('first'),entry('disabled',enabled=False)),50000))
    result = await executor.execute_async(CanonicalToolCall(id='call', name='skills_load', arguments=arguments))
    assert result.is_error and result.error_type.startswith('SkillsLoad')
    assert PRIVATE not in result.content and PROPS_PRIVATE not in result.content
    assert 'skills' not in json.loads(result.content)


@pytest.mark.asyncio
async def test_exact_output_limit_errors_allow_smaller_retry_without_truncation():
    private = bundle(entry('one',prompt='é'*600), entry('two',prompt='é'*600))
    exact_single = ToolExecutor.serialize_output({'schema_version':1,'source_node_id':'skills','skills':[private.skills[0].definition()]})
    executor=ToolExecutor(max_content_size=len(exact_single))
    executor.register('skills_load',create_skills_loader(private,executor.content_limit('skills_load')))
    large = await executor.execute_async(CanonicalToolCall(id='a',name='skills_load',arguments={'skill_ids':['one','two']}))
    assert large.is_error and 'SKILLS_LOAD_RESULT_LIMIT' in large.content
    assert 'TRUNCATED' not in large.content and 'é' not in large.content
    small = await executor.execute_async(CanonicalToolCall(id='b',name='skills_load',arguments={'skill_ids':['one']}))
    assert not small.is_error and small.content == exact_single
    with pytest.raises(ValueError,match='SKILLS_CONTRACT_UNSUPPORTED'):
        create_skills_loader(private,20)


def graph(aliases=False, fanout=False):
    output, input_ = ('private-skills-out','private-skills-in') if aliases else ('handle-skills','handle-skills')
    nodes=[{'id':'skills','type':'skills','data':{'schema_version':1,'skills':[entry()], 'handles':{'output':output}}},
        {'id':'client','type':'client','data':{'engine':'openai','model':'fake','api_info':{'private_key':'dummy'}}},
        {'id':'input','type':'user_input','data':{}},
        {'id':'writer','type':'llm','data':{'handles':{'skills':input_}}}]
    edges=[{'id':'s','source':'skills','target':'writer','sourceHandle':output,'targetHandle':input_},
        {'id':'c','source':'client','target':'writer','sourceHandle':'handle-client-provider','targetHandle':'handle-client-provider'},
        {'id':'u','source':'input','target':'writer','sourceHandle':'handle_user_message','targetHandle':'handle_user_message'}]
    if fanout:
        nodes.append({'id':'other','type':'llm','data':{}})
        edges.append({'id':'s2','source':'skills','target':'other','sourceHandle':output,'targetHandle':'handle-skills'})
    return {'nodes':nodes,'edges':edges}


@pytest.mark.asyncio
@pytest.mark.parametrize('aliases',[False,True])
async def test_factory_alias_fanout_and_opaque_route(aliases):
    built=build(graph(aliases,True),message='write')
    source=built.nodes['skills']
    assert isinstance(source,NodeSkills)
    output=next(item for item in [item async for item in source.process(LOG)] if item['type']==source.OUTPUT_HANDLE)
    dispatcher=GraphEventDispatcher(built.nodes,built.edges)
    await dispatcher.propagate_outputs('skills',{output['type']:output['content']})
    first=built.nodes['writer']; second=built.nodes['other']
    assert first.inputs[first.INPUT_HANDLER_SKILLS] is second.inputs[second.INPUT_HANDLER_SKILLS]
    assert isinstance(first.inputs[first.INPUT_HANDLER_SKILLS],SkillPromptBundle)
    assert PRIVATE not in str(source._capture_internal_state())
    assert PRIVATE not in str(first._safe_copy_dict(first.inputs))
    fresh=first._invocation_factory()
    assert fresh._skills_source_node_id == 'skills'


@pytest.mark.parametrize('change',[ 'wrong_target','wrong_source','wrong_handle','incoming','duplicate','alias_collision'])
def test_topology_rules_always_block(change):
    raw=graph()
    if change=='wrong_target': raw['edges'][0]['target']='input'
    if change=='wrong_source': raw['edges'][0]['source']='input'
    if change=='wrong_handle': raw['edges'][0]['sourceHandle']='handle-other'
    if change=='incoming': raw['edges'].append({'id':'bad','source':'input','target':'skills','sourceHandle':'handle_user_message','targetHandle':'handle-skills'})
    if change=='duplicate': raw['edges'].append({**raw['edges'][0], 'id':'duplicate'})
    if change=='alias_collision': raw['nodes'][-1]['data']={'handles':{'skills':'handle_user_message'}}; raw['edges'][0]['targetHandle']='handle_user_message'
    with pytest.raises(ValueError,match='SKILLS_CONNECTION_INVALID'):
        build(raw,message='write')


def response(calls=None):
    return ModelChatResponse(id='response',object='chat.completion',created=0,model='fake',choices=[{'index':0,
        'message':{'role':'assistant','content':None if calls else 'done','tool_calls':calls},
        'finish_reason':'tool_calls' if calls else 'stop'}])


def selected(ids,call_id='load'):
    return {'id':call_id,'type':'function','function':{'name':'skills_load','arguments':json.dumps({'skill_ids':ids})}}


def client_with_wire_responses(calls_by_request):
    client=MagicLLM(engine='openai',model='fake',private_key='dummy')
    wire=[]
    async def generate(chat,**kwargs):
        encoded,_=client.llm.base.prepare_data(chat,**kwargs)
        wire.append(json.loads(encoded))
        return response(calls_by_request[len(wire)-1] if len(wire)<=len(calls_by_request) else None)
    client.llm.async_generate=generate
    return client,wire


def llm_node(client,private=None,**model_options):
    node=NodeLLM(LlmNodeModel(stream=False,json_output=False,**model_options),node_id='writer')
    node.inputs.update({node.INPUT_HANDLER_CLIENT_PROVIDER:client,node.INPUT_HANDLER_USER_MESSAGE:'Write release'})
    if private is not None: node.inputs[node.INPUT_HANDLER_SKILLS]=private
    return node


@pytest.mark.asyncio
async def test_model_selects_batches_repeats_and_safe_observer_never_replaces_canonical():
    client,wire=client_with_wire_responses([[selected(['second','first'],'load-1')],[selected(['first'],'load-2')],None])
    node=llm_node(client,bundle(entry('first'),entry('second')))
    chat=ModelChat('base one');chat.add_system_message('base two');chat.add_user_message('earlier question')
    before=copy.deepcopy(chat.messages)
    node.inputs[node.INPUT_HANDLER_CHAT]=chat
    events=[item async for item in node.process(LOG)]
    assert len(wire)==3 and chat.messages==before
    assert PRIVATE not in json.dumps(wire[0]) and PROPS_PRIVATE not in json.dumps(wire[0])
    assert PRIVATE in json.dumps(wire[1]) and PROPS_PRIVATE in json.dumps(wire[1])
    first_results=[m for m in wire[1]['messages'] if m['role']=='tool']
    assert [s['id'] for s in json.loads(first_results[0]['content'])['skills']]==['second','first']
    assert len([m for m in wire[2]['messages'] if m['role']=='tool'])==2
    for request in wire:
        systems=[m for m in request['messages'] if m['role']=='system']
        assert len(systems)==1 and systems[0]['content'].count('Available skills (metadata only)')==1
        assert 'base one' in systems[0]['content'] and 'base two' in systems[0]['content']
        assert PRIVATE not in systems[0]['content']
    assert PRIVATE not in str(events) and PROPS_PRIVATE not in str(events)
    tool_events=[e['content']['data'] for e in events if e['type']=='debug' and e['content'].get('event_type')=='TOOL_RESULT']
    assert [json.loads(e['content'])['loaded_ids'] for e in tool_events]==[['second','first'],['first']]
    assert all(e['source']=='builtin_skills' and e['ephemeral'] is True for e in tool_events)
    assert node._capture_internal_state()['skills']['loaded_ids']==['second','first']
    assert node.generated=='done'


@pytest.mark.asyncio
async def test_metadata_discovery_does_not_require_model_loading():
    client,wire=client_with_wire_responses([None])
    node=llm_node(client,bundle())
    events=[item async for item in node.process(LOG)]
    assert len(wire)==1 and PRIVATE not in json.dumps(wire)
    assert not any(e['type']=='debug' and e['content'].get('event_type')=='TOOL_RESULT' for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize('disabled',[False,True])
async def test_no_source_and_all_disabled_keep_direct_baseline(disabled):
    client,wire=client_with_wire_responses([None])
    node=llm_node(client,bundle(entry(enabled=False)) if disabled else None)
    client.run_agent_async=AsyncMock(side_effect=AssertionError('empty catalog must not create loop'))
    [item async for item in node.process(LOG)]
    assert len(wire)==1 and 'skills_load' not in json.dumps(wire) and PRIVATE not in json.dumps(wire)
    assert node._skills_request_guard is None


@pytest.mark.asyncio
@pytest.mark.parametrize('case',['missing','schema_only','collision','choice_none','inherited_choice','unsupported_provider','registered_collision'])
async def test_active_catalog_preflight_fails_before_provider(case):
    client,wire=client_with_wire_responses([None])
    node=llm_node(client,bundle())
    schemas=[]; functions={}
    node.extra_data={}
    if case=='missing':
        node._skills_source_node_id='skills';node.inputs.pop(node.INPUT_HANDLER_SKILLS)
        with pytest.raises(OperationFailure,match='Connected Skills'):
            [item async for item in node.process(LOG)]
        assert not wire
        return
    node._prepare_skills()
    if case=='schema_only': schemas=[{'type':'function','function':{'name':'client_tool','parameters':{'type':'object','properties':{}}}}]
    if case=='collision': functions={'skills_load':lambda:None}
    if case=='choice_none': node.extra_data={'tool_choice':'none'}
    if case=='inherited_choice': client.llm.kwargs['tool_choice']='none'
    if case=='unsupported_provider': client.llm.engine='amazon'
    if case=='registered_collision':
        client.register_task(TaskManifest(id='skills_load',name='old',description='old',input_schema={'type':'object','properties':{}}),lambda:None)
    with pytest.raises(OperationFailure):
        node._install_skills(client,ModelChat('base'),schemas,functions,SimpleNamespace(tool_schemas=[]))
    assert not wire


def old_history():
    return [{'role':'user','content':'old'}, {'role':'assistant','content':None,'tool_calls':[
        {'id':'old-load','type':'function','function':{'name':'skills_load','arguments':'{}'},**SKILLS_PROVENANCE},
        {'id':'keep','type':'function','function':{'name':'lookup','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'old-load','content':PRIVATE},
        {'role':'tool','tool_call_id':'keep','content':'keep-result'},
        {'role':'assistant','content':'old-final'}]


@pytest.mark.parametrize('native',['openai','anthropic','google'])
def test_stale_history_atomic_mixed_and_unmarked_names_preserved(native):
    old=old_history()
    if native=='anthropic': old[2]={'role':'user','content':[{'type':'tool_result','tool_use_id':'old-load','content':PRIVATE},{'type':'tool_result','tool_use_id':'keep','content':'keep-result'}]};old.pop(3)
    if native=='google': old[2]={'role':'user','content':[{'functionResponse':{'id':'old-load','name':'skills_load','response':{'result':PRIVATE}}},{'functionResponse':{'id':'keep','name':'lookup','response':{'result':'keep-result'}}}]};old.pop(3)
    old += [{'role':'assistant','tool_calls':[{'id':'unmarked','function':{'name':'skills_load','arguments':'{}'}}]},
            {'role':'tool','tool_call_id':'unmarked','content':'ordinary same-name result'}]
    before=copy.deepcopy(old);clean=strip_ephemeral_skills_history(old)
    assert old==before and PRIVATE not in json.dumps(clean)
    assert 'keep-result' in json.dumps(clean) and 'ordinary same-name result' in json.dumps(clean)
    assert 'old-load' not in json.dumps(clean)


@pytest.mark.asyncio
async def test_chat_strips_before_windowing_and_llm_strips_after_source_removal():
    chat_node=NodeChat(ChatNodeModel(history_messages=old_history(),max_messages=1),node_id='chat')
    [item async for item in chat_node.process(LOG)]
    assert PRIVATE not in str(chat_node.chat.messages)
    client,wire=client_with_wire_responses([None])
    node=llm_node(client)
    direct=ModelChat();direct.messages=old_history()
    before=copy.deepcopy(direct.messages)
    node.inputs[node.INPUT_HANDLER_CHAT]=direct
    [item async for item in node.process(LOG)]
    assert PRIVATE not in json.dumps(wire) and direct.messages==before


@pytest.mark.asyncio
async def test_context_overflow_after_loading_is_terminal_before_second_request():
    client,wire=client_with_wire_responses([[selected(['release-notes'])],None])
    node=llm_node(client,bundle(entry(prompt='x'*9000)),max_input_tokens=7000)
    events=[]
    with pytest.raises(Exception) as failure:
        async for item in node.process(LOG): events.append(item)
    assert error_code(failure.value)=='SKILLS_CONTEXT_LIMIT'
    assert len(wire)==1
    results=[e for e in events if e['type']=='debug' and e['content'].get('event_type')=='TOOL_RESULT']
    assert len(results)==1 and 'x'*9000 not in str(results)


def test_final_wire_budget_includes_adapter_expansion():
    chat=ModelChat();chat.add_user_message('small')
    context=AgentRequestContext(chat=chat,tools=[],tool_choice=None,provider='openai',model='fake',generation_options={'max_tokens':10})
    create_request_guard(2000)(context)
    assert chat.complete_context_required
    with pytest.raises(Exception) as failure:
        chat.validate_provider_payload({'messages':chat.messages,'native_schema':'x'*2000})
    assert error_code(failure.value)=='SKILLS_CONTEXT_LIMIT'


def test_loader_observer_summary_is_idempotent_and_error_status_correct():
    canonical=ToolResult(tool_call_id='load',name='skills_load',content=ToolExecutor.serialize_output({'schema_version':1,'source_node_id':'skills','skills':[bundle().skills[0].definition()]}))
    safe=safe_loader_result(canonical)
    assert PRIVATE in canonical.content and PRIVATE not in safe.content
    assert safe_loader_result(safe).content==safe.content
    relay=HookRelay(node_id='writer',skills_source_node_id='skills',skills_available_ids=['release-notes'])
    relay.on_tool_complete(safe,SimpleNamespace(iteration=1))
    event=relay.get_collected_tool_data_for_yield()[0]
    assert json.loads(event['data']['content'])['loaded_ids']==['release-notes']
    failed=ToolResult(tool_call_id='bad',name='skills_load',content='{}',is_error=True)
    relay.on_tool_complete(failed,SimpleNamespace(iteration=1))
    assert relay.get_collected_tool_data_for_yield()[0]['data']['status']=='error'

@pytest.mark.asyncio
@pytest.mark.parametrize('overflow',[False,True])
async def test_streamed_selection_replays_full_result_and_budget_failure(overflow):
    from magic_llm.model.ModelChatStream import ChatCompletionModel
    client=MagicLLM(engine='openai',model='fake',private_key='dummy')
    wire=[]
    async def stream(chat,**kwargs):
        encoded,_=client.llm.base.prepare_data(chat,**kwargs)
        wire.append(json.loads(encoded))
        calls=[{**selected(['release-notes'],'stream-load'),'index':0}] if len(wire)==1 else None
        yield ChatCompletionModel(id='stream',model='fake',choices=[{'index':0,'delta':{
            'content': None if calls else 'done','tool_calls':calls},'finish_reason':'tool_calls' if calls else 'stop'}])
    client.llm.async_stream_generate=stream
    node=NodeLLM(LlmNodeModel(stream=True,json_output=False,max_input_tokens=7000 if overflow else None),node_id='writer')
    node.inputs.update({node.INPUT_HANDLER_CLIENT_PROVIDER:client,node.INPUT_HANDLER_USER_MESSAGE:'Write release',
        node.INPUT_HANDLER_SKILLS:bundle(entry(prompt='x'*9000 if overflow else PRIVATE))})
    events=[]
    if overflow:
        with pytest.raises(Exception) as failure:
            async for item in node.process(LOG): events.append(item)
        assert error_code(failure.value)=='SKILLS_CONTEXT_LIMIT' and len(wire)==1
    else:
        events=[item async for item in node.process(LOG)]
        assert len(wire)==2 and PRIVATE in json.dumps(wire[1])
        assert node.generated=='done'
    assert PRIVATE not in json.dumps(wire[0]) and PROPS_PRIVATE not in json.dumps(wire[0])
    assert PRIVATE not in str(events) and PROPS_PRIVATE not in str(events)


@pytest.mark.asyncio
async def test_invocation_rebuild_clears_loaded_summary_and_uses_fresh_definitions():
    client,wire=client_with_wire_responses([[selected(['release-notes'],'first-load')],None])
    node=llm_node(client,bundle())
    [item async for item in node.process(LOG)]
    assert node._capture_internal_state()['skills']['loaded_ids']==['release-notes']
    node.inputs[node.INPUT_HANDLER_SKILLS]=bundle(entry(prompt='updated private',enabled=False))
    [item async for item in node.process(LOG)]
    assert node._capture_internal_state()['skills']['loaded_ids']==[]
    assert 'skills_load' not in json.dumps(wire[2]) and PRIVATE not in json.dumps(wire[2])


@pytest.mark.asyncio
async def test_invalid_batch_enters_error_history_then_valid_retry():
    client,wire=client_with_wire_responses([[selected(['release-notes','missing'],'bad')],[selected(['release-notes'],'good')],None])
    node=llm_node(client,bundle())
    [item async for item in node.process(LOG)]
    first_result=next(m for m in wire[1]['messages'] if m['role']=='tool')
    assert 'SKILLS_LOAD_UNKNOWN_ID' in first_result['content'] and PRIVATE not in first_result['content']
    assert PRIVATE in json.dumps(wire[2])
    assert node._skills_relay.skills_loaded_ids==['release-notes']


@pytest.mark.asyncio
async def test_bypass_clears_previous_input_and_missing_connected_source_fails():
    built=build(graph(),message='write')
    node=built.nodes['writer'];node.inputs[node.INPUT_HANDLER_SKILLS]=bundle()
    dispatcher=GraphEventDispatcher(built.nodes,built.edges)
    await dispatcher.dispatch_bypass('writer',node.INPUT_HANDLER_SKILLS)
    assert node.INPUT_HANDLER_SKILLS not in node.inputs
    with pytest.raises(OperationFailure): node._prepare_skills()


@pytest.mark.asyncio
async def test_pre_execution_observer_and_telemetry_never_receive_stale_bodies(caplog):
    from magic_agents.util.telemetry import _redact
    caplog.set_level('DEBUG')
    node=NodeChat(ChatNodeModel(max_messages=1),node_id='chat',debug=True)
    node.inputs[node.INPUT_HANDLER_MESSAGES]=old_history()
    observer=SimpleNamespace(on_node_start=AsyncMock(),on_node_end=AsyncMock(),on_node_error=AsyncMock())
    [item async for item in node(LOG,observer=observer)]
    assert PRIVATE not in str(observer.on_node_start.call_args)
    assert PRIVATE not in str(observer.on_node_end.call_args)
    assert PRIVATE not in caplog.text
    assert PRIVATE not in str(_redact(old_history()))
    raw=ModelChat();raw.messages=old_history()
    assert PRIVATE not in str(_redact(raw))


def test_native_google_parts_selectively_follow_canonical_order():
    old=old_history()
    old[1]['gemini_parts']=[{'text':'signed reasoning','thoughtSignature':'opaque'},
        {'functionCall':{'name':'skills_load','args':{'skill_ids':['release-notes']}}},
        {'functionCall':{'name':'lookup','args':{}}}]
    clean=strip_ephemeral_skills_history(old)
    assistant=next(m for m in clean if m.get('tool_calls'))
    assert [part['functionCall']['name'] for part in assistant['gemini_parts'] if 'functionCall' in part]==['lookup']
    assert assistant['gemini_parts'][0]['thoughtSignature']=='opaque'


def test_callback_projection_removes_only_new_builtin_pairs_and_preserves_canonical():
    guard=create_request_guard(20000,source_node_id='skills')
    chat=ModelChat();chat.messages=[{'role':'assistant','tool_calls':[{'id':'ordinary','function':{'name':'skills_load','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'ordinary','content':'ordinary same-name result'}]
    def context(): return AgentRequestContext(chat=chat,tools=[],tool_choice=None,provider='openai',model='fake',generation_options={})
    guard(context())
    chat.messages += [{'role':'assistant','tool_calls':[selected(['release-notes'],'new-load'), {'id':'other','function':{'name':'lookup','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'new-load','content':ToolExecutor.serialize_output({'schema_version':1,'source_node_id':'skills','skills':[bundle().skills[0].definition()]})},
        {'role':'tool','tool_call_id':'other','content':'ordinary mixed result'}]
    before=copy.deepcopy(chat.messages);guard(context())
    projected=chat.observer_projection()
    assert chat.messages==before and PRIVATE in json.dumps(chat.messages)
    assert PRIVATE not in json.dumps(projected.messages) and PROPS_PRIVATE not in json.dumps(projected.messages)
    assert 'ordinary same-name result' in json.dumps(projected.messages) and 'ordinary mixed result' in json.dumps(projected.messages)
    assert 'new-load' not in json.dumps(projected.messages)
    assert projected.extra_args['skills']['loaded_call_count']==1

@pytest.mark.parametrize('options',[{'max_tokens':2500},{'max_output_tokens':2500},{'generationConfig':{'maxOutputTokens':2500}}])
def test_effective_provider_output_reserve_is_required(options):
    chat=ModelChat();chat.add_user_message('small')
    context=AgentRequestContext(chat=chat,tools=[],tool_choice=None,provider='google',model='fake',generation_options=options)
    with pytest.raises(OperationFailure) as failure: create_request_guard(2000)(context)
    assert error_code(failure.value)=='SKILLS_CONTEXT_LIMIT'


def test_final_native_payload_output_reserve_catches_adapter_defaults():
    chat=ModelChat();chat.add_user_message('small')
    context=AgentRequestContext(chat=chat,tools=[],tool_choice=None,provider='google',model='fake',generation_options={'max_tokens':10})
    create_request_guard(2000)(context)
    with pytest.raises(Exception) as failure:
        chat.validate_provider_payload({'contents':[],'generationConfig':{'maxOutputTokens':2500}})
    assert error_code(failure.value)=='SKILLS_CONTEXT_LIMIT'


@pytest.mark.asyncio
async def test_real_provider_callback_and_debug_observe_only_projected_history(monkeypatch,caplog):
    import magic_llm.engine.engine_openai as engine_module
    wire=[];observed=[]
    class FakeHttp:
        async def __aenter__(self): return self
        async def __aexit__(self,*args): return False
        async def post_json(self,*,data,**kwargs):
            wire.append(json.loads(data))
            return response([selected(['release-notes'],'callback-load')] if len(wire)==1 else None).model_dump(exclude_none=True)
    monkeypatch.setattr(engine_module,'AsyncHttpClient',FakeHttp)
    monkeypatch.setenv('MAGIC_LLM_DEBUG_PAYLOAD','1')
    monkeypatch.setenv('MAGIC_LLM_DEBUG_PAYLOAD_FULL','1')
    caplog.set_level('DEBUG')
    async def callback(chat,*args): observed.append(copy.deepcopy(chat.messages))
    client=MagicLLM(engine='openai',model='fake',private_key='dummy',callback=callback)
    node=llm_node(client,bundle())
    [item async for item in node.process(LOG)]
    assert len(wire)==2 and PRIVATE in json.dumps(wire[1])
    assert len(observed)==2 and PRIVATE not in json.dumps(observed) and PROPS_PRIVATE not in json.dumps(observed)
    assert PRIVATE not in caplog.text and PROPS_PRIVATE not in caplog.text


@pytest.mark.parametrize('client_options',[{'api_info':{'private_key':'dummy','tool_choice':'none'}}, {'api_info':{'private_key':'dummy'},'extra_data':{'tool_choice':'required'}}])
def test_client_json_hidden_tool_choice_cannot_disable_skills(client_options):
    from magic_agents.node_system.NodeClientLLM import NodeClientLLM
    from magic_agents.models.factory.Nodes.ClientNodeModel import ClientNodeModel
    client=NodeClientLLM(ClientNodeModel(engine='openai',model='fake',**client_options),node_id='client').client
    node=llm_node(client,bundle());node._prepare_skills()
    with pytest.raises(OperationFailure) as failure:
        node._install_skills(client,ModelChat('base'),[],{},SimpleNamespace(tool_schemas=[]))
    assert error_code(failure.value)=='SKILLS_TOOL_CONFLICT'


def test_observer_idempotency_never_accepts_definition_fields_as_summary():
    raw=ToolResult(tool_call_id='x',name='skills_load',content=json.dumps({**SKILLS_PROVENANCE,'loaded_ids':[], 'status':'success','result_chars':10,'prompt':PRIVATE}))
    assert PRIVATE not in safe_loader_result(raw).content


def test_callback_projection_handles_reused_historical_id_by_invocation_boundary():
    guard=create_request_guard(20000,source_node_id='skills')
    chat=ModelChat();chat.messages=[{'role':'assistant','tool_calls':[{'id':'reused','function':{'name':'lookup','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'reused','content':'preserve initial ordinary result'},
        {'role':'user','content':'new request'}]
    def context(): return AgentRequestContext(chat=chat,tools=[],tool_choice=None,provider='openai',model='fake',generation_options={})
    initial=copy.deepcopy(chat.messages)
    guard(context())
    chat.messages += [{'role':'assistant','tool_calls':[selected(['release-notes'],'reused')]},
        {'role':'tool','tool_call_id':'reused','content':ToolExecutor.serialize_output({'schema_version':1,'source_node_id':'skills','skills':[bundle().skills[0].definition()]})}]
    canonical=copy.deepcopy(chat.messages)
    guard(context());projected=chat.observer_projection()
    assert chat.messages==canonical and PRIVATE in json.dumps(chat.messages)
    assert projected.messages==initial
    assert PRIVATE not in json.dumps(projected.messages) and PROPS_PRIVATE not in json.dumps(projected.messages)
    assert projected.extra_args['skills']['loaded_call_count']==1
