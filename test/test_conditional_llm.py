"""Typed LLM judgments, deterministic routing, failure isolation and usage."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from magic_agents.agt_flow import create_node
from magic_agents.execution.condition_evaluator_llm import build_judgment_chat, parse_judgments
from magic_agents.execution.reactive_executor import execute_graph_reactive, execute_graph_loop_reactive
from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalNodeModel
from magic_agents.node_system.Node import Node
from magic_agents.node_system.NodeConditional import NodeConditional
from magic_agents.node_system.NodeLoop import NodeLoop


QUESTIONS = {
    'intent': {'type': 'choice', 'instructions': 'Which intent?', 'criteria': {'approve': 'Explicit approval', 'reject': 'Explicit rejection', 'other': None}},
    'urgency': {'type': 'score', 'instructions': 'How urgent?', 'criteria': ['No urgency', 'Time sensitive', 'Immediate emergency']},
    'ok': {'type': 'noul', 'instructions': 'Does it qualify?', 'criteria': {'true': 'Qualifies', 'false': 'Does not'}},
}
RAW = {'answers': {'intent': {'approve': .8, 'reject': .1, 'other': .1}, 'urgency': {'0': .1, '1': .6, '2': .3}, 'ok': .9}}


def config(**overrides):
    return {'evaluation_mode': 'llm', 'questions': deepcopy(QUESTIONS),
            'condition': "{{ 'yes' if answers.ok.noul >= 0.8 else 'no' }}",
            'output_handles': ['yes', 'no'], **overrides}


def response(content=None):
    return SimpleNamespace(content=json.dumps(RAW) if content is None else content,
        model='test-model', id='request-1', usage=SimpleNamespace(
            prompt_tokens=12, completion_tokens=7, total_tokens=19,
            cached_tokens_read=3, reasoning_tokens=2))


def client(content=None):
    return SimpleNamespace(llm=SimpleNamespace(model='test-model', engine_name='openai',
        async_generate=AsyncMock(return_value=response(content))))


def conditional(**overrides):
    return NodeConditional(node_id='cond', node_type='conditional', **config(**overrides))


async def collect(generator):
    return [item async for item in generator]


@pytest.mark.parametrize('changes', [
    {'questions': None}, {'questions': {}}, {'output_handles': None},
    {'evaluation_timeout': 0}, {'evaluation_timeout': -1},
    {'evaluation_timeout': float('inf')}, {'evaluation_timeout': float('nan')},
    {'evaluation_timeout': True}, {'evaluation_timeout': '30'},
    {'evaluation_mode': 'unknown'},
    {'questions': {'': QUESTIONS['ok']}},
    {'output_handles': ['yes', 'yes']}, {'output_handles': ['__bypass_all__']},
    {'output_handles': ['debug']},
    {'handles': {'input': 'same', 'client': 'same'}},
    {'handles': {'input': 'same', 'client_provider': 'same'}},
    {'handles': {'input': 'same', 'client_provider': 'same', 'client': 'different'}},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValidationError):
        ConditionalNodeModel(**config(**changes))


@pytest.mark.parametrize('question', [
    {'type': 'choice', 'instructions': 'x', 'criteria': {}},
    {'type': 'choice', 'instructions': 'x', 'criteria': {str(i): None for i in range(256)}},
    {'type': 'choice', 'instructions': 'x', 'criteria': {'a': 3, 'b': None}},
    {'type': 'score', 'instructions': 'x', 'criteria': ['one']},
    {'type': 'score', 'instructions': 'x', 'criteria': ['x'] * 11},
    {'type': 'noul', 'instructions': 'x', 'criteria': {'yes': 'wrong key'}},
    {'type': 'noul', 'instructions': True},
    {'type': 'noul', 'instructions': 'x', 'unknown': 'not allowed'},
    {'type': 'noul', 'instructions': {'value': float('nan')}},
    {'type': 'other', 'instructions': 'x'},
])
def test_invalid_question_grammar_is_rejected(question):
    with pytest.raises(ValidationError):
        ConditionalNodeModel(**config(questions={'q': question}))


def test_schema_supports_structured_descriptions_and_flexible_ids():
    model = ConditionalNodeModel(**config(questions={
        'question-id': {'type': 'choice', 'instructions': {'task': ['Judge the state']}, 'criteria': {'a': ['A'], 'b': {'meaning': 'B'}}},
        'score': {'type': 'score', 'instructions': ['Score'], 'criteria': ['low', {'label': 'high'}]},
        'flag': {'type': 'noul', 'instructions': 'True?'},
    }))
    assert model.evaluation_timeout == 30
    assert ConditionalNodeModel(condition="{{ 'yes' }}").evaluation_mode == 'jinja'


@pytest.mark.asyncio
async def test_jinja_mode_preserves_question_definitions_without_calling_llm():
    node = conditional(evaluation_mode='jinja', condition="{{ 'yes' if value else 'no' }}")
    llm_client = client()
    node.inputs = {'handle_input': 'hello', 'handle-client-provider': llm_client}
    await collect(node.process(None))
    assert node.selected_handle == 'yes'
    assert node.questions == QUESTIONS
    assert node.questions['intent']['criteria']['other'] is None
    llm_client.llm.async_generate.assert_not_called()


def test_questions_require_explicit_evaluation_mode():
    with pytest.raises(ValidationError, match='Specify evaluation_mode'):
        ConditionalNodeModel(condition="{{ 'yes' }}", questions=QUESTIONS, output_handles=['yes'])


@pytest.mark.asyncio
async def test_jinja_custom_primary_state_handle_can_use_the_new_client_default_name():
    data = {'evaluation_mode': 'jinja', 'condition': "{{ 'yes' if age >= 18 else 'no' }}",
        'handles': {'input': 'handle-client-provider'}, 'output_handles': ['yes', 'no']}
    validated = ConditionalNodeModel(**data)
    node = create_node({'id': 'cond', 'type': 'conditional', 'data': validated.model_dump(exclude_none=True)}, load_chat=None)
    node.inputs = {'handle-client-provider': {'age': 20}}
    assert node.has_state_inputs()
    events = await collect(node.process(None))
    assert node.selected_handle == 'yes'
    assert node.outputs['yes']['content']['age'] == 20
    assert next(event['content']['content']['input_count'] for event in events if event['type'] == 'end') == 1


def test_typed_results_match_reference_algebra():
    answers = parse_judgments(json.dumps(RAW), QUESTIONS)
    assert answers['intent']['choice'] == 'approve'
    assert answers['intent']['confidence'] == pytest.approx(.7)
    assert answers['urgency']['score'] == pytest.approx(1.2)
    assert answers['urgency']['confidence'] == pytest.approx(.4)
    assert answers['urgency']['legend'] == {'0': 'No urgency', '1': 'Time sensitive', '2': 'Immediate emergency'}
    assert answers['ok'] == {'type': 'noul', 'noul': .9}


def test_uniform_ties_follow_declared_order_and_normalization_is_auditable():
    raw = deepcopy(RAW)
    raw['answers']['intent'] = {'other': .4, 'reject': .4, 'approve': .4}
    raw['answers']['urgency'] = {'0': 1/3, '1': 1/3, '2': 1/3}
    diagnostics = {}
    answers = parse_judgments(json.dumps(raw), QUESTIONS, diagnostics=diagnostics)
    assert answers['intent']['choice'] == 'approve'
    assert answers['intent']['confidence'] == pytest.approx(0)
    assert answers['urgency']['confidence'] == 0
    assert diagnostics['normalized_distributions']['intent']['original_sum'] == pytest.approx(1.2)


@pytest.mark.parametrize('bad', [None, True, '0.8', -0.01, 1.01, float('nan'), float('inf')])
def test_invalid_probabilities_are_rejected(bad):
    for question, label in [('intent', 'approve'), ('urgency', '1'), ('ok', None)]:
        raw = deepcopy(RAW)
        if label is None:
            raw['answers'][question] = bad
        else:
            raw['answers'][question][label] = bad
        with pytest.raises(ValueError):
            parse_judgments(json.dumps(raw), QUESTIONS)


@pytest.mark.parametrize('content', [
    'not JSON', '```json\n{}\n```', 'null', '[]', '{}',
    '{"answers": {}}', '{"answers": {}, "route": "yes"}',
    '{"answers": {}, "answers": {}}',
])
def test_malformed_response_envelopes_are_rejected(content):
    with pytest.raises(ValueError):
        parse_judgments(content, QUESTIONS)


def test_missing_extra_and_zero_distribution_rejected():
    for probabilities in [{'approve': 1}, {'approve': 1, 'reject': 0, 'other': 0, 'extra': 0}, {'approve': 0, 'reject': 0, 'other': 0}]:
        raw = deepcopy(RAW)
        raw['answers']['intent'] = probabilities
        with pytest.raises(ValueError):
            parse_judgments(json.dumps(raw), QUESTIONS)


@pytest.mark.asyncio
async def test_factory_runtime_keeps_judgment_separate_from_state_and_emits_usage():
    node = create_node({'id': 'cond', 'type': 'conditional', 'data': config(handles={'input': 'state_in', 'client': 'judge_client'})}, load_chat=None)
    llm_client = client()
    node.inputs = {'state_in': {'message': 'Yes please', 'answers': 'user supplied evidence'}, 'judge_client': llm_client}
    events = await collect(node.process(None))
    assert node.selected_handle == 'yes'
    payload = node.outputs['yes']['content']
    assert payload['answers'] == 'user supplied evidence'
    assert payload['value']['message'] == 'Yes please'
    assert 'judge_client' not in payload
    assert node.answers['ok']['noul'] == .9
    assert node.questions['intent']['criteria']['other'] is None
    call = llm_client.llm.async_generate.call_args
    assert call.kwargs == {'json_output': True}
    messages = call.args[0].messages
    assert len(messages) == 2
    assert json.loads(messages[-1]['content'])['state'] == payload
    assert 'independently' in messages[0]['content']
    assert not any(q['type'] == 'noul' and 'noul' in q for q in node.questions.values())
    usage = next(e['content'] for e in events if e.get('content', {}).get('event_type') == 'LLM_GENERATION')
    assert usage['prompt_tokens'] == 12
    assert usage['cached_tokens_read'] == 3
    assert usage['provider_request_id'] == 'request-1'
    assert usage['node_type'] == 'conditional'
    assert usage['id'] == 'request-1'
    assert node._capture_internal_state()['confidence_source'] == 'llm_estimated_not_calibrated'


@pytest.mark.asyncio
async def test_usage_events_have_unique_call_identity_when_provider_omits_request_ids():
    node = conditional()
    llm_client = client()
    llm_client.llm.async_generate.return_value = response()
    llm_client.llm.async_generate.return_value.id = None
    node.inputs = {'handle_input': 'approve', 'handle-client-provider': llm_client}
    events = await collect(node.process(None)) + await collect(node.process(None))
    usage = [event['content'] for event in events if isinstance(event.get('content'), dict) and event['content'].get('event_type') == 'LLM_GENERATION']
    assert len(usage) == 2 and usage[0]['id'] != usage[1]['id']
    assert all(item['node_type'] == 'conditional' and item['provider_request_id'] is None for item in usage)


@pytest.mark.asyncio
async def test_client_provider_alias_has_same_priority_in_schema_and_runtime():
    node = conditional(handles={'client_provider': 'canonical-client', 'client': 'ignored-client'})
    assert node.INPUT_HANDLE_CLIENT == 'canonical-client'
    llm_client = client()
    node.inputs = {'handle_input': 'approve', 'canonical-client': llm_client}
    await collect(node.process(None))
    assert node.selected_handle == 'yes'
    assert 'canonical-client' not in node.outputs['yes']['content']


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['bad_json', 'provider', 'missing_client', 'timeout', 'unknown_route', 'missing_answer_route'])
async def test_errors_never_use_default_or_stale_route(failure):
    node = conditional(default_handle='no')
    llm_client = client()
    node.inputs = {'handle_input': 'hello', 'handle-client-provider': llm_client}
    await collect(node.process(None))
    assert node.selected_handle == 'yes'
    if failure == 'bad_json':
        llm_client.llm.async_generate.return_value = response('invalid')
    elif failure == 'provider':
        llm_client.llm.async_generate.side_effect = RuntimeError('unavailable')
    elif failure == 'missing_client':
        node.inputs.pop('handle-client-provider')
    elif failure == 'timeout':
        async def slow(*args, **kwargs):
            await asyncio.sleep(10)
        llm_client.llm.async_generate.side_effect = slow
        node.evaluation_timeout = .001
    elif failure == 'unknown_route':
        node.condition_template = "{{ 'unknown' }}"
    else:
        node.condition_template = "{{ answers.nonexistent.choice }}"
    events = await collect(node.process(None))
    assert node.selected_handle is None
    assert node.outputs == {}
    assert node._response is None
    assert any(e['type'] == '__bypass_all__' for e in events)
    if failure == 'bad_json':
        assert any(e.get('content', {}).get('event_type') == 'LLM_GENERATION' for e in events if isinstance(e.get('content'), dict))


@pytest.mark.asyncio
@pytest.mark.parametrize('metadata', [
    {'finish_reason': 'length'}, {'finish_reason': 'content_filter'},
    {'finish_reason': 'tool_calls'}, {'refusal': 'Cannot answer'},
    {'tool_calls': [{'id': 'call-1'}]},
    {'choices': [SimpleNamespace(finish_reason='stop', message=SimpleNamespace(refusal='Cannot answer'))]},
    {'choices': [SimpleNamespace(finish_reason='length', message=SimpleNamespace())]},
])
async def test_valid_json_cannot_override_incomplete_refused_or_tool_call_provider_status(metadata):
    node = conditional(default_handle='no')
    llm_client = client()
    provider_response = response()
    for key, value in metadata.items():
        setattr(provider_response, key, value)
    llm_client.llm.async_generate.return_value = provider_response
    node.inputs = {'handle_input': 'approve', 'handle-client-provider': llm_client}
    events = await collect(node.process(None))
    assert node.selected_handle is None and node.answers == {} and node.outputs == {}
    assert any(event['type'] == '__bypass_all__' for event in events)
    assert any(event.get('content', {}).get('event_type') == 'LLM_GENERATION' for event in events if isinstance(event.get('content'), dict))


class Source(Node):
    def __init__(self, node_id, value):
        super().__init__(node_id=node_id, node_type='test')
        self.value = value
    async def process(self, log):
        yield self.yield_static(self.value, 'out')


class Sink(Node):
    def __init__(self, node_id):
        super().__init__(node_id=node_id, node_type='test')
        self.seen = []
    async def process(self, log):
        self.seen.append(deepcopy(self.inputs))
        yield self.yield_static(self.inputs, 'out')


def edge(source, target, source_handle='out', target_handle='in'):
    return EdgeNodeModel(id=f'{source}-{source_handle}-{target}-{target_handle}', source=source, target=target, sourceHandle=source_handle, targetHandle=target_handle)


def graph(nodes, edges):
    result = AgentFlowModel(type='graph', nodes={n.node_id: n for n in nodes}, edges=edges, debug=False)
    result._validation_errors = []
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [False, True])
async def test_graph_fanout_and_convergence_or_failure_bypass(bad):
    c = conditional()
    a, b, no, end = (Sink(name) for name in ['a', 'b', 'no', 'end'])
    source, provider = Source('state', 'hello'), Source('client', client('bad' if bad else None))
    g = graph([source, provider, c, a, b, no, end], [
        edge('state', 'cond', target_handle='handle_input'), edge('client', 'cond', target_handle='handle-client-provider'),
        edge('cond', 'a', 'yes'), edge('cond', 'b', 'yes'), edge('cond', 'no', 'no'),
        edge('a', 'end', target_handle='a'), edge('b', 'end', target_handle='b'), edge('no', 'end', target_handle='no'),
    ])
    await asyncio.wait_for(collect(execute_graph_reactive(g)), 2)
    assert len(a.seen) == len(b.seen) == len(end.seen) == (0 if bad else 1)
    assert not no.seen


@pytest.mark.asyncio
async def test_loop_client_readiness_does_not_evaluate_before_item_and_failure_does_not_reuse_route():
    from magic_agents.hooks.hook_registry import HookRegistry
    lifecycle = SimpleNamespace(on_graph_error=AsyncMock(), on_graph_end=AsyncMock())
    llm_client = client()
    llm_client.llm.async_generate.side_effect = [response(), response('malformed'), response()]
    loop = NodeLoop(node_id='loop', node_type='loop')
    c = conditional()
    yes, no, end = Sink('yes'), Sink('no'), Sink('end')
    g = graph([Source('list', ['first', 'second', 'third']), Source('client', llm_client), loop, c, yes, no, end], [
        edge('list', 'loop', target_handle='handle_list'), edge('client', 'cond', target_handle='handle-client-provider'),
        edge('loop', 'cond', 'handle_item', 'handle_input'), edge('cond', 'yes', 'yes'), edge('cond', 'no', 'no'),
        edge('yes', 'loop', target_handle='handle_loop'), edge('no', 'loop', target_handle='handle_loop'),
        edge('loop', 'end', 'handle_end'),
    ])
    events = await asyncio.wait_for(collect(execute_graph_loop_reactive(g, hooks=HookRegistry(global_hooks=[lifecycle]))), 2)
    assert llm_client.llm.async_generate.call_count == 3
    states = [json.loads(call.args[0].messages[-1]['content'])['state']['value'] for call in llm_client.llm.async_generate.call_args_list]
    assert states == ['first', 'second', 'third']
    assert len(yes.seen) == 2
    assert not no.seen
    assert c.selected_handle == 'yes'
    assert c.answers['ok']['noul'] == .9
    lifecycle.on_graph_error.assert_awaited_once()
    lifecycle.on_graph_end.assert_not_awaited()
    assert any(e.get('content', {}).get('error_type') == 'JudgmentError' for e in events if isinstance(e.get('content'), dict))


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['static', 'post_loop'])
async def test_loop_failure_bypasses_branches_in_each_phase(phase):
    c = conditional()
    loop = NodeLoop(node_id='loop', node_type='loop')
    yes, no, end = Sink('yes'), Sink('no'), Sink('end')
    nodes = [Source('list', ['item']), Source('client', client('bad')), loop, c, yes, no, end]
    edges = [edge('list', 'loop', target_handle='handle_list'), edge('client', 'cond', target_handle='handle-client-provider'),
        edge('cond', 'yes', 'yes'), edge('cond', 'no', 'no')]
    if phase == 'static':
        nodes.append(Source('state', 'hello'))
        edges += [edge('state', 'cond', target_handle='handle_input'), edge('loop', 'end', 'handle_end')]
    else:
        edges += [edge('loop', 'cond', 'handle_end', 'handle_input'), edge('yes', 'end'), edge('no', 'end')]
    await asyncio.wait_for(collect(execute_graph_loop_reactive(graph(nodes, edges))), 2)
    assert not yes.seen and not no.seen
