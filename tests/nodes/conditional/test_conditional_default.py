"""Opt-in judgment recovery selects original state without masking route errors."""
import asyncio
from copy import deepcopy
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from magic_agents.execution.condition_evaluator_jev import JevResponse
from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalNodeModel
from magic_agents.node_system.NodeConditional import NodeConditional


QUESTIONS = {'urgency': {'type': 'score', 'instructions': 'How urgent?', 'criteria': ['Low', 'Medium', 'High']}}
ANSWER = {'urgency': {'type': 'score', 'score': 1.65, 'legend': {'0': 'Low', '1': 'Medium', '2': 'High'},
    'probabilities': {'0': 0, '1': .35, '2': .65}, 'confidence': .5}}


def config(**changes):
    return {'evaluation_mode': 'jev', 'evaluation_error_policy': 'default',
        'questions': deepcopy(QUESTIONS), 'condition': "{{ 'urgent' if answers.urgency.score >= 1.5 else 'review' }}",
        'output_handles': ['urgent', 'review'], 'default_handle': 'review', **changes}


@pytest.mark.parametrize('changes', [
    {'evaluation_error_policy': 'other'}, {'default_handle': None},
    {'output_handles': None}, {'default_handle': 'missing'},
])
def test_default_error_policy_requires_a_declared_default(changes):
    with pytest.raises(ValidationError):
        ConditionalNodeModel(**config(**changes))


def test_error_policy_defaults_to_fail_and_can_be_preserved_in_exact_mode():
    assert ConditionalNodeModel(**config(evaluation_error_policy='fail')).evaluation_error_policy == 'fail'
    assert ConditionalNodeModel(condition="{{ 'yes' }}").evaluation_error_policy == 'fail'
    assert ConditionalNodeModel(**config(evaluation_mode='jinja')).evaluation_error_policy == 'default'


def provider(monkeypatch, judge, answers=None, error=None):
    answers = deepcopy(ANSWER) if answers is None else answers
    usage = {'prompt_tokens': 12, 'completion_tokens': 7, 'total_tokens': 19}
    if judge.evaluation_mode == 'jev':
        response = JevResponse(model='jev-test', answers=answers, usage=usage, id='jev-test')
        evaluate = AsyncMock(return_value=response, side_effect=error)
        monkeypatch.setattr(importlib.import_module('magic_agents.node_system.NodeConditional'), 'evaluate_jev', evaluate)
    else:
        response = SimpleNamespace(content=json.dumps({'answers': answers}), usage=usage, model='llm-test', id='llm-test')
        evaluate = AsyncMock(return_value=response, side_effect=error)
        judge.inputs['handle-client-provider'] = SimpleNamespace(llm=SimpleNamespace(
            model='llm-test', engine_name='openai', async_generate=evaluate))
    return evaluate


async def collect(judge):
    return [event async for event in judge.process(None)]


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['jev', 'llm'])
@pytest.mark.parametrize('failure', ['score_mismatch', 'provider', 'timeout'])
async def test_judgment_failures_use_original_state_default_and_safe_diagnostics(monkeypatch, mode, failure):
    judge = NodeConditional(node_id='judge', node_type='conditional', **config(evaluation_mode=mode))
    state = {'message': 'Handle my ticket', 'answers': 'untrusted evidence'}
    judge.inputs = {'handle_input': deepcopy(state)}
    malformed = deepcopy(ANSWER)
    malformed['urgency']['score'] = 1.4
    error = None
    if failure == 'provider':
        error = RuntimeError('credential secret must never appear in recovery diagnostics')
    elif failure == 'timeout':
        async def slow(*args, **kwargs):
            await asyncio.sleep(10)
        error = slow
        judge.evaluation_timeout = .001
    provider(monkeypatch, judge, malformed, error)
    events = await collect(judge)
    assert judge.selected_handle == 'review'
    assert judge.outputs['review']['content']['message'] == state['message']
    assert judge.outputs['review']['content']['answers'] == state['answers']
    assert judge.answers == {} and judge.used_default is True
    assert not any(event['type'] == '__bypass_all__' for event in events)
    recovery = next(event['content'] for event in events if isinstance(event.get('content'), dict)
        and event['content'].get('event_type') == 'CONDITIONAL_FALLBACK')
    assert recovery['judgment_error']['code'] == ('EVALUATION_TIMEOUT' if failure == 'timeout' else 'JUDGMENT_ERROR')
    assert 'secret' not in json.dumps(recovery)
    summary = next(event['content']['content'] for event in events if event['type'] == 'end')
    assert summary['used_default'] is True and summary['judgment_error'] == recovery['judgment_error']
    assert summary['answers'] == {}
    usage_events = [event for event in events if isinstance(event.get('content'), dict)
        and event['content'].get('event_type') == 'LLM_GENERATION']
    assert len(usage_events) == (1 if failure == 'score_mismatch' else 0)
    if usage_events:
        assert usage_events[0]['content']['total_tokens'] == 19


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['jev', 'llm'])
async def test_success_after_default_drops_previous_error_and_default_route(monkeypatch, mode):
    judge = NodeConditional(node_id='judge', node_type='conditional', **config(evaluation_mode=mode))
    judge.inputs = {'handle_input': 'ticket'}
    malformed = deepcopy(ANSWER)
    malformed['urgency']['score'] = 1.4
    evaluate = provider(monkeypatch, judge, malformed)
    await collect(judge)
    assert judge.used_default is True and judge.selected_handle == 'review'
    evaluate.return_value = JevResponse(model='jev-test', answers=ANSWER, usage={}, id='test') if mode == 'jev' else SimpleNamespace(
        content=json.dumps({'answers': ANSWER}), usage=None, model='llm-test', id='test')
    await collect(judge)
    assert judge.used_default is False and judge.judgment_error is None
    assert judge.selected_handle == 'urgent' and 'review' not in judge.outputs


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['jev', 'llm'])
@pytest.mark.parametrize('condition', ["{{ answers.missing.score }}", "{{ 'undeclared' }}"])
async def test_successful_judgments_do_not_mask_condition_or_routing_errors(monkeypatch, mode, condition):
    judge = NodeConditional(node_id='judge', node_type='conditional', **config(evaluation_mode=mode, condition=condition))
    judge.inputs = {'handle_input': 'ticket'}
    provider(monkeypatch, judge)
    events = await collect(judge)
    assert judge.selected_handle is None and judge.outputs == {}
    assert judge.used_default is False and judge.judgment_error is None
    assert any(event['type'] == '__bypass_all__' for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['jev', 'llm'])
async def test_default_handle_alone_still_fails_judgment_errors_unless_policy_enabled(monkeypatch, mode):
    judge = NodeConditional(node_id='judge', node_type='conditional', **config(evaluation_mode=mode, evaluation_error_policy='fail'))
    judge.inputs = {'handle_input': 'ticket'}
    provider(monkeypatch, judge, {})
    events = await collect(judge)
    assert judge.selected_handle is None and judge.outputs == {} and judge.used_default is False
    assert any(event['type'] == '__bypass_all__' for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['jev', 'llm'])
async def test_recovered_provider_error_hooks_have_safe_messages(monkeypatch, mode):
    judge = NodeConditional(node_id='judge', node_type='conditional', **config(evaluation_mode=mode))
    judge.inputs = {'handle_input': 'ticket'}
    provider(monkeypatch, judge, error=RuntimeError('raw provider credential-secret'))
    captured = []
    async def invoke(event, context):
        captured.append((event, str(context.error), deepcopy(context.outputs)))
    judge._hooks = SimpleNamespace(is_empty=lambda: False, execution_id='test', run_id='test', invoke=invoke)
    await collect(judge)
    ended = next(item for item in captured if item[0] == 'on_llm_end')
    assert 'credential-secret' not in json.dumps(ended)
    assert judge.selected_handle == 'review'
