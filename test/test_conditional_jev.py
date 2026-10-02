"""Jev transport and LLM adapter share one contract and fail atomically."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from pydantic import ValidationError

from magic_agents.execution.condition_evaluator_jev import evaluate_jev, JevEvaluationError
from magic_agents.execution.condition_evaluator_llm import parse_judgments
from magic_agents.execution.condition_judgment_contract import validate_judgment_answers
from magic_agents.models.factory.Nodes.ConditionalNodeModel import ConditionalNodeModel, JevConnection
from magic_agents.node_system.NodeConditional import NodeConditional


QUESTIONS = {
    'intent': {'type': 'choice', 'instructions': 'Which intent?', 'criteria': {'yes': 'Approve', 'no': None}},
    'urgency': {'type': 'score', 'instructions': 'How urgent?', 'criteria': [{'label': 'Low'}, ['High']]},
    'ok': {'type': 'noul', 'instructions': 'Does it qualify?', 'criteria': {'true': None, 'false': 'Rejected'}},
}
ESTIMATES = {'answers': {'intent': {'yes': .8, 'no': .2}, 'urgency': {'0': .25, '1': .75}, 'ok': .9}}
ANSWERS = parse_judgments(json.dumps(ESTIMATES), QUESTIONS)


def payload(**changes):
    return {'model': 'jev-1', 'answers': deepcopy(ANSWERS), 'usage': {'input_tokens': 12, 'output_tokens': 7}, **changes}


class Response:
    def __init__(self, data=None, *, status=200, headers=None, text=None):
        self.status, self.headers = status, headers or {}
        self.body = json.dumps(payload() if data is None else data) if text is None else text
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    async def text(self):
        return self.body


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def mock_http(monkeypatch, *responses):
    session = Session(responses or [Response()])
    monkeypatch.setattr(aiohttp, 'ClientSession', lambda **kwargs: session)
    return session


def node(**changes):
    return NodeConditional(node_id='judge', node_type='conditional', evaluation_mode='jev',
        questions=deepcopy(QUESTIONS), condition="{{ 'yes' if answers.ok.noul >= 0.8 else 'no' }}",
        output_handles=['yes', 'no'], **changes)


async def collect(node):
    return [event async for event in node.process(None)]


@pytest.mark.parametrize('url', ['ftp://example.com', 'https://user:secret@example.com/v1',
    'https://example.com/v1?key=secret', 'https://example.com/v1#key',
    'https://example.com/v1?', 'https://example.com/v1#', 'https://example.com:bad', 'http:///v1'])
def test_invalid_base_urls_rejected(url):
    with pytest.raises(ValidationError):
        JevConnection(base_url=url)


def test_same_request_schema_accepts_singleton_choice_and_null_noul_criteria():
    questions = {'single': {'type': 'choice', 'instructions': 'Pick', 'criteria': {'only': None}}, **QUESTIONS}
    for mode in ['jev', 'llm']:
        cfg = ConditionalNodeModel(evaluation_mode=mode, questions=questions, condition="{{ 'yes' }}", output_handles=['yes'])
        assert cfg.questions['single'].criteria == {'only': None}
        assert cfg.questions['ok'].criteria['true'] is None
    answer = parse_judgments('{"answers":{"single":{"only":1}}}', {'single': questions['single']})
    assert answer['single'] == {'type': 'choice', 'choice': 'only', 'probabilities': {'only': 1.0}, 'confidence': 1.0}


def test_llm_native_and_legacy_response_formats_produce_the_same_jev_answers():
    native = deepcopy(ANSWERS)
    native['intent']['confidence'] = .9
    assert parse_judgments(json.dumps({'answers': native}), QUESTIONS) == ANSWERS


@pytest.mark.parametrize('mode', ['llm', 'jev'])
def test_semantic_modes_ignore_both_reserved_provider_handles(mode):
    judge = NodeConditional(node_id='judge', node_type='conditional', evaluation_mode=mode,
        questions=QUESTIONS, condition="{{ 'yes' }}", output_handles=['yes'],
        handles={'client_provider': 'custom-provider'})
    judge.inputs = {'handle_input': {'message': 'approve'}, 'custom-provider': object(), 'handle-client-provider': object()}
    assert judge._merge_inputs()['message'] == 'approve'
    assert not {'custom-provider', 'handle-client-provider'} & set(judge._merge_inputs())
    with pytest.raises(ValidationError):
        ConditionalNodeModel(evaluation_mode=mode, questions=QUESTIONS, condition="{{ 'yes' }}", output_handles=['yes'],
            handles={'input': 'handle-client-provider', 'client_provider': 'custom-provider'})


@pytest.mark.asyncio
@pytest.mark.parametrize('configured', [None, {}, {'api_key': ''}, {'api_key': '{{env.JEV_API_KEY}}'}])
async def test_jev_batches_questions_uses_env_and_requires_no_llm_provider(monkeypatch, configured):
    monkeypatch.setenv('JEV_API_KEY', 'fixture-secret')
    session = mock_http(monkeypatch)
    result = await evaluate_jev({'request': 'approve'}, QUESTIONS, configured, timeout=1)
    assert result.answers == ANSWERS
    assert result.usage['total_tokens'] == 19
    assert len(session.calls) == 1
    args, kwargs = session.calls[0]
    assert args == ('POST', 'https://api.typesafe.ai/v1/systemone')
    assert kwargs['headers']['Authorization'] == 'Bearer fixture-secret'
    assert kwargs['json'] == {'state': {'request': 'approve'}, 'model': 'jev-latest', 'questions': QUESTIONS}
    assert kwargs['allow_redirects'] is False


@pytest.mark.asyncio
async def test_explicit_connection_wins_and_base_url_trailing_slash_is_normalized(monkeypatch):
    monkeypatch.setenv('JEV_API_KEY', 'environment-secret')
    session = mock_http(monkeypatch)
    await evaluate_jev({}, QUESTIONS, {'base_url': 'https://proxy.example/api/', 'api_key': 'explicit-secret'}, timeout=1)
    assert session.calls[0][0][1] == 'https://proxy.example/api/systemone'
    assert session.calls[0][1]['headers']['Authorization'] == 'Bearer explicit-secret'


@pytest.mark.asyncio
async def test_native_request_id_header_is_used_for_usage_correlation(monkeypatch):
    mock_http(monkeypatch, Response(headers={'x-typesafe-request-id': 'jev-request-123', 'x-request-id': 'proxy-request'}))
    result = await evaluate_jev({}, QUESTIONS, {'api_key': 'fixture-secret'}, timeout=1)
    assert result.id == 'jev-request-123'


@pytest.mark.asyncio
async def test_base_url_environment_reference_resolves_only_at_transport(monkeypatch):
    connection = {'base_url': '{{env.JEV_BASE_URL}}', 'api_key': 'fixture-secret'}
    assert JevConnection(**connection).base_url == '{{env.JEV_BASE_URL}}'
    monkeypatch.setenv('JEV_BASE_URL', 'https://proxy.example/jev/v1/')
    session = mock_http(monkeypatch)
    await evaluate_jev({}, QUESTIONS, connection, timeout=1)
    assert session.calls[0][0][1] == 'https://proxy.example/jev/v1/systemone'
    monkeypatch.setenv('JEV_BASE_URL', 'https://user:fixture-secret@example.com/v1')
    with pytest.raises(JevEvaluationError) as exc:
        await evaluate_jev({}, QUESTIONS, connection, timeout=1)
    assert 'fixture-secret' not in str(exc.value)
    assert len(session.calls) == 1
    monkeypatch.delenv('JEV_BASE_URL')
    with pytest.raises(JevEvaluationError, match='base URL'):
        await evaluate_jev({}, QUESTIONS, connection, timeout=1)


@pytest.mark.asyncio
async def test_streamed_response_cap_prevents_buffering_oversized_bodies(monkeypatch):
    response = Response()
    class Stream:
        async def iter_chunked(self, size):
            for _ in range(33):
                yield b'x' * size
    response.content = Stream()
    mock_http(monkeypatch, response)
    with pytest.raises(JevEvaluationError, match='maximum size'):
        await evaluate_jev({}, QUESTIONS, {'api_key': 'fixture-secret'}, timeout=1)


@pytest.mark.asyncio
async def test_missing_key_fails_before_network(monkeypatch):
    monkeypatch.delenv('JEV_API_KEY', raising=False)
    session = mock_http(monkeypatch)
    with pytest.raises(JevEvaluationError, match='JEV_API_KEY'):
        await evaluate_jev({}, QUESTIONS, None, timeout=1)
    assert session.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [301, 401, 403, 422])
async def test_http_errors_do_not_expose_response_bodies_or_key(monkeypatch, status):
    session = mock_http(monkeypatch, Response(status=status, text='fixture-secret internal error'))
    with pytest.raises(JevEvaluationError) as exc:
        await evaluate_jev({}, QUESTIONS, {'api_key': 'fixture-secret'}, timeout=1)
    assert 'fixture-secret' not in str(exc.value)
    assert f'HTTP {status}' in str(exc.value)
    assert len(session.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [429, 500, 502, 503, 504, 529])
async def test_transient_http_retries_honor_retry_after(monkeypatch, status):
    session = mock_http(monkeypatch, Response(status=status, headers={'retry-after-ms': '0'}), Response())
    result = await evaluate_jev({}, QUESTIONS, {'api_key': 'fixture-secret'}, timeout=1)
    assert result.answers == ANSWERS
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_retry_budget_bounds_backoff_and_does_not_retry_timeouts(monkeypatch):
    session = mock_http(monkeypatch, Response(status=429, headers={'Retry-After': '60'}), Response())
    with pytest.raises(asyncio.TimeoutError):
        await evaluate_jev({}, QUESTIONS, {'api_key': 'fixture-secret'}, timeout=.001)
    assert len(session.calls) == 1
    session = mock_http(monkeypatch, aiohttp.ServerDisconnectedError('fixture-secret'))
    with pytest.raises(JevEvaluationError, match='connection failed') as exc:
        await evaluate_jev({}, QUESTIONS, {'api_key': 'fixture-secret'}, timeout=1)
    assert 'fixture-secret' not in str(exc.value)
    assert len(session.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['not JSON', 'null', '{"model":"jev","answers":{},"answers":{}}',
    '{"model":"jev","answers":{"x":NaN},"usage":{"input_tokens":1,"output_tokens":1}}'])
async def test_malformed_or_ambiguous_native_response_rejected(monkeypatch, text):
    mock_http(monkeypatch, Response(text=text))
    with pytest.raises(JevEvaluationError):
        await evaluate_jev({}, QUESTIONS, {'api_key': 'fixture-secret'}, timeout=1)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'wrong_type', 'bad_probability', 'bad_confidence',
    'wrong_choice', 'wrong_score', 'wrong_legend', 'extra_noul_confidence'])
def test_native_answer_validation_is_atomic(mutation):
    raw = deepcopy(ANSWERS)
    if mutation == 'missing':
        raw.pop('ok')
    elif mutation == 'extra':
        raw['extra'] = raw['ok']
    elif mutation == 'wrong_type':
        raw['ok']['type'] = 'score'
    elif mutation == 'bad_probability':
        raw['intent']['probabilities']['yes'] = True
    elif mutation == 'bad_confidence':
        raw['intent']['confidence'] = float('nan')
    elif mutation == 'wrong_choice':
        raw['intent']['choice'] = 'no'
    elif mutation == 'wrong_score':
        raw['urgency']['score'] = .1
    elif mutation == 'wrong_legend':
        raw['urgency']['legend']['0'] = 'Changed'
    else:
        raw['ok']['confidence'] = .9
    with pytest.raises(ValueError):
        validate_judgment_answers(raw, QUESTIONS)


@pytest.mark.asyncio
async def test_native_confidence_preserved_and_provider_switch_keeps_route_and_payload(monkeypatch):
    monkeypatch.setenv('JEV_API_KEY', 'fixture-secret')
    native = payload()
    native['answers']['intent']['confidence'] = .72
    mock_http(monkeypatch, Response(native))
    judge = node()
    judge.inputs = {'handle_input': {'message': 'approve', 'answers': 'untrusted state'}}
    events = await collect(judge)
    assert judge.selected_handle == 'yes'
    assert judge.answers['intent']['confidence'] == .72
    assert judge.answers['urgency']['legend'] == {'0': {'label': 'Low'}, '1': ['High']}
    state = judge.outputs['yes']['content']
    assert state['answers'] == 'untrusted state'
    assert 'fixture-secret' not in json.dumps(events, default=str)
    usage = next(e['content'] for e in events if isinstance(e.get('content'), dict) and e['content'].get('event_type') == 'LLM_GENERATION')
    assert usage['total_tokens'] == 19 and usage['provider'] == 'typesafe'
    llm = NodeConditional(node_id='llm', node_type='conditional', evaluation_mode='llm', questions=QUESTIONS,
        condition=judge.condition_template, output_handles=judge.output_handles)
    llm.inputs = {'handle_input': deepcopy(judge.inputs['handle_input']), 'handle-client-provider': SimpleNamespace(
        llm=SimpleNamespace(model='mock', engine_name='openai', async_generate=AsyncMock(return_value=SimpleNamespace(
            content=json.dumps(ESTIMATES), usage=None, model='mock', id='mock'))))}
    await collect(llm)
    assert llm.selected_handle == judge.selected_handle
    assert llm.outputs['yes']['content'] == state
    assert {k: set(v) for k, v in llm.answers.items()} == {k: set(v) for k, v in judge.answers.items()}


@pytest.mark.asyncio
async def test_invalid_billed_native_answers_emit_usage_and_never_reuse_route(monkeypatch):
    monkeypatch.setenv('JEV_API_KEY', 'fixture-secret')
    session = mock_http(monkeypatch, Response(), Response(payload(answers={})))
    judge = node(default_handle='no')
    judge.inputs = {'handle_input': 'approve'}
    await collect(judge)
    assert judge.selected_handle == 'yes'
    events = await collect(judge)
    assert judge.answers == {} and judge.outputs == {} and judge.selected_handle is None
    assert any(e['type'] == '__bypass_all__' for e in events)
    assert any(e.get('content', {}).get('event_type') == 'LLM_GENERATION' for e in events if isinstance(e.get('content'), dict))
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_overflowed_native_probability_keeps_billed_usage_before_validation_failure(monkeypatch):
    monkeypatch.setenv('JEV_API_KEY', 'fixture-secret')
    raw = json.dumps(payload()).replace('"noul": 0.9', '"noul": 1e999')
    mock_http(monkeypatch, Response(text=raw))
    judge = node(default_handle='no')
    judge.inputs = {'handle_input': 'approve'}
    events = await collect(judge)
    assert judge.answers == {} and judge.outputs == {} and judge.selected_handle is None
    assert any(e['type'] == '__bypass_all__' for e in events)
    usage = next(e['content'] for e in events if isinstance(e.get('content'), dict) and e['content'].get('event_type') == 'LLM_GENERATION')
    assert usage['total_tokens'] == 19
