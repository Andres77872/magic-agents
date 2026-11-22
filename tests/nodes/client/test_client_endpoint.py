import json

import pytest
from pydantic import ValidationError

from magic_agents.models.factory.Nodes import ClientNodeModel
from magic_agents.node_system.NodeClientLLM import NodeClientLLM


@pytest.mark.parametrize('config', [
    {'endpoint': 'responses'},
    {'api_info': {'api_key': 'test-only', 'endpoint': 'responses'}},
    {'config': '{"api_key":"test-only","endpoint":"responses"}'},
    {'extra_data': {'endpoint': 'responses'}},
])
def test_client_initializes_real_responses_engine(config):
    node = NodeClientLLM(ClientNodeModel(engine='openai', model='gpt-6-luna', **config), node_id='client')
    assert node.init_error is None
    assert node.client.llm.chat_url == 'https://api.openai.com/v1/responses'


def test_explicit_node_choice_overrides_nested_config_and_extra_data(monkeypatch):
    monkeypatch.setenv('CLIENT_CONFIG', json.dumps({'api_key': 'test-only', 'endpoint': 'responses'}))
    node = NodeClientLLM(ClientNodeModel(engine='openai', endpoint='chat_completions',
        api_info='{{env.CLIENT_CONFIG}}', extra_data={'endpoint': 'responses'}), node_id='client')
    assert node.init_error is None
    assert node.client.llm.endpoint == 'chat_completions'


def test_omitted_endpoint_keeps_existing_default():
    node = NodeClientLLM(ClientNodeModel(engine='openai', model='gpt-4o', api_info={'api_key': 'test-only'}), node_id='client')
    assert node.init_error is None
    assert node.client.llm.endpoint == 'chat_completions'


def test_invalid_endpoint_is_rejected_by_graph_schema():
    with pytest.raises(ValidationError):
        ClientNodeModel(engine='openai', endpoint='invalid')


def test_responses_with_other_engines_returns_configuration_error():
    node = NodeClientLLM(ClientNodeModel(engine='anthropic', endpoint='responses'), node_id='client')
    assert "Responses requires engine='openai'" in node.init_error
    assert node.init_error_type == 'ClientConfigurationError'


async def test_provider_constructor_errors_stay_redacted(monkeypatch, caplog):
    import sys
    # magic_agents.node_system re-exports the class under the module's name.
    module = sys.modules[NodeClientLLM.__module__]

    def failing_client(**kwargs):
        raise ValueError(f"bad key {kwargs['private_key']}")

    monkeypatch.setattr(module, 'MagicLLM', failing_client)
    caplog.set_level('DEBUG')
    node = NodeClientLLM(ClientNodeModel(engine='openai', api_info={'api_key': 'sk-secret-value'}),
                         node_id='client', debug=True)
    events = [event async for event in node.process(None)]

    rendered = repr(events) + repr(node._capture_internal_state()) + caplog.text
    assert node.client is None
    assert node.init_error == module.PROVIDER_INIT_ERROR
    assert node.init_error_type == 'ValueError'
    assert [e['content']['error_type'] for e in events if e['type'] == 'debug'] == ['ConfigurationError']
    assert not [e for e in events if e['type'] == node.OUTPUT_HANDLE]
    assert 'sk-secret-value' not in rendered


async def test_client_is_built_once_and_reused_across_runs(monkeypatch):
    import sys
    module = sys.modules[NodeClientLLM.__module__]
    built = []

    def counting_client(**kwargs):
        built.append(kwargs)
        return object()

    monkeypatch.setattr(module, 'MagicLLM', counting_client)
    node = NodeClientLLM(ClientNodeModel(engine='openai', model='m', api_info={'api_key': 'test-only'}),
                         node_id='client')

    first = [e['content']['content'] async for e in node.process(None) if e['type'] == node.OUTPUT_HANDLE]
    second = [e['content']['content'] async for e in node.process(None) if e['type'] == node.OUTPUT_HANDLE]

    assert len(built) == 1
    assert built[0]['private_key'] == 'test-only'  # api_key is aliased for MagicLLM
    assert first == second == [node.client]


async def test_runtime_model_change_keeps_endpoint():
    node = NodeClientLLM(ClientNodeModel(engine='openai', model='first', endpoint='responses'), node_id='client')
    node.inputs[node.INPUT_HANDLE_MODEL] = 'second'
    events = [event async for event in node.process(None)]
    assert events
    assert node.client.llm.model == 'second'
    assert node.client.llm.endpoint == 'responses'


@pytest.mark.parametrize(("authored", "expected"), [
    ({"provider": "anthropic"}, {"engine": "anthropic"}),
    ({"engine": "openai", "provider": "anthropic"}, {"engine": "openai"}),
    ({"model_name": "m-1"}, {"model": "m-1"}),
    ({"model": "m-2", "model_name": "m-1"}, {"model": "m-2"}),
    ({"config": {"api_key": "k"}}, {"api_info": {"api_key": "k"}}),
    ({"credentials": {"api_key": "k"}}, {"api_info": {"api_key": "k"}}),
    ({"config": {"a": 1}, "credentials": {"b": 2}}, {"api_info": {"a": 1}}),
    ({"api_info": {"x": 1}, "config": {"a": 1}}, {"api_info": {"x": 1}}),
])
def test_client_model_resolves_documented_aliases_with_canonical_fields_winning(authored, expected):
    model = ClientNodeModel(**authored)

    assert {field: getattr(model, field) for field in expected} == expected
