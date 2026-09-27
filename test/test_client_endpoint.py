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


async def test_runtime_model_change_keeps_endpoint():
    node = NodeClientLLM(ClientNodeModel(engine='openai', model='first', endpoint='responses'), node_id='client')
    node.inputs[node.INPUT_HANDLE_MODEL] = 'second'
    events = [event async for event in node.process(None)]
    assert events
    assert node.client.llm.model == 'second'
    assert node.client.llm.endpoint == 'responses'
