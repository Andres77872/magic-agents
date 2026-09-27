"""Reasoning effort is a provider string, with per-invocation graph overrides."""
import copy

import pytest
from pydantic import ValidationError

from magic_agents.agt_flow import create_node, validate_graph
from magic_agents.models.factory.Nodes import LlmNodeModel
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.util.handle_registry import get_port_cardinality


@pytest.mark.parametrize('effort', ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'Vendor/Deep-v2'])
def test_factory_accepts_and_round_trips_provider_names(effort):
    data = LlmNodeModel(reasoning_effort=effort).model_dump(exclude_none=True)
    node = create_node({'id': 'llm', 'type': 'llm', 'data': data}, load_chat=None)
    assert node._build_runtime_extra_data()['reasoning_effort'] == effort


@pytest.mark.parametrize('effort', [None, '', '  '])
def test_empty_is_omitted_and_old_graphs_keep_defaults(effort):
    assert 'reasoning_effort' not in NodeLLM(LlmNodeModel(reasoning_effort=effort), 'llm')._build_runtime_extra_data()
    assert 'reasoning_effort' not in NodeLLM(LlmNodeModel(), 'llm')._build_runtime_extra_data()


@pytest.mark.parametrize('effort', [False, 1, [], {}])
def test_non_string_names_are_rejected(effort):
    with pytest.raises(ValidationError, match='reasoning_effort'):
        LlmNodeModel(reasoning_effort=effort)
    node = NodeLLM(LlmNodeModel(), 'llm')
    node.inputs[node.INPUT_HANDLER_REASONING_EFFORT] = effort
    with pytest.raises(ValueError, match='reasoning_effort'):
        node._build_runtime_extra_data()


def test_runtime_then_node_then_legacy_precedence_and_no_loop_leakage():
    legacy = {'reasoning_effort': 'low', 'reasoning': {'effort': 'low', 'summary': 'auto'}}
    original = copy.deepcopy(legacy)
    node = NodeLLM(LlmNodeModel(reasoning_effort=' high ', extra_data=legacy), 'llm', handles={'reasoning_effort': 'effort'})
    assert node._build_runtime_extra_data()['reasoning_effort'] == 'high'
    node.inputs['effort'] = ' Vendor/Deep-v2 '
    assert node._build_runtime_extra_data()['reasoning_effort'] == 'Vendor/Deep-v2'
    node.inputs['effort'] = '   '
    cleared = node._build_runtime_extra_data()
    assert 'reasoning_effort' not in cleared
    assert cleared['reasoning'] == {'summary': 'auto'}
    node.inputs.clear()
    assert node._build_runtime_extra_data()['reasoning_effort'] == 'high'
    assert legacy == original
    assert NodeLLM(LlmNodeModel(extra_data=legacy), 'legacy')._build_runtime_extra_data()['reasoning_effort'] == 'low'


def test_reasoning_input_is_registered_as_exclusive():
    from magic_agents.util.handle_registry import CANONICAL_INPUT_HANDLES
    assert 'handle-llm-reasoning_effort' in CANONICAL_INPUT_HANDLES['llm']
    assert get_port_cardinality('llm', 'handle-llm-reasoning_effort').exclusive
