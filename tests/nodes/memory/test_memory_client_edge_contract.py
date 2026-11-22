"""Memory's LLM client edge is a declared input, including when renamed."""
import pytest

from magic_agents.agt_flow import build


def memory_graph(client_handle='handle-client-provider', handles=None):
    memory_data = {'instructions': 'Extract key points'}
    if handles:
        memory_data['handles'] = handles
    return {
        'contract_config': {'mode': 'strict'},
        'nodes': [
            {'id': 'input', 'type': 'user_input', 'data': {}},
            {'id': 'client', 'type': 'client', 'data': {'engine': 'openai', 'api_info': {'api_key': 'test-only'}}},
            {'id': 'memory', 'type': 'memory', 'data': memory_data},
            {'id': 'chat', 'type': 'chat'},
            {'id': 'end', 'type': 'end'},
        ],
        'edges': [
            {'source': 'input', 'target': 'memory', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_memory_input'},
            {'source': 'client', 'target': 'memory', 'sourceHandle': 'handle-client-provider', 'targetHandle': client_handle},
            {'source': 'memory', 'target': 'chat', 'sourceHandle': 'handle_memory_output', 'targetHandle': 'handle_user_message'},
            {'source': 'chat', 'target': 'end', 'sourceHandle': 'handle_chat_output', 'targetHandle': 'handle_flow_input'},
        ],
    }


def unknown_targets(graph):
    return [d for d in graph._contract_report.diagnostics if d.get('code') == 'PORT_TARGET_HANDLE_UNKNOWN']


@pytest.mark.parametrize('client_handle,handles', [
    ('handle-client-provider', None),
    ('memory-llm', {'client': 'memory-llm'}),
])
def test_memory_client_edge_is_a_known_input(client_handle, handles):
    graph = build(memory_graph(client_handle, handles), message='hello')
    assert unknown_targets(graph) == []
    assert graph.nodes['memory'].CLIENT_HANDLE == client_handle
