"""Nested JSON graphs retain host runtime services, history and custom handles."""
import pytest

from magic_agents.agt_flow import build, create_node
from magic_agents.vector_storage import InMemoryVectorDB


def _child_graph():
    return {
        'nodes': [
            {'id': 'child-input', 'type': 'user_input'},
            {'id': 'child-memory', 'type': 'memory'},
            {'id': 'child-chat', 'type': 'chat'},
            {'id': 'child-llm', 'type': 'llm'},
        ],
        'edges': [],
    }


@pytest.mark.parametrize('alias', ['magic_flow', 'flow', 'graph', 'subgraph'])
@pytest.mark.parametrize('wrapped', [False, True])
def test_inner_graph_inherits_host_dependencies_and_history(alias, wrapped):
    child = _child_graph()
    if wrapped:
        child = {'type': 'graph', 'content': child}
    definition = {
        'nodes': [
            {'id': 'input', 'type': 'user_input'},
            {'id': 'inner', 'type': 'inner', 'data': {alias: child}},
        ],
        'edges': [],
    }
    vector_db, embedding = InMemoryVectorDB(), object()
    history = [{'role': 'user', 'content': 'Remember the project context'}]
    graph = build(definition, message='current', history_messages=history, deps={
        'child-memory': {'vector_db': vector_db, 'embedding_client': embedding, 'history_messages': history},
    })
    nested = graph.nodes['inner'].inner_graph
    memory = nested.nodes['child-memory']
    assert memory._vector_db is vector_db
    assert memory._embedding_client is embedding
    assert memory._history_messages == history
    assert nested.nodes['child-chat']._history_messages == history
    assert nested.nodes['child-llm']._history_messages == history


def test_chat_factory_preserves_json_handle_overrides():
    chat = create_node({
        'id': 'chat', 'type': 'chat',
        'data': {'handles': {'user_message': 'custom-message', 'output': 'custom-chat'}},
    }, load_chat=None)
    assert chat.INPUT_HANDLER_USER_MESSAGE == 'custom-message'
    assert chat.OUTPUT_HANDLE == 'custom-chat'
