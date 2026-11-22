"""Nested MEMORY nodes with repeated local IDs never share runtime/vector scope."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from magic_agents.agt_flow import build, run_agent


def _child():
    return {'nodes': [{'id': 'input', 'type': 'user_input'}, {'id': 'memory', 'type': 'memory'}, {'id': 'end', 'type': 'end'}],
            'edges': [{'id': 'a', 'source': 'input', 'target': 'memory', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_memory_input'},
                      {'id': 'b', 'source': 'memory', 'target': 'end', 'sourceHandle': 'handle_memory_output', 'targetHandle': 'handle_flow_input'}]}


@pytest.mark.asyncio
async def test_duplicate_nested_ids_use_path_dependencies_and_vector_scope():
    graph = {'nodes': [{'id': 'input', 'type': 'user_input'},
                       {'id': 'left', 'type': 'inner', 'data': {'magic_flow': _child()}},
                       {'id': 'right', 'type': 'inner', 'data': {'magic_flow': _child()}}],
             'edges': [{'id': side, 'source': 'input', 'target': side, 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'} for side in ['left', 'right']]}
    embedding = SimpleNamespace(llm=SimpleNamespace(async_embedding=AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])]))))
    stores = {side: SimpleNamespace(search=AsyncMock(return_value=[])) for side in ['left', 'right']}
    deps = {(side, 'memory'): {'vector_db': stores[side], 'embedding_client': embedding} for side in stores}
    built = build(graph, message='scope-test', deps=deps)
    _ = [event async for event in run_agent(built, id_chat='same-chat')]
    for side in stores:
        stores[side].search.assert_awaited_once()
        assert stores[side].search.call_args.kwargs['filter_scope'] == {'session_id': 'same-chat', 'node_id': f'/{side}/memory'}
        assert built.nodes[side].inner_graph.nodes['memory'].node_id == 'memory'


def test_root_memory_scope_remains_compatible_with_existing_storage():
    built = build(_child(), message='hello')
    assert built.nodes['memory']._scope_node_id == 'memory'
