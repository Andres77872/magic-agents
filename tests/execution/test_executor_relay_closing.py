"""Closing a run closes the nested executor generators it relays, in-task.

``async for`` never closes its iterator on early exit; without ``aclosing`` the
inner generator is finalized later by the GC hook in another task, after the
outer run has already ended.
"""
import pytest

from magic_agents.agt_flow import build
from magic_agents.execution import reactive_executor


def loop_graph():
    edge = lambda source, target, sh, th: {'id': f'{source}-{target}-{th}', 'source': source, 'target': target,
                                           'sourceHandle': sh, 'targetHandle': th}
    return {'type': 'graph', 'debug': False, 'timeout': 2, 'nodes': [
        {'id': 'input', 'type': 'user_input'},
        {'id': 'loop', 'type': 'loop'},
        {'id': 'target', 'type': 'parser', 'data': {'text': '{{ handle_parser_input }}'}},
        {'id': 'end', 'type': 'end'},
    ], 'edges': [
        edge('input', 'loop', 'handle_user_message', 'handle_list'),
        edge('loop', 'target', 'handle_item', 'handle_parser_input'),
        edge('target', 'loop', 'handle_parser_output', 'handle_loop'),
        edge('loop', 'end', 'handle_end', 'handle_flow_input'),
    ]}


@pytest.mark.asyncio
async def test_closing_a_loop_graph_run_closes_the_loop_executor_first(monkeypatch):
    closed = []

    async def loop_executor(graph, **kwargs):
        try:
            yield {'type': 'content', 'content': 'first'}
            yield {'type': 'content', 'content': 'second'}
        finally:
            closed.append('loop executor')

    monkeypatch.setattr(reactive_executor, 'execute_graph_loop_reactive', loop_executor)
    run = reactive_executor.execute_graph_reactive(build(loop_graph(), message='["a", "b"]'))
    async for event in run:
        if event.get('content') == 'first':
            break
    await run.aclose()
    assert closed == ['loop executor']
