"""Nested streams keep child provenance and usage-only terminal chunks."""
from types import MethodType

import pytest
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel
from magic_agents.agt_flow import build, run_agent


def _wrap(child, node_id='inner'):
    return {'nodes': [{'id': 'input', 'type': 'user_input'}, {'id': node_id, 'type': 'inner', 'data': {'magic_flow': child}}, {'id': 'end', 'type': 'end'}],
            'edges': [{'id': 'a', 'source': 'input', 'target': node_id, 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
                      {'id': 'b', 'source': node_id, 'target': 'end', 'sourceHandle': 'handle_execution_content', 'targetHandle': 'handle_flow_input'}]}


def _child():
    return {'nodes': [{'id': 'input', 'type': 'user_input'}, {'id': 'send', 'type': 'send_message', 'data': {'message': 'VISIBLE'}}, {'id': 'end', 'type': 'end'}],
            'edges': [{'id': 'a', 'source': 'input', 'target': 'send', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_send_extra'},
                      {'id': 'b', 'source': 'send', 'target': 'end', 'sourceHandle': 'handle_message_output', 'targetHandle': 'handle_flow_input'}]}


@pytest.mark.asyncio
async def test_two_inner_levels_preserve_leaf_origin():
    graph = build(_wrap(_wrap(_child(), 'inside'), 'outside'), message='hello')
    events = [e async for e in run_agent(graph) if e.get('type') == 'content']
    message = next(e for e in events if e['content'].choices[0].delta.content == 'VISIBLE')
    assert message['source_node'] == 'outside'
    assert message['source_node_path'] == ['outside', 'inside', 'send']


@pytest.mark.asyncio
async def test_inner_forwards_final_chunk_even_when_it_has_no_text():
    graph = build(_wrap(_child()), message='hello')
    terminal = ChatCompletionModel(id='provider-id', model='test', choices=[ChoiceModel(delta=DeltaModel(content=None), finish_reason='stop')],
                                   usage={'prompt_tokens': 7, 'completion_tokens': 3, 'total_tokens': 10})
    async def stream(self, chat_log):
        yield self.yield_static(terminal, content_type=self.OUTPUT_HANDLE_CONTENT)
        yield self.yield_static('done', content_type=self.OUTPUT_HANDLE)
    leaf = graph.nodes['inner'].inner_graph.nodes['send']
    leaf.process = MethodType(stream, leaf)
    events = [e async for e in run_agent(graph) if e.get('type') == 'content']
    chunk = next(e for e in events if e['content'].id == 'provider-id')
    assert chunk['content'].usage.total_tokens == 10
    assert chunk['source_node_path'] == ['inner', 'send']



def _loop_child():
    return {
        "nodes": [
            {"id": "input", "type": "user_input"},
            {"id": "items", "type": "text", "data": {"text": '["one", "two"]'}},
            {"id": "loop", "type": "loop"},
            {"id": "parse", "type": "parser", "data": {"text": "{{ handle_parser_input }}"}},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"id": "a", "source": "input", "target": "items", "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
            {"id": "b", "source": "items", "target": "loop", "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
            {"id": "c", "source": "loop", "target": "parse", "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
            {"id": "d", "source": "parse", "target": "loop", "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
            {"id": "e", "source": "loop", "target": "end", "sourceHandle": "handle_end", "targetHandle": "handle_flow_input"},
        ],
    }


@pytest.mark.asyncio
async def test_real_nested_graph_keeps_debug_and_loop_progress_ancestry():
    import asyncio
    from magic_agents.hooks.runtime_config import RuntimeConfig

    graph = build(_wrap(_wrap(_loop_child(), "inside"), "outside"), message="hello")
    sink = asyncio.Queue()
    config = RuntimeConfig()
    config.enable_debug_sse(sink)
    events = [event async for event in run_agent(graph, hooks=config)]
    progress = [event for event in events if event.get("type") == "loop_progress"]
    assert [event["content"]["current"] for event in progress] == [0, 1]
    assert all(event["source_node_path"] == ["outside", "inside", "loop"] for event in progress)
    iterations = [event for event in events if isinstance(event.get("content"), dict)
                  and event["content"].get("event_type") == "ITERATION_START"]
    assert len(iterations) == 2
    assert all(event["source_node_path"] == ["outside", "inside", "loop"] for event in iterations)
    hook_events = []
    while not sink.empty():
        hook_events.append(sink.get_nowait())
    inputs = [event for event in hook_events if event["event_type"] == "node_start" and event["content"]["node_id"] == "input"]
    assert sorted(event["source_node_path"] for event in inputs) == sorted([
        ["input"], ["outside", "input"], ["outside", "inside", "input"],
    ])
    graph_starts = [event["source_node_path"] for event in hook_events if event["event_type"] == "graph_start"]
    assert graph_starts == [[], ["outside"], ["outside", "inside"]]
    # Provenance does not alter the reducer/data route or duplicate iteration outputs.
    assert graph.nodes["outside"].outputs["handle_execution_content"]["content"] == '["one", "two"]'
