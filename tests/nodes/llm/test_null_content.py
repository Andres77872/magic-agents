"""A provider reply with ``content: null`` must not crash NodeLLM."""
import pytest
from magic_llm.model.ModelChatResponse import ModelChatResponse

from magic_agents import run_agent
from magic_agents.agt_flow import build


def graph(json_output):
    return {
        # A JSON failure leaves the node silent (an onFinish Hook may repair it),
        # so End waits for its input until the graph timeout.
        'timeout': 1,
        'nodes': [
            {'id': 'input', 'type': 'user_input', 'data': {}},
            {'id': 'client', 'type': 'client', 'data': {'engine': 'openai', 'model': 'fake',
                                                        'api_info': {'api_key': 'test-only'}}},
            {'id': 'llm', 'type': 'llm', 'data': {'stream': False, 'json_output': json_output}},
            {'id': 'end', 'type': 'end'},
        ],
        'edges': [
            {'id': 'e1', 'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
            {'id': 'e2', 'source': 'client', 'target': 'llm', 'sourceHandle': 'handle-client-provider', 'targetHandle': 'handle-client-provider'},
            {'id': 'e3', 'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('json_output', [False, True])
async def test_null_content_becomes_empty_text(monkeypatch, json_output):
    import magic_llm.engine.engine_openai as engine_openai

    class FakeHTTP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post_json(self, *, data, **kwargs):
            return ModelChatResponse(id='r', object='chat.completion', created=0, model='fake', choices=[{
                'index': 0, 'message': {'role': 'assistant', 'content': None}, 'finish_reason': 'stop'}],
            ).model_dump(exclude_none=True)

    monkeypatch.setattr(engine_openai, 'AsyncHttpClient', FakeHTTP)
    built = build(graph(json_output), message='hello')
    events = [event async for event in run_agent(built)]
    error_types = [e['content'].get('error_type') for e in events
                   if e.get('type') == 'debug' and isinstance(e.get('content'), dict)]
    assert 'TypeError' not in error_types
    assert built.nodes['llm'].generated == ''
    if json_output:
        assert 'JSONExtractionError' in error_types
    else:
        assert built.nodes['llm'].outputs['handle_generated_content']['content'] == ''
