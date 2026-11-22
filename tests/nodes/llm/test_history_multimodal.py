"""History content parts reach the provider intact when no Chat node is present."""
import json

import pytest
from magic_llm.model.ModelChatResponse import ModelChatResponse

from magic_agents import run_agent
from magic_agents.agt_flow import build

IMAGE = 'data:image/png;base64,iVBORw0KGgo='


def graph():
    return {
        'nodes': [
            {'id': 'input', 'type': 'user_input', 'data': {}},
            {'id': 'client', 'type': 'client', 'data': {'engine': 'openai', 'model': 'fake',
                                                        'api_info': {'api_key': 'test-only'}}},
            {'id': 'llm', 'type': 'llm', 'data': {'stream': False}},
            {'id': 'end', 'type': 'end'},
        ],
        'edges': [
            {'id': 'e1', 'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
            {'id': 'e2', 'source': 'client', 'target': 'llm', 'sourceHandle': 'handle-client-provider', 'targetHandle': 'handle-client-provider'},
            {'id': 'e3', 'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'},
        ],
    }


@pytest.mark.asyncio
async def test_multimodal_history_is_not_flattened(monkeypatch):
    import magic_llm.engine.engine_openai as engine_openai
    sent = []

    class FakeHTTP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post_json(self, *, data, **kwargs):
            sent.append(json.loads(data)['messages'])
            return ModelChatResponse(id='r', object='chat.completion', created=0, model='fake', choices=[{
                'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
            ).model_dump(exclude_none=True)

    monkeypatch.setattr(engine_openai, 'AsyncHttpClient', FakeHTTP)
    parts = [{'type': 'text', 'text': 'what is this?'}, {'type': 'image_url', 'image_url': {'url': IMAGE}}]
    history = [{'role': 'user', 'content': parts}, {'role': 'assistant', 'content': 'a logo'}]
    built = build(graph(), message='and now?', history_messages=history)
    [event async for event in run_agent(built)]
    assert len(sent) == 1
    assert sent[0][0] == {'role': 'user', 'content': parts}
    assert sent[0][-1] == {'role': 'user', 'content': 'and now?'}
    assert history[0]['content'] == parts
