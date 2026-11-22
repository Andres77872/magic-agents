"""LLM nodes must not mutate the ModelChat a Chat node shares with them."""
import json

import pytest
from magic_llm.model.ModelChatResponse import ModelChatResponse

from magic_agents import run_agent
from magic_agents.agt_flow import build


def graph():
    def edge(source, target, source_handle, target_handle):
        return {'id': f'{source}-{target}-{target_handle}', 'source': source, 'target': target,
                'sourceHandle': source_handle, 'targetHandle': target_handle}
    return {
        'nodes': [
            {'id': 'input', 'type': 'user_input', 'data': {}},
            {'id': 'client', 'type': 'client', 'data': {'engine': 'openai', 'model': 'fake',
                                                        'api_info': {'api_key': 'test-only'}}},
            {'id': 'chat', 'type': 'chat', 'data': {}},
            {'id': 'prompt-a', 'type': 'text', 'data': {'text': 'PROMPT_A'}},
            {'id': 'prompt-b', 'type': 'text', 'data': {'text': 'PROMPT_B'}},
            {'id': 'system-a', 'type': 'text', 'data': {'text': 'SYSTEM_A'}},
            {'id': 'llm-a', 'type': 'llm', 'data': {'stream': False}},
            {'id': 'llm-b', 'type': 'llm', 'data': {'stream': False}},
            {'id': 'end', 'type': 'end'},
            {'id': 'sink', 'type': 'void'},
        ],
        'edges': [
            edge('input', 'chat', 'handle_user_message', 'handle_user_message'),
            edge('chat', 'llm-a', 'handle_chat_output', 'handle-chat'),
            edge('chat', 'llm-b', 'handle_chat_output', 'handle-chat'),
            edge('prompt-a', 'llm-a', 'handle_text_output', 'handle_user_message'),
            edge('system-a', 'llm-a', 'handle_text_output', 'handle-system-context'),
            edge('prompt-b', 'llm-b', 'handle_text_output', 'handle_user_message'),
            edge('client', 'llm-a', 'handle-client-provider', 'handle-client-provider'),
            edge('client', 'llm-b', 'handle-client-provider', 'handle-client-provider'),
            edge('llm-a', 'end', 'handle_generated_content', 'handle_flow_input'),
            edge('llm-b', 'sink', 'handle_generated_content', 'handle-void'),
        ],
    }


@pytest.mark.asyncio
async def test_consumers_of_one_chat_do_not_see_each_others_prompts(monkeypatch):
    import magic_llm.engine.engine_openai as engine_openai
    payloads = []

    class FakeHTTP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post_json(self, *, data, **kwargs):
            payloads.append(json.loads(data)['messages'])
            return ModelChatResponse(id='r', object='chat.completion', created=0, model='fake', choices=[{
                'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
            ).model_dump(exclude_none=True)

    monkeypatch.setattr(engine_openai, 'AsyncHttpClient', FakeHTTP)
    built = build(graph(), message='HELLO')
    [event async for event in run_agent(built)]

    assert len(payloads) == 2
    by_prompt = {m[-1]['content']: m for m in payloads}
    assert set(by_prompt) == {'PROMPT_A', 'PROMPT_B'}
    rendered_b = json.dumps(by_prompt['PROMPT_B'])
    assert 'PROMPT_A' not in rendered_b and 'SYSTEM_A' not in rendered_b
    assert 'PROMPT_B' not in json.dumps(by_prompt['PROMPT_A'])
    assert built.nodes['chat'].chat.messages == [{'role': 'user', 'content': 'HELLO'}]
