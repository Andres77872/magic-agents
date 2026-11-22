"""Live-provider smoke tests: real OpenAI (and optionally Serper/Jina) calls.

Deselected by default; run with ``python -m pytest tests/live -m needs_api``.
They prove the authored graphs work against the real provider wire format;
deterministic behaviour is covered by the credential-free suite.
"""

import json
import os

import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from tests.live import LIVE
from tests.support import _is_placeholder_value, debug_errors, load_fixture_graph

pytestmark = LIVE


def _client(keys):
    return {"id": "client", "type": "client", "data": {
        "model": "gpt-4o-mini", "engine": "openai",
        "api_info": {"api_key": keys["openai_key"], "base_url": "https://api.openai.com/v1"},
    }}


def _streamed_text(events):
    text = ""
    for event in events:
        content = event.get("content") if event.get("type") == "content" else None
        if getattr(content, "choices", None) and content.choices[0].delta.content:
            text += content.choices[0].delta.content
    return text


async def _run(graph):
    return [event async for event in run_agent(graph)]


async def test_single_llm_graph_streams_an_answer(api_keys):
    graph = build({
        "type": "graph",
        "nodes": [
            {"id": "user_input", "type": "user_input"},
            {"id": "llm", "type": "llm", "data": {"stream": True, "max_tokens": 64, "temperature": 0}},
            _client(api_keys),
            {"id": "finish", "type": "end"},
        ],
        "edges": [
            {"id": "e1", "source": "user_input", "target": "llm",
             "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"},
            {"id": "e2", "source": "client", "target": "llm",
             "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
            {"id": "e3", "source": "llm", "target": "finish",
             "sourceHandle": "handle_generated_content", "targetHandle": "handle_generated_content"},
        ],
    }, message="Reply with the single word: pong", load_chat=None)

    events = await _run(graph)

    assert debug_errors(events) == []
    assert "pong" in _streamed_text(events).lower()


async def test_loop_runs_the_llm_once_per_item_and_aggregates(api_keys):
    graph = build({
        "type": "graph",
        "nodes": [
            {"id": "input", "type": "user_input"},
            {"id": "items", "type": "text", "data": {"text": '["apple", "banana"]'}},
            {"id": "loop", "type": "loop", "data": {}},
            _client(api_keys),
            {"id": "describe", "type": "llm", "data": {"stream": True, "iterate": True, "max_tokens": 20, "temperature": 0}},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "describe",
             "sourceHandle": "handle_user_message", "targetHandle": "handle-system-context"},
            {"id": "e2", "source": "client", "target": "describe",
             "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
            {"id": "e3", "source": "items", "target": "loop",
             "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
            {"id": "e4", "source": "loop", "target": "describe",
             "sourceHandle": "handle_item", "targetHandle": "handle_user_message"},
            {"id": "e5", "source": "describe", "target": "loop",
             "sourceHandle": "handle_generated_content", "targetHandle": "handle_loop"},
            {"id": "e6", "source": "loop", "target": "end", "sourceHandle": "handle_end", "targetHandle": "h1"},
        ],
    }, message="Describe the given fruit in one word.", load_chat=None)

    events = await _run(graph)

    assert debug_errors(events) == []
    aggregated = graph.nodes["end"].inputs["h1"]
    if isinstance(aggregated, str):
        aggregated = json.loads(aggregated)
    assert len(aggregated) == 2


async def test_image_graph_system_prompt_steers_the_reply_language(api_keys):
    graph = build(load_fixture_graph("image.json"), message="What is the capital of France?", load_chat=None)

    events = await _run(graph)

    assert debug_errors(events) == []
    assert "paris" in _streamed_text(events).lower()


async def test_browsing_graph_searches_and_answers(api_keys):
    serper_key = os.environ.get("SERPER_API_KEY", "")
    jina_key = os.environ.get("JINA_API_KEY", "")
    if _is_placeholder_value(serper_key) or _is_placeholder_value(jina_key):
        pytest.skip("browsing search needs real SERPER_API_KEY and JINA_API_KEY")
    graph = build(load_fixture_graph("browsing.json"), message="What is the population of Tokyo?", load_chat=None)

    events = await _run(graph)

    assert debug_errors(events) == []
    assert _streamed_text(events).strip()
