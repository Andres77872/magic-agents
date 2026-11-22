"""The tracked ``browsing.json`` graph, executed without credentials.

Its real parser, conditional and fetch templates run end to end. Only the two
external boundaries are faked: the LLM provider (a scripted client patched at
``NodeClientLLM.MagicLLM``) and the Jina search endpoint (a local aiohttp server
the fetch nodes are pointed at).
"""

import copy
import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

from magic_agents import run_agent
from magic_agents.agt_flow import build, validate_graph
from magic_agents.hooks import RuntimeConfig
from tests.support import RecordingHook, debug_errors, load_fixture_graph

FETCH_IDS = {"fetch", "39431f32-00c4-47a9-b915-b8f6fa907b34", "5b581c5d-3551-464b-84bf-263fcc699dc9"}


class ScriptedEngine:
    """Answers the JSON query-expansion call and streams every other call."""

    model = "fake"

    def __init__(self, queries):
        self.queries = queries
        self.streamed_prompts = []

    async def async_generate(self, chat, **kwargs):
        return type("Response", (), {"content": json.dumps({"queries": self.queries}), "usage": None})()

    async def async_stream_generate(self, chat, **kwargs):
        self.streamed_prompts.append(json.dumps(chat.messages))
        choice = ChoiceModel(index=0, delta=DeltaModel(content="final answer"), logprobs=None, finish_reason=None)
        yield ChatCompletionModel(id="chunk", model="fake", choices=[choice])


@pytest.fixture
def browsing():
    return load_fixture_graph("browsing.json")


def _scripted_client(queries):
    engine = ScriptedEngine(queries)
    client = type("Client", (), {"llm": engine, "model": "fake"})()
    return engine, patch("magic_agents.node_system.NodeClientLLM.MagicLLM", return_value=client)


async def _run(graph):
    """Run the graph and return the ids of the nodes that completed."""
    hook = RecordingHook()
    events = [event async for event in run_agent(graph, hooks=RuntimeConfig(graph_hooks=[hook]))]
    assert debug_errors(events) == []
    return {inv["context"].node_id for inv in hook.get_invocations("on_node_end")}


def test_fixture_declares_conditional_search_routing(browsing):
    nodes = {node["id"]: node for node in browsing["nodes"]}
    conditional = nodes["conditional-queries-check"]["data"]

    assert validate_graph(browsing["nodes"], browsing["edges"])["valid"]
    assert conditional["output_handles"] == ["no_queries", "has_queries"]
    assert conditional["default_handle"] == "no_queries"
    assert {n for n, node in nodes.items() if node["type"] == "fetch"} == FETCH_IDS
    routes = {(e["sourceHandle"], e["target"]) for e in browsing["edges"] if e["source"] == "conditional-queries-check"}
    assert ("no_queries", "text-no-search-response") in routes
    assert ("has_queries", "parser-browsing-rewrite") in routes


@pytest.mark.parametrize("message", ["", "Hi there!", "x" * 10_000, "<script>{{ 7*7 }}</script> & \"'", "こんにちは 🌍"])
def test_fixture_builds_without_error_diagnostics_for_any_message(browsing, message):
    graph = build(browsing, message=message, load_chat=None)

    errors = [e for e in (graph._validation_errors or []) if e.get("severity") != "warning"]
    assert errors == []
    assert graph.nodes["user_input"]._text == message


async def test_greeting_without_queries_skips_search_and_sends_the_canned_reply(browsing):
    engine, patched = _scripted_client(queries=[])
    with patched:
        graph = build(browsing, message="Hi there!", load_chat=None)
        executed = await _run(graph)

    assert {"conditional-queries-check", "text-no-search-response", "send-message-no-search"} <= executed
    assert not (FETCH_IDS & executed)
    assert graph.nodes["conditional-queries-check"].selected_handle == "no_queries"


async def test_search_path_sends_each_query_and_dedups_results_into_the_final_prompt(browsing, monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "jina-test-key")
    received = []

    async def search(request):
        received.append((request.headers.get("Authorization"), (await request.json())["q"]))
        shared = {"title": "Shared", "url": "https://shared.example", "content": "shared body",
                  "description": "shared snippet", "images": {}}
        own = {"title": f"About {received[-1][1]}", "url": f"https://{len(received)}.example",
               "content": "own body", "description": "own snippet", "images": {}}
        return web.json_response({"data": [shared, own]})

    app = web.Application()
    app.router.add_post("/", search)
    async with TestServer(app) as server:
        graph_def = copy.deepcopy(browsing)
        for node in graph_def["nodes"]:
            if node["id"] in FETCH_IDS:
                node["data"]["url"] = str(server.make_url("/"))
        engine, patched = _scripted_client(queries=["tokyo population", "tokyo census", "tokyo 2024"])
        with patched:
            graph = build(graph_def, message="How many people live in Tokyo?", load_chat=None)
            executed = await _run(graph)

    assert sorted(q for _, q in received) == ["tokyo 2024", "tokyo census", "tokyo population"]
    assert {auth for auth, _ in received} == {"Bearer jina-test-key"}
    assert FETCH_IDS <= executed
    references = json.loads(graph.nodes["send-message"].inputs["handle_send_extra"])["results_ref"]
    assert [ref["link"] for ref in references].count("https://shared.example") == 1
    assert len(references) == 4
    final_prompt = next(p for p in engine.streamed_prompts if "<search_results>" in p)
    assert final_prompt.count("https://shared.example") == 1
    assert "How many people live in Tokyo?" in final_prompt
