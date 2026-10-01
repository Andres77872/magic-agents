"""Fetch configuration syntax rejects bad graphs before model or HTTP work."""
from __future__ import annotations

from copy import deepcopy
import json

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from magic_agents.agt_flow import build, run_agent, validate_graph
from magic_agents.node_system.NodeClientLLM import NodeClientLLM
from magic_agents.util.fetch_template_validation import validate_fetch_templates
from magic_llm import MagicLLM
from magic_llm.model.ModelChatResponse import Choice, FunctionCall, Message, ModelChatResponse, ToolCall


ENV_SECRET = "canary_environment_secret_do_not_emit"
SOURCE_SECRET = "canary_literal_secret_do_not_emit"


def edge(identifier, source, target, source_handle, target_handle):
    return {"id": identifier, "source": source, "target": target,
            "sourceHandle": source_handle, "targetHandle": target_handle}


def graph_data(url, header, *, executor="normal"):
    nodes = [{"id": "input", "type": "user_input"},
             {"id": "client", "type": "client", "data": {"engine": "openai", "model": "local-scripted-model"}},
             {"id": "llm", "type": "llm", "data": {"stream": False, "agent_config": {"max_iterations": 2}}},
             {"id": "tavily", "type": "fetch", "data": {"url": url, "method": "GET", "tool_mode": True,
                 "tool_name": "browse", "headers": {"Authorization": header}}},
             {"id": "end", "type": "end"}]
    edges = [edge("client", "client", "llm", "handle-client-provider", "handle-client-provider"),
             edge("tool", "tavily", "llm", "handle_fetch_output", "handle-tool-definition-0")]
    if executor == "loop":
        nodes.append({"id": "loop", "type": "loop"})
        edges.extend([edge("input", "input", "loop", "handle_user_message", "handle_list"),
                      edge("item", "loop", "llm", "handle_item", "handle_user_message"),
                      edge("feedback", "llm", "loop", "handle_generated_content", "handle_loop"),
                      edge("result", "loop", "end", "handle_end", "handle_flow_input")])
    else:
        edges.extend([edge("input", "input", "llm", "handle_user_message", "handle_user_message"),
                      edge("result", "llm", "end", "handle_generated_content", "handle_flow_input")])
    return {"type": "graph", "debug": True, "timeout": 3, "nodes": nodes, "edges": edges}


class LocalEngine:
    engine = engine_name = "openai"
    model = "local-scripted-model"

    def __init__(self):
        self.chats = []
        self.schemas = []

    async def async_generate(self, chat, **kwargs):
        self.chats.append(deepcopy(chat.messages))
        self.schemas.append(deepcopy(kwargs.get("tools", [])))
        if len(self.chats) == 1:
            message = Message(role="assistant", content=None, tool_calls=[ToolCall(
                id="original-fetch-call", function=FunctionCall(name="browse", arguments='{"query":"valid query"}'))])
            finish = "tool_calls"
        else:
            assert len(self.chats) == 2
            tools = [message for message in chat.messages if message.get("role") == "tool"]
            assert len(tools) == 1 and tools[0]["tool_call_id"] == "original-fetch-call"
            content = json.loads(tools[0]["content"])
            if isinstance(content, str):
                content = json.loads(content)
            assert content == {"query": "valid query", "ok": True}
            message = Message(role="assistant", content="done")
            finish = "stop"
        return ModelChatResponse(id="local", object="chat.completion", created=0, model=self.model,
                                 choices=[Choice(index=0, message=message, finish_reason=finish)])


class LocalClient:
    run_agent_async = MagicLLM.run_agent_async
    _task_executor = None

    def __init__(self):
        self.llm = LocalEngine()


def use_local_client(monkeypatch):
    client = LocalClient()

    def initialize(self, engine, model):
        self.client = client
        self.init_error = self.init_error_type = None
        self._current_engine, self._current_model = engine, model

    monkeypatch.setattr(NodeClientLLM, "_initialize_client", initialize)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("executor", ["normal", "loop"])
async def test_actual_malformed_authorization_blocks_graph_before_http_and_model(monkeypatch, executor, caplog):
    monkeypatch.setenv("TAVILY_API_KEY", ENV_SECRET)
    client = use_local_client(monkeypatch)
    requests = []

    async def receive(request):
        requests.append(request.path)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_get("/search", receive)
    async with TestServer(app) as server:
        malformed = "Bearer " + SOURCE_SECRET + " {{env.TAVILY_API_KEY}2}"
        data = graph_data(str(server.make_url("/search")) + "?query={{query}}", malformed, executor=executor)
        # The header is valid JSON; this is specifically template syntax.
        assert json.loads(json.dumps(data))["nodes"][3]["data"]["headers"]["Authorization"] == malformed
        validation = validate_graph(data["nodes"], data["edges"])
        assert not validation["valid"]
        errors = [error for error in validation["errors"] if error["error_type"] == "FetchTemplateValidationError"]
        assert len(errors) == 1
        assert errors[0]["context"] == {"node_id": "tavily", "field": "headers.Authorization", "lineno": 1}
        graph = build(data, message='["first", "second"]' if executor == "loop" else "search")
        events = [event async for event in run_agent(graph)]
    assert requests == [] and client.llm.chats == []
    published = [event["content"] for event in events
                 if event.get("type") == "debug" and event.get("content", {}).get("error_type") == "FetchTemplateValidationError"]
    assert len(published) == 1 and published[0]["context"]["node_id"] == "tavily"
    diagnostic = json.dumps({"validation": validation, "events": events}) + caplog.text
    assert ENV_SECRET not in diagnostic and SOURCE_SECRET not in diagnostic
    assert malformed not in diagnostic and "unexpected '}'" not in diagnostic


@pytest.mark.parametrize("field,value,location", [
    ("url", "https://example.test/?q={{query}2}", "url"),
    ("headers", json.dumps({"Authorization": "Bearer {{env.KEY}2}"}), "headers"),
    ("params", {"q": "{{query}2}"}, "params.q"),
    ("data", {"items": ["{{query}}", {"token": "{{env.KEY}2}"}]}, "data.items[1].token"),
    ("json_data", {"credentials": {"token": "{{env.KEY}2}"}}, "json_data.credentials.token"),
    ("tool_parameters", {"token": "{{env.KEY}2}"}, "tool_parameters.token"),
])
def test_request_templates_are_checked_recursively_with_field_location(field, value, location):
    node = {"id": "fetch", "type": "fetch", "data": {field: value}}
    original = deepcopy(node)
    errors = validate_fetch_templates([node])
    assert len(errors) == 1
    assert errors[0]["error_type"] == "FetchTemplateValidationError"
    assert errors[0]["context"]["node_id"] == "fetch"
    assert errors[0]["context"]["field"] == location
    assert node == original


def test_valid_placeholders_remain_authored_and_schema_descriptions_are_not_request_templates(monkeypatch):
    monkeypatch.setenv("KEY", ENV_SECRET)
    nodes = [{"id": "fetch", "type": "fetch", "data": {
        "url": "https://example.test/?q={{query}}", "headers": {"Authorization": "Bearer {{env.KEY}}"},
        "tool_parameters": {"query": {"type": "string", "description": "Literal docs: {{not a template"}},
    }}]
    original = deepcopy(nodes)
    assert validate_fetch_templates(nodes) == []
    assert nodes == original and ENV_SECRET not in json.dumps(nodes)


@pytest.mark.asyncio
async def test_valid_env_header_and_query_work_in_real_http_tool_loop(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", ENV_SECRET)
    client = use_local_client(monkeypatch)
    requests = []

    async def receive(request):
        requests.append({"authorization": request.headers.get("Authorization"), "query": request.query.get("query")})
        return web.json_response({"ok": True, "query": request.query.get("query")})

    app = web.Application()
    app.router.add_get("/search", receive)
    async with TestServer(app) as server:
        data = graph_data(str(server.make_url("/search")) + "?query={{query}}", "Bearer {{env.TAVILY_API_KEY}}")
        authored_fetch = deepcopy(data["nodes"][3]["data"])
        assert validate_graph(data["nodes"], data["edges"])["valid"]
        graph = build(data, message="search")
        events = [event async for event in run_agent(graph)]
    assert requests == [{"authorization": "Bearer " + ENV_SECRET, "query": "valid query"}]
    assert data["nodes"][3]["data"] == authored_fetch
    assert graph.nodes["end"].inputs["handle_flow_input"] == "done"
    parameters = next(tool["function"]["parameters"] for tool in client.llm.schemas[0]
                      if tool["function"]["name"] == "browse")
    assert set(parameters["properties"]) == {"query"}
    assert ENV_SECRET not in json.dumps(client.llm.chats)
    results = [event for event in events if event.get("type") == "debug"
               and event.get("content", {}).get("event_type") == "TOOL_RESULT"]
    assert len(results) == 1 and results[0]["content"]["data"]["tool_call_id"] == "original-fetch-call"
