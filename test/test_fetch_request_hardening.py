"""Fetch request rendering, data/template boundary and typed failures.

Every test runs the real runtime (build/run_agent, NodeFetch, InvocationControl)
against a loopback aiohttp server; no external service or credential is used.

Covers:
- one renderer: plain step and Hook-controlled step Fetch send the same
  request and produce the same output for hostile input text;
- values from input handles, Hook content and tool arguments are data: never
  compiled as Jinja templates and never env-resolved;
- step-mode failures raise HTTPError/NetworkError/TemplateError/UnexpectedError
  at once (downstream bypassed, no input timeout) with sanitized diagnostics.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import copy
import json
import logging
import os
import socket
import time

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from magic_agents.agt_flow import build, run_agent
from magic_agents.hooks.debug_sse import DebugSSEHook
from magic_agents.hooks.runtime_config import RuntimeConfig
from magic_agents.models.factory.Nodes import FetchNodeModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.NodeFetch import FetchToolCallable, NodeFetch
from magic_agents.node_system.fetch_request import (
    FETCH_TEMPLATE_ENV,
    FetchRenderer,
    FetchSecrets,
    HTTPError,
    NetworkError,
    TemplateError,
    UnexpectedError,
    safe_url,
)


pytestmark = pytest.mark.asyncio

CANARY_NAME = "FETCH_HARDENING_CANARY"
CANARY = "canary-fetch-secret-7f3a91c2"
SSTI = "{{ cycler.__init__.__globals__.os.getpid() }}"
PID = str(os.getpid())

HOSTILE_INPUTS = [
    "plain words",
    'say "hi"',
    "line1\nline2\n",
    "back\\slash \\n \\u0041",
    '{"a": 1} {{ not_a_var }} }}{{',
    "<b>&amp; 'html'</b>",
    'x", "admin": true, "y": "',
    "{{env." + CANARY_NAME + "}}",
    "{% raw %}{{ 7*7 }}{% endraw %}",
    SSTI,
]


@pytest.fixture(autouse=True)
def canary_env(monkeypatch):
    monkeypatch.setenv(CANARY_NAME, CANARY)


@asynccontextmanager
async def http_server():
    requests = []

    async def record(request):
        body = await request.text()
        item = {"method": request.method, "path": request.path, "query": dict(request.query),
                "headers": {key: value for key, value in request.headers.items() if key.lower().startswith("x-")},
                "body": body}
        requests.append(item)
        return item

    async def echo(request):
        item = await record(request)
        try:
            parsed = json.loads(item["body"]) if item["body"] else None
        except ValueError:
            parsed = None
        return web.json_response({"query": item["query"], "headers": item["headers"], "json": parsed,
                                  "body": item["body"]})

    async def status(request):
        item = await record(request)
        # The error body echoes what it received, secrets included, so tests
        # can prove diagnostics scrub them.
        return web.json_response({"error": "boom", "query": item["query"], "headers": item["headers"]},
                                 status=int(request.match_info["code"]))

    async def bad_json(request):
        await record(request)
        return web.Response(text="{not json", content_type="application/json")

    async def raw_status(request):
        # Echo the raw (still percent-encoded) path and query plus the raw
        # body, like a server error page that quotes the request.
        await record(request)
        return web.Response(text=request.raw_path + "\n" + (await request.text()),
                            status=int(request.match_info["code"]), content_type="text/plain")

    async def plain_text(request):
        await record(request)
        return web.Response(text="123", content_type="text/plain")

    async def flaky(request):
        await record(request)
        item = request.match_info["item"]
        if item == "a":
            return web.json_response({"name": "repo-" + item})
        return web.json_response({"error": "boom"}, status=500)

    app = web.Application()
    app.router.add_route("*", "/echo", echo)
    app.router.add_route("*", "/status/{code}", status)
    app.router.add_route("*", "/raw-status/{code}/{tail:.*}", raw_status)
    app.router.add_get("/bad-json", bad_json)
    app.router.add_get("/text", plain_text)
    app.router.add_get("/flaky/{item}", flaky)
    async with TestServer(app) as server:
        yield str(server.make_url("")).rstrip("/"), requests


def closed_port_url():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


def edge(source, source_handle, target, target_handle):
    return {"id": f"{source}.{source_handle}->{target}.{target_handle}", "source": source,
            "sourceHandle": source_handle, "target": target, "targetHandle": target_handle}


PASS_HOOK = "def control(context, chat_log):\n    return {'action': 'pass'}\n"
DEGRADE_HOOK = (
    "def degrade(context, chat_log):\n"
    "    if context.outcome['status'] != 'error':\n"
    "        return None\n"
    "    error = context.outcome['error']\n"
    "    return {'action': 'outcome', 'outcome': {'status': 'success', 'content': "
    "{'handle_fetch_output': {'name': 'fallback', 'code': error['code'], 'message': error['message']}}}}\n"
)
RECORD_ERROR_HOOK = (
    "def record(context, chat_log):\n"
    "    if context.outcome['status'] == 'error':\n"
    "        context.emit.debug({'seen_error': context.outcome['error']})\n"
    "    return None\n"
)


def fetch_graph(fetch_data, *, handle="handle_fetch_input", control=None, hook_code=PASS_HOOK,
                parser=None, timeout=5):
    """User input -> Fetch [-> Parser] -> End, optionally Hook-controlled."""
    nodes = [{"id": "input", "type": "user_input"},
             {"id": "fetch", "type": "fetch", "data": fetch_data},
             {"id": "end", "type": "end"}]
    edges = [edge("input", "handle_user_message", "fetch", handle)]
    if parser is None:
        edges.append(edge("fetch", "handle_fetch_output", "end", "handle_flow_input"))
    else:
        nodes.append({"id": "parser", "type": "parser", "data": {"text": parser}})
        edges += [edge("fetch", "handle_fetch_output", "parser", "response"),
                  edge("parser", "handle_parser_output", "end", "handle_flow_input")]
    if control:
        nodes.append({"id": "control", "type": "hook", "data": {
            "lifecycle_event": control, "target_node_id": "fetch", "failure_policy": "preserve",
            "function_template": hook_code}})
    return {"type": "graph", "debug": True, "timeout": timeout, "nodes": nodes, "edges": edges}


async def run(data, message, *, hooks=None):
    graph = build(copy.deepcopy(data), message=message)
    started = time.monotonic()
    events = [event async for event in run_agent(graph, hooks=hooks)]
    return graph, events, time.monotonic() - started


def error_frames(events):
    return [event["content"] for event in events if event.get("type") == "debug"
            and isinstance(event.get("content"), dict) and event["content"].get("error_type")]


def hook_results(events):
    return [event["content"]["data"]["execution"] for event in events if event.get("type") == "debug"
            and isinstance(event.get("content"), dict) and event["content"].get("event_type") == "HOOK_RESULT"]


def output(graph, node_id, handle):
    value = graph.nodes[node_id].outputs.get(handle)
    if isinstance(value, dict) and "content" in value and "node" in value:
        return value["content"]
    return value


class Sink:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


# ---------------------------------------------------------------------------
# One renderer for plain step and Hook-controlled step mode
# ---------------------------------------------------------------------------


def parity_config(base):
    return {
        "method": "POST",
        "url": base + "/echo?u={{ handle_fetch_input | urlencode }}",
        "headers": {"X-Static": "static", "X-Key": "{{env." + CANARY_NAME + "}}"},
        "params": {"p": "{{ handle_fetch_input }}", "n": 3},
        "json_data": {
            "q": "{{ handle_fetch_input }}",
            "legacy": "{{ (handle_fetch_input | tojson)[1:-1] }}",
            "nested": {"inner": "pre {{ handle_fetch_input }} post", "list": ["{{ handle_fetch_input }}", 7, None]},
            "n": 5,
            "flag": True,
            "literal": 'say "{% raw %}{{ handle_fetch_input }}{% endraw %}"\nnext line',
            "env": "Bearer {{env." + CANARY_NAME + "}}",
        },
    }


def expected_json(value):
    return {
        "q": value, "legacy": value,
        "nested": {"inner": f"pre {value} post", "list": [value, 7, None]},
        "n": 5, "flag": True,
        "literal": 'say "{{ handle_fetch_input }}"\nnext line',
        "env": "Bearer " + CANARY,
    }


@pytest.mark.parametrize("value", HOSTILE_INPUTS)
async def test_step_and_hook_controlled_step_send_identical_requests_and_outputs(value):
    results = {}
    async with http_server() as (base, requests):
        for mode in ("step", "controlled"):
            data = fetch_graph(parity_config(base), control="onStart" if mode == "controlled" else None)
            graph, events, _ = await run(data, value)
            assert error_frames(events) == []
            assert len(requests) == 1
            results[mode] = (requests.pop(), output(graph, "fetch", "handle_fetch_output"))
            if mode == "controlled":
                assert [frame["node_id"] for frame in hook_results(events)] == ["fetch"]
    (step_request, step_output), (hook_request, hook_output) = results["step"], results["controlled"]
    assert step_request == hook_request
    assert step_output == hook_output
    # Exact values for every leaf: no escaping left behind, no lost newline,
    # no extra key from JSON-looking text, inputs never compiled.
    assert json.loads(step_request["body"]) == expected_json(value)
    assert step_request["query"] == {"u": value, "p": value, "n": "3"}
    assert step_request["headers"] == {"X-Static": "static", "X-Key": CANARY}
    assert PID not in step_request["body"] or PID in value
    assert step_request["body"].count(CANARY) == 1  # only the authored env placeholder


async def test_response_parsing_is_the_same_for_plain_and_hook_controlled_step():
    async with http_server() as (base, requests):
        for config in ({"url": base + "/text", "method": "GET"},
                       {"url": base + "/echo", "method": "POST"}):
            values = []
            for control in (None, "onStart"):
                graph, events, _ = await run(fetch_graph(config, control=control), "go")
                assert error_frames(events) == []
                values.append(output(graph, "fetch", "handle_fetch_output"))
            assert values[0] == values[1]
            # text/plain stays text; a POST without a body sends nothing.
            assert values[0] == ("123" if config["url"].endswith("/text") else {})
    assert [item["path"] for item in requests] == ["/text", "/text"]


async def test_headers_are_templated_in_both_modes_and_line_breaks_are_rejected():
    async with http_server() as (base, requests):
        config = {"url": base + "/echo", "method": "GET", "headers": {"X-Echo": "v={{ handle_fetch_input }}"}}
        for control in (None, "onStart"):
            graph, events, _ = await run(fetch_graph(config, control=control), "abc")
            assert error_frames(events) == []
        assert [item["headers"]["X-Echo"] for item in requests] == ["v=abc", "v=abc"]
        requests.clear()

        graph, events, elapsed = await run(fetch_graph(config), "abc\r\nX-Injected: 1")
        assert requests == [] and elapsed < 2
        frames = error_frames(events)
        assert [frame["error_type"] for frame in frames] == ["TemplateError"]
        assert "line break" in frames[0]["error_message"]

        graph, events, _ = await run(fetch_graph(config, control="onStart"), "abc\nX-Injected: 1")
        assert requests == []
        record = hook_results(events)[0]
        assert record["outcome"]["error"]["code"] == "NODE_EXCEPTION"
        assert record["outcome"]["error"]["details"] == {
            "exception_type": "TemplateError", "context": {"field": "headers", "method": "GET"}}


@pytest.mark.parametrize("json_data,value,expected", [
    # Valid JSON text before rendering: parsed, then rendered per leaf.
    ('{"q": "{{ handle_fetch_input }}", "n": 1}', 'say "hi"', {"q": 'say "hi"', "n": 1}),
    # Not JSON before rendering: rendered as a whole, expression results escaped.
    ('{"n": {{ handle_fetch_input }}, "s": "{{ handle_fetch_input }}"}', "42", {"n": 42, "s": "42"}),
    ('{"o": {{ handle_fetch_input | fromjson | tojson }}}', '{"k": [1, "x"]}', {"o": {"k": [1, "x"]}}),
])
async def test_json_data_given_as_text_keeps_working_in_both_modes(json_data, value, expected):
    async with http_server() as (base, requests):
        config = {"url": base + "/echo", "method": "POST", "json_data": json_data}
        for control in (None, "onStart"):
            _, events, _ = await run(fetch_graph(config, control=control), value)
            assert error_frames(events) == []
        assert [json.loads(item["body"]) for item in requests] == [expected, expected]


async def test_json_text_that_cannot_render_to_json_fails_fast_as_template_error():
    async with http_server() as (base, requests):
        config = {"url": base + "/echo", "method": "POST",
                  "json_data": '{"n": {{ handle_fetch_input }}}'}
        graph, events, elapsed = await run(fetch_graph(config, parser="{{ response }}"), 'abc", "admin": true')
    assert requests == [] and elapsed < 2
    frames = error_frames(events)
    assert [(frame["node_id"], frame["error_type"]) for frame in frames] == [("fetch", "TemplateError")]
    assert frames[0]["context"]["field"] == "json_data"
    assert not any("TimeoutError" == frame["error_type"] for frame in frames)


# ---------------------------------------------------------------------------
# Data is not a template: input handles, Hook content, tool arguments
# ---------------------------------------------------------------------------


DATA_CHANNELS = [
    # (input handle, static config, message, check(request))
    ("handle-url", {"method": "GET"},
     None, lambda request, payload: request["query"] == {"q": payload, "k": "{{env." + CANARY_NAME + "}}"}),
    ("handle-fetch-headers", {"method": "GET"},
     None, lambda request, payload: request["headers"] == {"X-Echo": payload, "X-Key": "{{env." + CANARY_NAME + "}}"}),
    ("handle-fetch-json_data", {"method": "POST"},
     None, lambda request, payload: json.loads(request["body"]) == {"q": payload, "k": "{{env." + CANARY_NAME + "}}"}),
    ("handle-fetch-data", {"method": "POST"},
     None, lambda request, payload: request["body"] == payload + "|{{env." + CANARY_NAME + "}}"),
    ("handle_fetch_input", {"method": "POST", "json_data": {"q": "{{ handle_fetch_input }}"}},
     None, lambda request, payload: json.loads(request["body"]) == {"q": payload + "|{{env." + CANARY_NAME + "}}"}),
]


def channel_message(handle, base, payload):
    env = "{{env." + CANARY_NAME + "}}"
    if handle == "handle-url":
        return f"{base}/echo?q={payload}&k={env}"
    if handle == "handle-fetch-headers":
        return json.dumps({"X-Echo": payload, "X-Key": env})
    if handle == "handle-fetch-json_data":
        return json.dumps({"q": payload, "k": env})
    return payload + "|" + env


@pytest.mark.parametrize("payload", [SSTI, "{{ 7*7 }}", "{% if true %}yes{% endif %}"])
@pytest.mark.parametrize("handle,config,_message,check", DATA_CHANNELS, ids=[item[0] for item in DATA_CHANNELS])
@pytest.mark.parametrize("control", [None, "onStart"])
async def test_input_handle_values_are_sent_verbatim_never_rendered(payload, handle, config, _message, check, control):
    async with http_server() as (base, requests):
        fetch_data = {"url": base + "/echo", **config}
        if handle == "handle-fetch-headers":
            fetch_data["headers"] = {"X-Static": "ignored when headers arrive on the input"}
        message = channel_message(handle, base, payload)
        _, events, _ = await run(fetch_graph(fetch_data, handle=handle, control=control), message)
    assert error_frames(events) == []
    assert len(requests) == 1
    request = requests[0]
    assert check(request, payload), request
    sent = json.dumps(request)
    assert CANARY not in sent and PID not in sent and "49" not in sent


async def test_tool_arguments_and_tool_parameters_render_once_and_stay_literal():
    payloads = [SSTI, "{{env." + CANARY_NAME + "}}", "{{ 7*7 }}"]
    async with http_server() as (base, requests):
        plain = FetchToolCallable(url_template=base + "/echo?q={{q}}", method="POST",
                                  headers={"X-Key": "{{env." + CANARY_NAME + "}}"},
                                  json_data={"q": "{{q}}", "nested": ["{{q}}"]}, tool_name="plain")
        schema = FetchToolCallable(url_template=base + "/unused", method="GET", tool_name="schema", tool_parameters={
            "url": base + "/echo?q={{q}}&k={{env." + CANARY_NAME + "}}",
            "method": "POST",
            "headers": '{"X-Echo": "{{q}}"}',
            "json_data": '{"q": "{{q}}"}',
            "extra": "{{q}}",
        })
        _ = schema.tool_schema
        for payload in payloads:
            for tool in (plain, schema):
                result = json.loads(await tool(q=payload))
                assert result["json"]["q"] == payload
                assert result["query"]["q"] == payload
        assert len(requests) == 6
        for index, payload in enumerate(payloads):
            plain_request, schema_request = requests[2 * index], requests[2 * index + 1]
            assert json.loads(plain_request["body"]) == {"q": payload, "nested": [payload]}
            assert plain_request["headers"] == {"X-Key": CANARY}
            assert json.loads(schema_request["body"]) == {"q": payload, "extra": payload}
            assert schema_request["headers"] == {"X-Echo": payload}
            # The authored env placeholder resolves; the argument never does.
            assert schema_request["query"] == {"q": payload, "k": CANARY}
            assert PID not in json.dumps([plain_request["body"], schema_request["body"]])


async def test_tool_mode_fetch_with_wired_url_sends_it_verbatim():
    async with http_server() as (base, requests):
        node = NodeFetch(FetchNodeModel(url=base + "/unused", method="GET", tool_mode=True, tool_name="t"),
                         node_id="tool")
        node.inputs[node.INPUT_HANDLE_URL] = base + "/echo?q={{q}}&k={{env." + CANARY_NAME + "}}"
        events = [event async for event in node.process(ModelAgentRunLog()) if event["type"] == node.OUTPUT_HANDLE]
        tool = events[0]["content"]["content"]
        # A wired URL is data: its braces are not tool arguments either.
        assert set(tool.tool_schema["function"]["parameters"]["properties"]) == {"parameters"}
        await tool(q="model-arg")
    assert requests[0]["query"] == {"q": "{{q}}", "k": "{{env." + CANARY_NAME + "}}"}


async def test_one_sandboxed_environment_with_project_filters():
    from jinja2.sandbox import SandboxedEnvironment
    assert isinstance(FETCH_TEMPLATE_ENV, SandboxedEnvironment)
    assert {"fromjson", "regex_replace", "regex_findall", "tojson"} <= set(FETCH_TEMPLATE_ENV.filters)
    renderer = FetchRenderer({"x": '{"a": "b c"}'}, FetchSecrets())
    assert renderer.tree({"v": "{{ (x | fromjson).a | regex_replace(' ', '-') }}"}, "json_data") == {"v": "b-c"}
    assert renderer.tree(["{{ x | regex_findall('[a-c]') | join }}"], "json_data") == ["abc"]
    # An authored template that tries to escape the sandbox fails as TemplateError.
    with pytest.raises(TemplateError) as raised:
        FetchRenderer({}, FetchSecrets()).text(SSTI, "url")
    assert raised.value.context["exception_type"] == "SecurityError"


async def test_legacy_tojson_idiom_and_whole_tojson_documents():
    for value in HOSTILE_INPUTS + ["unicode é ✓", "tab\tand sep"]:
        renderer = FetchRenderer({"x": value}, FetchSecrets())
        assert renderer.leaf("{{ (x | tojson)[1:-1] }}", "json_data") == value
        assert renderer.leaf("{{ x }}", "json_data") == value
        # A whole tojson document in a string leaf is that JSON text.
        assert renderer.leaf("{{ x | tojson }}", "json_data") == json.dumps(value)
        # In JSON text it is the JSON value.
        assert renderer.json_source('{"v": {{ x | tojson }}}', "json_data") == {"v": value}


# ---------------------------------------------------------------------------
# Typed, fast step-mode failures with sanitized diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["http_500", "closed_port", "url_template", "body_template", "bad_json_response"])
async def test_step_failure_raises_typed_error_and_bypasses_downstream_at_once(case):
    async with http_server() as (base, requests):
        config = {
            "http_500": {"url": base + "/status/500", "method": "GET"},
            "closed_port": {"url": closed_port_url() + "/x", "method": "GET"},
            "url_template": {"url": base + "/echo?q={{ handle_fetch_input.a.b }}", "method": "GET"},
            "body_template": {"url": base + "/echo", "method": "POST",
                              "json_data": {"q": "{{ handle_fetch_input.missing.attr }}"}},
            "bad_json_response": {"url": base + "/bad-json", "method": "GET"},
        }[case]
        expected = {"http_500": "HTTPError", "closed_port": "NetworkError", "url_template": "TemplateError",
                    "body_template": "TemplateError", "bad_json_response": "UnexpectedError"}[case]
        data = fetch_graph(config, parser="Got {{ response }}", timeout=5)
        graph, events, elapsed = await run(data, "hello")
    assert elapsed < 2, "downstream must not wait for the input timeout"
    frames = error_frames(events)
    assert [(frame["node_id"], frame["error_type"]) for frame in frames] == [("fetch", expected)]
    assert "timed out waiting for inputs" not in json.dumps(frames)
    assert graph.nodes["parser"].outputs == {} and graph.nodes["end"].outputs == {}
    if case == "http_500":
        assert frames[0]["error_message"] == "HTTP request failed with status 500: Internal Server Error"
        assert frames[0]["context"] == {"method": "GET", "url": base + "/status/500", "status_code": 500}
    if case in ("url_template", "body_template"):
        assert requests == []
        assert frames[0]["context"]["field"] == ("url" if case == "url_template" else "json_data")


async def test_failure_diagnostics_never_contain_env_values(caplog):
    caplog.set_level(logging.DEBUG)
    sink = Sink()
    env = "{{env." + CANARY_NAME + "}}"
    async with http_server() as (base, requests):
        config = {"url": f"http://user:{env}@127.0.0.1:{base.rsplit(':', 1)[1]}/status/503?key={env}#frag",
                  "method": "GET", "headers": {"X-Key": env, "X-Auth": "Bearer " + env}}
        graph, events, elapsed = await run(fetch_graph(config, parser="{{ response }}"), "hello",
                                           hooks=RuntimeConfig(graph_hooks=[DebugSSEHook(sink=sink, id_chat="c")]))
        # Hook-controlled: the onError Hook sees the scrubbed typed failure.
        controlled = fetch_graph(config, control="onError", hook_code=RECORD_ERROR_HOOK)
        _, controlled_events, _ = await run(controlled, "hello")
    assert requests[0]["query"] == {"key": CANARY}  # the request itself is authentic
    assert elapsed < 2
    node_errors = [event["content"] for event in sink.events if event["event_type"] == "node_error"]
    fetch_error = next(item for item in node_errors if item["node_id"] == "fetch")
    assert fetch_error["error_type"] == "HTTPError"
    assert fetch_error["context"] == {"method": "GET", "url": base + "/status/503", "status_code": 503}
    frames = [frame for frame in error_frames(events) if frame["node_id"] == "fetch"]
    assert len(frames) == 1 and frames[0]["lifecycle_reported"] is True  # no duplicate diagnostic
    record = hook_results(controlled_events)[0]
    error = record["outcome"]["error"]
    assert error["code"] == "HTTP_ERROR" and error["retryable"] is True
    assert error["details"]["status_code"] == 503 and "[REDACTED]" in error["details"]["response_body"]
    # The loopback server's own access log is the receiving side, not ours.
    runtime_logs = "\n".join(record.getMessage() for record in caplog.records
                              if not record.name.startswith("aiohttp.access"))
    diagnostics = json.dumps([events, sink.events, controlled_events], default=str)
    assert CANARY not in diagnostics + runtime_logs
    # No query string, fragment or user info in any frame or SSE payload.
    assert "key=" not in diagnostics and "#frag" not in diagnostics and "user:" not in diagnostics


async def test_network_and_template_errors_never_echo_the_rendered_url():
    secrets = FetchSecrets()
    renderer = FetchRenderer({}, secrets)
    url = renderer.text(closed_port_url() + "/x?key={{env." + CANARY_NAME + "}}", "url")
    assert CANARY in url
    assert safe_url(url, secrets).endswith("/x")
    node = NodeFetch(FetchNodeModel(url=closed_port_url() + "/p?key={{env." + CANARY_NAME + "}}",
                                    headers={"X-Key": "{{env." + CANARY_NAME + "}}"}), node_id="f")
    node.inputs["handle_fetch_input"] = "go"
    with pytest.raises(NetworkError) as raised:
        async for _ in node.process(ModelAgentRunLog()):
            pass
    assert CANARY not in str(raised.value) + json.dumps(raised.value.context)
    assert "key=" not in str(raised.value)
    assert raised.value.context["exception_type"] == "ClientConnectorError"


async def test_hook_controlled_network_and_template_failures_keep_their_codes():
    async with http_server() as (base, _):
        closed = closed_port_url() + "/x"
        network = fetch_graph({"url": closed + "?key=q#frag", "method": "GET"}, control="onFinish")
        _, events, elapsed = await run(network, "go")
        template = fetch_graph({"url": base + "/echo?q={{ handle_fetch_input.a.b }}"}, control="onFinish")
        _, template_events, _ = await run(template, "go")
    assert elapsed < 2
    error = hook_results(events)[0]["outcome"]["error"]
    assert error["code"] == "NETWORK_ERROR" and error["retryable"] is True
    assert error["details"]["exception_type"] == "ClientConnectorError"
    # The sanitized Fetch context travels in the outcome details.
    assert error["details"]["context"] == {"method": "GET", "url": closed,
                                           "exception_type": "ClientConnectorError"}
    template_error = hook_results(template_events)[0]["outcome"]["error"]
    assert template_error["code"] == "NODE_EXCEPTION"
    assert template_error["details"]["exception_type"] == "TemplateError"
    assert template_error["details"]["context"]["field"] == "url"
    assert template_error["message"].startswith("URL templating failed:")
    # The controlled node fails like any other node: OperationFailure, at
    # once, and its frame carries the same sanitized context.
    frames = [frame for frame in error_frames(events) if frame["node_id"] == "fetch"]
    assert [frame["error_type"] for frame in frames] == ["OperationFailure"]
    assert frames[0]["context"] == error["details"]["context"]


async def test_on_error_degrade_hook_still_recovers_a_failed_step_fetch():
    async with http_server() as (base, _):
        data = fetch_graph({"url": base + "/status/500", "method": "GET"}, control="onError",
                           hook_code=DEGRADE_HOOK, parser="{{ response.name }}|{{ response.code }}|{{ response.message }}")
        graph, events, elapsed = await run(data, "go")
    assert elapsed < 2
    record = hook_results(events)[0]
    assert record["recovered"] is True
    assert record["original_outcome"]["error"]["details"]["status_code"] == 500
    assert output(graph, "parser", "handle_parser_output") == "fallback|HTTP_ERROR|HTTP 500: Internal Server Error"
    assert [frame for frame in error_frames(events) if frame["node_id"] == "fetch"] == []


async def test_loop_iteration_with_a_failing_fetch_aggregates_none_not_stale_data():
    async with http_server() as (base, requests):
        data = {"type": "graph", "debug": True, "timeout": 5, "nodes": [
            {"id": "input", "type": "user_input"},
            {"id": "loop", "type": "loop"},
            {"id": "fetch", "type": "fetch", "data": {"url": base + "/flaky/{{ handle_fetch_input }}", "method": "GET"}},
            {"id": "parser", "type": "parser", "data": {"text": "{{ repo.name }}"}},
            {"id": "end", "type": "end"},
        ], "edges": [
            edge("input", "handle_user_message", "loop", "handle_list"),
            edge("loop", "handle_item", "fetch", "handle_fetch_input"),
            edge("fetch", "handle_fetch_output", "parser", "repo"),
            edge("parser", "handle_parser_output", "loop", "handle_loop"),
            edge("loop", "handle_end", "end", "handle_flow_input"),
        ]}
        graph, events, elapsed = await run(data, '["a", "b"]')
    assert elapsed < 2 and len(requests) == 2
    assert output(graph, "loop", "handle_end") == ["repo-a", None]
    frames = [frame for frame in error_frames(events) if frame["node_id"] == "fetch"]
    assert [frame["error_type"] for frame in frames] == ["HTTPError"]
    assert frames[0]["context"] == {"method": "GET", "url": base + "/flaky/b", "status_code": 500}


async def test_inner_flow_with_a_failing_child_fetch_returns_fast():
    async with http_server() as (base, _):
        inner = fetch_graph({"url": base + "/status/500", "method": "GET"}, parser="{{ response }}")
        inner.pop("timeout")  # the child keeps its 60 s default input timeout
        data = {"type": "graph", "debug": True, "timeout": 5, "nodes": [
            {"id": "input", "type": "user_input"},
            {"id": "inner", "type": "inner", "data": {"magic_flow": inner}},
            {"id": "parser", "type": "parser", "data": {"text": "Inner said: {{ x }}"}},
            {"id": "end", "type": "end"},
        ], "edges": [
            edge("input", "handle_user_message", "inner", "handle_user_message"),
            edge("inner", "handle_execution_content", "parser", "x"),
            edge("parser", "handle_parser_output", "end", "handle_flow_input"),
        ]}
        _, events, elapsed = await asyncio.wait_for(run(data, "go"), 10)
    assert elapsed < 2
    frames = error_frames(events)
    assert any(frame["error_type"] == "HTTPError" for frame in frames)
    assert "timed out waiting for inputs" not in json.dumps(frames)


async def test_no_inputs_still_yields_an_empty_result_without_a_request():
    async with http_server() as (base, requests):
        node = NodeFetch(FetchNodeModel(url=base + "/status/500", method="GET"), node_id="f")
        events = [event async for event in node.process(ModelAgentRunLog()) if event["type"] == node.OUTPUT_HANDLE]
    assert requests == [] and events[0]["content"]["content"] == {}


async def test_tool_mode_non_2xx_still_returns_the_body_to_the_model_with_secrets_scrubbed():
    async with http_server() as (base, _):
        tool = FetchToolCallable(url_template=base + "/status/401?key={{env." + CANARY_NAME + "}}",
                                 headers={"X-Key": "{{env." + CANARY_NAME + "}}"}, tool_name="t")
        result = await tool()
    assert result.startswith("HTTP 401: Unauthorized\n\nResponse body:\n")
    body = json.loads(result.split("Response body:\n", 1)[1])
    assert body["error"] == "boom" and body["query"] == {"key": "[REDACTED]"}
    assert CANARY not in result


async def test_typed_error_classes_expose_name_type_and_status():
    for cls in (HTTPError, NetworkError, TemplateError, UnexpectedError):
        error = cls("message", context={"method": "GET"}, status_code=418 if cls is HTTPError else None)
        assert error.error_type == cls.__name__ == type(error).__name__
    assert HTTPError("m", status_code=418).context == {"status_code": 418}


RETRY_HOOK = '''async def retry(context, chat_log):
    error = context.outcome["error"]
    if error["code"] != "HTTP_ERROR" or not error["retryable"]:
        return None
    child = await context.call("recover.handle-child-call->retry.handle_fetch_input", context.request["content"])
    if child["outcome"]["status"] != "success":
        return None
    return {"action": "outcome", "outcome": child["outcome"]}
'''


async def test_child_call_into_a_step_fetch_replays_the_request_content_as_its_inputs():
    async with http_server() as (base, requests):
        data = fetch_graph({"url": base + "/status/503?q={{ handle_fetch_input }}", "method": "GET"},
                           control="onError", hook_code=RETRY_HOOK)
        data["nodes"][-1]["id"] = "recover"
        data["nodes"].append({"id": "retry", "type": "fetch", "data": {
            "url": base + "/echo?q={{ handle_fetch_input }}", "method": "GET"}})
        data["edges"].append(edge("recover", "handle-child-call", "retry", "handle_fetch_input"))
        graph, events, elapsed = await run(data, 'a "quoted" {{ value }}')
    assert elapsed < 2
    assert [(item["path"], item["query"]["q"]) for item in requests] == [
        ("/status/503", 'a "quoted" {{ value }}'), ("/echo", 'a "quoted" {{ value }}')]
    record = hook_results(events)[0]
    assert record["recovered"] is True
    assert output(graph, "fetch", "handle_fetch_output")["query"] == {"q": 'a "quoted" {{ value }}'}


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------

ENCODED_CANARY_NAME = "FETCH_ENCODED_CANARY"
# '+', '/', '=', a space and a non-ASCII character: every encoder (aiohttp
# params, yarl path/query, form, JSON) writes it differently; "ZZ9marker"
# survives all of them, so its absence proves every form was scrubbed.
ENCODED_CANARY = "enc+Q/w=K9 \u00e9-ZZ9marker"


def encoded_config(base, code=400):
    env = "{{env." + ENCODED_CANARY_NAME + "}}"
    return {"url": base + f"/raw-status/{code}/p/{env}?q={env}", "method": "POST",
            "params": {"k": env}, "json_data": {"s": env}}


async def test_encoded_env_values_are_scrubbed_from_hook_bodies_tool_strings_and_logs(caplog, monkeypatch):
    monkeypatch.setenv(ENCODED_CANARY_NAME, ENCODED_CANARY)
    caplog.set_level(logging.DEBUG)
    async with http_server() as (base, requests):
        controlled = fetch_graph(encoded_config(base), control="onError", hook_code=RECORD_ERROR_HOOK)
        _, events, _ = await run(controlled, "go")
        config = encoded_config(base)
        tool = FetchToolCallable(url_template=config["url"], method="POST", params=config["params"],
                                 json_data=config["json_data"], tool_name="t")
        tool_result = await tool()
        strict = FetchToolCallable(url_template=config["url"], method="POST", params=config["params"],
                                   json_data=config["json_data"], tool_name="t")
        strict._strict_errors = True
        from magic_agents.hooks.invocation_control import OperationFailure
        with pytest.raises(OperationFailure) as raised:
            await strict()
    # The requests themselves carried the secret in its encoded forms.
    assert len(requests) == 3 and all(item["query"]["k"] == ENCODED_CANARY for item in requests)
    error = hook_results(events)[0]["outcome"]["error"]
    assert error["code"] == "HTTP_ERROR" and "[REDACTED]" in error["details"]["response_body"]
    assert tool_result.startswith("HTTP 400: Bad Request") and "[REDACTED]" in tool_result
    runtime_logs = "\n".join(record.getMessage() for record in caplog.records
                              if not record.name.startswith("aiohttp.access"))
    diagnostics = json.dumps([events, tool_result, raised.value.outcome], default=str)
    assert "ZZ9marker" not in diagnostics + runtime_logs


async def test_tool_mode_network_errors_never_echo_the_full_url():
    from magic_agents.hooks.invocation_control import OperationFailure
    env = "{{env." + CANARY_NAME + "}}"
    port = closed_port_url().rsplit(":", 1)[1]
    urls = [
        f"http://user:LITERALPASSWORD@127.0.0.1:99999/x?token=LITERALQUERYTOKEN&key={env}",
        f"http://user:LITERALPASSWORD@[::1/x?token=LITERALQUERYTOKEN&key={env}",
        f"ftp://user:LITERALPASSWORD@127.0.0.1/x?token=LITERALQUERYTOKEN&key={env}",
        f"http://user:LITERALPASSWORD@127.0.0.1:{port}/x?token=LITERALQUERYTOKEN&key={env}",
    ]
    for url in urls:
        result = await FetchToolCallable(url_template=url, tool_name="t")()
        strict = FetchToolCallable(url_template=url, tool_name="t")
        strict._strict_errors = True
        with pytest.raises(OperationFailure) as raised:
            await strict()
        outcome = raised.value.outcome
        assert outcome["error"]["code"] == "NETWORK_ERROR"
        text = result + json.dumps(outcome)
        assert json.loads(result)["error"].startswith("Network error: ")
        for leaked in ("LITERALPASSWORD", "LITERALQUERYTOKEN", CANARY, "key=", "user:"):
            assert leaked not in text, (url, text)


@pytest.mark.parametrize("template,value,expected", [
    ("{{ x|e }}", "C:\\temp\\new", "C:\\temp\\new"),
    ("{{ x|e }}", "q\\u0022x", "q\\u0022x"),
    ("{{ x|e }}", "end\\", "end\\"),
    ("{{ x|e }}", 'say "hi"', "say &#34;hi&#34;"),
    ("{{ x|safe }}", 'say "hi"\\n', 'say "hi"\\n'),
    ("{{ x|forceescape }}", "a\\b<", "a\\b&lt;"),
    ("{% filter upper %}{{ x }}{% endfilter %}", 'say "hi"\\', 'SAY "HI"\\'),
])
async def test_markup_and_filter_blocks_are_json_escaped_in_body_leaves(template, value, expected):
    assert FetchRenderer({"x": value}, FetchSecrets()).leaf(template, "json_data") == expected


@pytest.mark.parametrize("json_data,value,expected", [
    # Exactly one JSON value outside a string is inserted raw (legacy).
    ('{"tags": {{ handle_fetch_input }}}', '["a", "b"]', {"tags": ["a", "b"]}),
    ('{{ handle_fetch_input }}', '{"q": "x", "n": 1}', {"q": "x", "n": 1}),
    ('{"cfg": {{env.FETCH_JSON_CANARY}}, "n": {{ handle_fetch_input }}}', "3", {"cfg": {"a": 1, "b": "two"}, "n": 3}),
    # Inside a string it is escaped: JSON text cannot add list items there.
    ('{"n": {{ handle_fetch_input }}, "s": ["{{ handle_fetch_input }}"]}', '[", 1, 2, "]',
     {"n": [", 1, 2, "], "s": ['[", 1, 2, "]']}),
    # A filter block writes through an expression: escaped like any value.
    ('{"n": 1, "q": "{% filter lower %}{{ handle_fetch_input }}{% endfilter %}"}', '", "ADMIN": "1',
     {"n": 1, "q": '", "admin": "1'}),
    ('{"n": {{ handle_fetch_input | tojson }}, "q": "{{ handle_fetch_input | tojson }}"}', 'a "b"',
     {"n": 'a "b"', "q": '"a \\"b\\""'}),
])
async def test_json_text_places_inserted_values_by_position_in_both_modes(monkeypatch, json_data, value, expected):
    monkeypatch.setenv("FETCH_JSON_CANARY", '{"a": 1, "b": "two"}')
    async with http_server() as (base, requests):
        config = {"url": base + "/echo", "method": "POST", "json_data": json_data}
        for control in (None, "onStart"):
            _, events, _ = await run(fetch_graph(config, control=control), value)
            assert error_frames(events) == []
        tool = FetchToolCallable(url_template=base + "/echo", method="POST", json_data=json_data, tool_name="t")
        await tool(handle_fetch_input=value)
    assert [json.loads(item["body"]) for item in requests] == [expected] * 3


@pytest.mark.parametrize("value", ['1, "admin": true', '1}, {"admin": true', "[1] [2]", "abc"])
async def test_json_text_rejects_values_that_are_not_one_json_value(value):
    renderer = FetchRenderer({"x": value}, FetchSecrets())
    with pytest.raises(TemplateError):
        renderer.json_source('{"n": {{ x }}}', "json_data")
    with pytest.raises(TemplateError):
        renderer.json_source('{"n": [{% filter trim %}{{ x }}{% endfilter %}]}', "json_data")


async def test_method_must_be_an_http_token(caplog):
    caplog.set_level(logging.DEBUG)
    forged = "POST\nINFO forged.logger admin login ok"
    async with http_server() as (base, requests):
        graph, events, elapsed = await run(
            fetch_graph({"url": base + "/echo", "json_data": {"a": 1}}, handle="handle-fetch-method",
                        parser="{{ response }}"), forged)
        tool = FetchToolCallable(url_template=base + "/echo", tool_name="t",
                                 tool_parameters={"method": "GET\r\nX-Injected: 1"})
        tool_result = await tool()
    assert requests == [] and elapsed < 2
    frames = [frame for frame in error_frames(events) if frame["node_id"] == "fetch"]
    assert [frame["error_type"] for frame in frames] == ["TemplateError"]
    assert frames[0]["context"] == {"field": "method"}
    assert json.loads(tool_result)["error"].startswith("Unexpected error: Request method is not a valid")
    runtime_logs = "\n".join(record.getMessage() for record in caplog.records
                              if record.name.startswith("magic_agents.node_system"))
    assert "INFO forged" not in runtime_logs and "X-Injected" not in runtime_logs


CHILD_EMPTY_HOOK = '''async def recover(context, chat_log):
    child = await context.call("recover.handle-child-call->static.handle_fetch_input", {})
    if child["outcome"]["status"] != "success":
        return None
    return {"action": "outcome", "outcome": child["outcome"]}
'''


async def test_child_call_with_empty_content_into_a_static_fetch_sends_the_request():
    async with http_server() as (base, requests):
        data = fetch_graph({"url": base + "/status/500", "method": "GET"}, control="onError",
                           hook_code=CHILD_EMPTY_HOOK)
        data["nodes"][-1]["id"] = "recover"
        data["nodes"].append({"id": "static", "type": "fetch", "data": {"url": base + "/echo?s=1", "method": "GET"}})
        data["edges"].append(edge("recover", "handle-child-call", "static", "handle_fetch_input"))
        graph, events, elapsed = await run(data, "go")
    assert elapsed < 2
    assert [item["path"] for item in requests] == ["/status/500", "/echo"]
    assert hook_results(events)[0]["recovered"] is True
    assert output(graph, "fetch", "handle_fetch_output")["query"] == {"s": "1"}


class GraphRecorder:
    def __init__(self):
        self.events = []

    async def on_node_error(self, context, error):
        self.events.append(("node_error", context.node_id, type(error).__name__))

    async def on_graph_error(self, context, error):
        self.events.append(("graph_error",))

    async def on_graph_end(self, context, *args, **kwargs):
        self.events.append(("graph_end",))


def side_branch_graph(base, control=None):
    """input -> notify Fetch (no downstream) and input -> parser -> end."""
    data = fetch_graph({"url": base + "/status/500", "method": "POST",
                        "json_data": {"event": "{{ handle_fetch_input }}"}}, control=control, hook_code=DEGRADE_HOOK)
    data["edges"] = [item for item in data["edges"] if item["source"] != "fetch"]
    data["nodes"].append({"id": "parser", "type": "parser", "data": {"text": "answer: {{ m }}"}})
    data["edges"] += [edge("input", "handle_user_message", "parser", "m"),
                      edge("parser", "handle_parser_output", "end", "handle_flow_input")]
    return data


async def test_a_failing_fetch_without_downstream_is_a_node_error_unless_a_hook_degrades_it():
    from magic_agents.hooks.flow_hooks import FlowHooks

    class Recorder(GraphRecorder, FlowHooks):
        pass

    async with http_server() as (base, requests):
        plain = Recorder()
        graph, _, elapsed = await run(side_branch_graph(base), "hi", hooks=RuntimeConfig(graph_hooks=[plain]))
        degraded = Recorder()
        degraded_graph, events, _ = await run(side_branch_graph(base, control="onError"), "hi",
                                              hooks=RuntimeConfig(graph_hooks=[degraded]))
    assert elapsed < 2 and len(requests) == 2
    # The rest of the graph still delivers its answer...
    assert output(graph, "parser", "handle_parser_output") == "answer: hi"
    # ...but the failed request is reported: the run ends with a graph error.
    assert ("node_error", "fetch", "HTTPError") in plain.events and ("graph_error",) in plain.events
    # A best-effort call keeps the run successful with an onError Hook.
    assert output(degraded_graph, "parser", "handle_parser_output") == "answer: hi"
    assert hook_results(events)[0]["recovered"] is True
    assert not any(item[0] in ("node_error", "graph_error") for item in degraded.events)


async def test_sandbox_allows_list_methods_but_never_mutates_inputs():
    shared = {"items": ["a"], "meta": {"k": "v"}}
    renderer = FetchRenderer({"d": shared, "x": "b,c"}, FetchSecrets())
    template = ("{% set l = [] %}{% for p in x.split(',') %}{% set _ = l.append(p|upper) %}{% endfor %}"
                "{% set _ = d['items'].append('z') %}{% set _ = d.meta.update({'k': 'w'}) %}"
                "{{ l|join('-') }}|{{ d['items']|join }}|{{ d.meta.k }}")
    assert renderer.text(template, "url") == "B-C|az|w"
    assert shared == {"items": ["a"], "meta": {"k": "v"}}
    with pytest.raises(TemplateError) as raised:
        renderer.text("{{ ''.__class__.__mro__ }}{{ x.format }}", "url")
    assert raised.value.context["exception_type"] == "SecurityError"


async def test_tool_schema_exposes_nested_body_placeholders():
    async with http_server() as (base, requests):
        tool = FetchToolCallable(url_template=base + "/echo", method="POST", tool_name="t",
                                 json_data={"filters": {"q": "{{q}}"}, "list": ["{{q}}", "{{page}}"]})
        assert set(tool.tool_schema["function"]["parameters"]["properties"]) == {"q", "page"}
        await tool(q="x", page="2")
        override = FetchToolCallable(url_template=base + "/echo", method="POST", tool_name="t",
                                     tool_parameters={"json_data": {"filters": {"q": "{{q}}"}}, "mode": "fast"})
        assert set(override.tool_schema["function"]["parameters"]["properties"]) == {"q"}
        await override(q="y")
    assert [json.loads(item["body"]) for item in requests] == [
        {"filters": {"q": "x"}, "list": ["x", "2"]}, {"filters": {"q": "y"}, "mode": "fast"}]
