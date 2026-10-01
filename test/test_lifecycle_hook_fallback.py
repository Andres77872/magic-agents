"""Lifecycle Hooks through the real graph, Fetch HTTP, and LLM agent loop.

Only model replies are scripted. Fetch uses aiohttp against a loopback server;
build/run_agent, NodeLLM, magic-llm tool execution, result injection, and Hook
execution use the installed editable production packages. No API credentials
or external service are used.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import os
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from magic_agents.agt_flow import build, run_agent
from magic_agents.node_system.NodeClientLLM import NodeClientLLM
from magic_agents.node_system.NodeParser import NodeParser
from magic_llm import MagicLLM
from magic_llm.model.ModelChatResponse import Choice, FunctionCall, Message, ModelChatResponse, ToolCall


pytestmark = pytest.mark.asyncio


class ScriptedEngine:
    """Model transport only; MagicLLM's real AsyncAgentLoop drives both turns."""
    engine = "openai"
    engine_name = "openai"
    model = "local-scripted-model"

    def __init__(self, call_id):
        self.call_id = call_id
        self.chats = []
        self.tool_messages = []
        self.rendezvous = None

    async def async_generate(self, chat, **kwargs):
        self.chats.append(deepcopy(chat.messages))
        if len(self.chats) == 1:
            if self.rendezvous is not None:
                self.rendezvous["started"] += 1
                if self.rendezvous["started"] == 2:
                    self.rendezvous["entered"].set()
                await self.rendezvous["entered"].wait()
            schemas = kwargs.get("tools", [])
            assert any(tool.get("function", {}).get("name") == "browse_primary" for tool in schemas)
            message = Message(role="assistant", content=None, tool_calls=[ToolCall(
                id=self.call_id, function=FunctionCall(name="browse_primary", arguments='{"query":"same query"}')
            )])
            finish_reason = "tool_calls"
        else:
            assert len(self.chats) == 2, "The script must not hide repeated provider turns"
            self.tool_messages = [message for message in chat.messages if message.get("role") == "tool"]
            assert len(self.tool_messages) == 1, "Internal fallback children must not invent provider tool results"
            assert self.tool_messages[0]["tool_call_id"] == self.call_id
            message = Message(role="assistant", content="Browsing completed")
            finish_reason = "stop"
        return ModelChatResponse(id=f"script-{len(self.chats)}", object="chat.completion", created=0,
                                 model=self.model, choices=[Choice(index=0, message=message, finish_reason=finish_reason)])


class ScriptedClient:
    # Keep the real magic-llm entry point, without initializing a provider client
    # or supplying a fake API key. Only its model transport is replaced.
    run_agent_async = MagicLLM.run_agent_async
    _task_executor = None

    def __init__(self, call_id):
        self.llm = ScriptedEngine(call_id)


@asynccontextmanager
async def http_providers(statuses, *, blocked=None):
    calls, queries = [], []
    entered, release = asyncio.Event(), asyncio.Event()

    async def handle(request):
        provider = request.match_info["provider"]
        calls.append(provider)
        query = request.query.get("query")
        if query is None and request.method == "POST":
            body = await request.json()
            query = body.get("query", body.get("q"))
        queries.append(query)
        if provider == blocked:
            entered.set()
            await release.wait()
        status = statuses[provider]
        if status >= 400:
            # Classification must use actual HTTP status, not this arbitrary
            # body (including misleading text on the success case below).
            return web.json_response({"provider": provider, "message": "Provider rejected request"}, status=status)
        return web.json_response({"provider": provider, "query": query,
                                  "text": "HTTP 429 is ordinary document text"})

    app = web.Application()
    app.router.add_get("/{provider}", handle)
    app.router.add_post("/{provider}", handle)
    async with TestServer(app) as server:
        try:
            yield server, calls, queries, entered, release
        finally:
            release.set()


def edge(identifier, source, target, source_handle, target_handle, *, hook=None):
    result = {"id": identifier, "source": source, "target": target,
              "sourceHandle": source_handle, "targetHandle": target_handle}
    if hook:
        result["hooks"] = {"enabled": True, "hook_node_id": hook}
    return result


RECOVER_A = '''async def recover(context, chat_log):
    failure = context.outcome["error"]
    if failure["code"] != "HTTP_ERROR" or failure["details"]["status_code"] != 429:
        return {"action": "pass"}
    child = await context.call("a-hook-to-b", context.request["content"])
    if child["outcome"]["status"] == "success":
        return {"action": "outcome", "outcome": child["outcome"]}
    return {"action": "pass"}
'''

RECOVER_B = '''async def recover(context, chat_log):
    failure = context.outcome["error"]
    if failure["code"] != "HTTP_ERROR" or failure["details"]["status_code"] != 429:
        return {"action": "pass"}
    return {"action": "redirect", "connection": "b-hook-to-c", "content": context.request["content"]}
'''

FINISH_C = '''async def finish(context, chat_log):
    import asyncio
    if context.outcome["status"] != "success":
        return {"action": "pass"}
    await asyncio.sleep(0)
    return {"action": "outcome", "outcome": {"status": "success", "content": {
        **context.outcome["content"], "reviewed_before_commit": True}}}
'''


def fallback_graph(server, *, scope="edge", callers=1):
    nodes = [{"id": "input", "type": "user_input", "data": {}}]
    edges = []
    for index in range(callers):
        suffix = str(index + 1)
        client_id, llm_id = "client-" + suffix, "llm-" + suffix
        nodes.extend([
            {"id": client_id, "type": "client", "data": {"engine": "openai", "model": "local-scripted-model"}},
            {"id": llm_id, "type": "llm", "data": {"stream": False, "json_output": False,
                                                        "agent_config": {"max_iterations": 3, "wall_clock_timeout": 5}}},
            {"id": "end-" + suffix, "type": "end", "data": {}},
        ])
        edges.extend([
            edge("input-" + suffix, "input", llm_id, "handle_user_message", "handle_user_message"),
            edge("client-" + suffix, client_id, llm_id, "handle-client-provider", "handle-client-provider"),
            edge("tool-" + suffix, "fetch-a", llm_id, "handle_fetch_output", "handle-tool-definition-0",
                 hook="recover-a" if scope == "edge" and index == 0 else None),
            edge("result-" + suffix, llm_id, "end-" + suffix, "handle_generated_content", "handle_flow_input"),
        ])
    for provider in ("a", "b", "c"):
        nodes.append({"id": "fetch-" + provider, "type": "fetch", "data": {
            "url": str(server.make_url("/" + provider)) + "?query={{query}}", "method": "GET", "tool_mode": True,
            "tool_name": "browse_primary" if provider == "a" else "browse_" + provider,
        }})
    a_data = {"function_template": RECOVER_A, "lifecycle_event": "onError"}
    if scope == "node":
        a_data["target_node_id"] = "fetch-a"
    nodes.extend([
        {"id": "recover-a", "type": "hook", "data": a_data},
        {"id": "recover-b", "type": "hook", "data": {"function_template": RECOVER_B,
                                                        "lifecycle_event": "onError", "target_node_id": "fetch-b"}},
        {"id": "finish-c", "type": "hook", "data": {"function_template": FINISH_C,
                                                      "lifecycle_event": "onFinish", "target_node_id": "fetch-c"}},
    ])
    edges.extend([
        edge("a-hook-to-b", "recover-a", "fetch-b", "handle-child-call", "handle_fetch_input"),
        edge("b-hook-to-c", "recover-b", "fetch-c", "handle-child-call", "handle_fetch_input"),
    ])
    return {"type": "graph", "debug": True, "timeout": 5, "nodes": nodes, "edges": edges}


def build_scripted(monkeypatch, data, *, callers=1):
    clients = {"client-" + str(index + 1): ScriptedClient("original-call-" + str(index + 1)) for index in range(callers)}

    def initialize(self, engine, model):
        self.client = clients[self.node_id]
        self.init_error = None
        self.init_error_type = None
        self._current_engine, self._current_model = engine, model

    monkeypatch.setattr(NodeClientLLM, "_initialize_client", initialize)
    graph = build(data, message="Find the same query")
    return graph, clients


async def collect(graph):
    return [event async for event in run_agent(graph)]


def tool_payload(client):
    assert len(client.llm.chats) == 2
    assert len(client.llm.tool_messages) == 1
    return json.loads(client.llm.tool_messages[0]["content"])


def walk(frame):
    yield frame
    for child in frame["child"]:
        yield from walk(child)


def root_record(graph, node_id, call_id):
    roots = [record for record in graph.nodes[node_id]._invocation_control.records
             if record["kind"] == "node" and record["node_id"] == node_id
             and record["parent_id"] is None and record["caller"].get("tool_call_id") == call_id]
    assert len(roots) == 1
    return roots[0]


def assert_original_provider_result(graph, client, expected_provider):
    payload = tool_payload(client)
    assert payload["provider"] == expected_provider
    # Accounting stays in runtime records/debug events; the model sees only
    # the adopted value under its original tool call ID.
    assert "execution" not in payload and "result" not in payload
    root = root_record(graph, "fetch-a", client.llm.call_id)
    assert root["outcome"] == {"status": "success", "content": payload}
    assert root["caller"]["tool_call_id"] == client.llm.call_id
    for frame in walk(root):
        if frame["kind"] == "node" and frame["id"] != root["id"]:
            assert "tool_call_id" not in frame["caller"]
    return payload, root


async def test_real_fetch_three_provider_hooks_finish_before_one_tool_result(monkeypatch):
    async with http_providers({"a": 429, "b": 429, "c": 200}) as (server, calls, queries, _, __):
        data = fallback_graph(server)
        data["nodes"].extend([
            {"id": "llm-start", "type": "hook", "data": {"lifecycle_event": "onStart", "target_node_id": "llm-1",
                "function_template": '''async def start(context, chat_log):
    import json
    json.dumps(context.request["content"])
    context.emit.debug({"phase": "start", "input_snapshot_is_json": True})
    return {"action": "pass"}
'''}},
            {"id": "llm-finish", "type": "hook", "data": {"lifecycle_event": "onFinish", "target_node_id": "llm-1",
                "function_template": '''async def finish(context, chat_log):
    if context.outcome["status"] != "success":
        return {"action": "pass"}
    return {"action": "outcome", "outcome": {"status": "success", "content": {**context.outcome["content"], "handle_generated_content": "LLM finalized:" + context.outcome["content"]["handle_generated_content"]}}}
'''}},
        ])
        graph, clients = build_scripted(monkeypatch, data)
        events = await asyncio.wait_for(collect(graph), 3)
    payload, root = assert_original_provider_result(graph, clients["client-1"], "c")
    assert calls == ["a", "b", "c"] and queries == ["same query"] * 3
    assert payload["reviewed_before_commit"] is True
    assert payload["text"] == "HTTP 429 is ordinary document text"
    assert root["original_outcome"]["error"]["code"] == "HTTP_ERROR" and root["recovered"]
    assert root["original_outcome"]["error"]["details"]["status_code"] == 429
    assert [frame["node_id"] for frame in walk(root)] == ["fetch-a", "recover-a", "fetch-b", "recover-b", "fetch-c", "finish-c"]
    results = [event for event in events if event.get("type") == "debug"
               and event.get("content", {}).get("event_type") == "TOOL_RESULT"]
    assert len(results) == 1 and results[0]["content"]["data"]["tool_call_id"] == "original-call-1"
    # These controls run on the actual NodeLLM, whose core needs the live client
    # and Fetch callable even though author-visible input snapshots are JSON.
    llm_frame = graph.nodes["llm-1"]._last_invocation_record
    assert [hook["node_id"] for hook in llm_frame["child"]] == ["llm-start", "llm-finish"]
    assert llm_frame["original_outcome"]["content"]["handle_generated_content"] == "Browsing completed"
    assert llm_frame["outcome"]["content"]["handle_generated_content"] == "LLM finalized:Browsing completed"
    assert graph.nodes["end-1"].inputs["handle_flow_input"] == "LLM finalized:Browsing completed"


async def test_first_success_avoids_the_configured_spare_http_provider(monkeypatch):
    async with http_providers({"a": 429, "b": 200, "c": 200}) as (server, calls, _, __, ___):
        graph, clients = build_scripted(monkeypatch, fallback_graph(server))
        await asyncio.wait_for(collect(graph), 3)
    _, root = assert_original_provider_result(graph, clients["client-1"], "b")
    assert calls == ["a", "b"]
    assert [frame["node_id"] for frame in walk(root)] == ["fetch-a", "recover-a", "fetch-b"]


async def test_all_http_providers_fail_and_pass_preserves_the_original_failure(monkeypatch):
    async with http_providers({"a": 429, "b": 429, "c": 429}) as (server, calls, _, __, ___):
        graph, clients = build_scripted(monkeypatch, fallback_graph(server))
        await asyncio.wait_for(collect(graph), 3)
    payload = tool_payload(clients["client-1"])
    root = root_record(graph, "fetch-a", clients["client-1"].llm.call_id)
    assert calls == ["a", "b", "c"]
    assert root["outcome"] == root["original_outcome"]
    assert payload["type"] == "OperationFailure" and "execution" not in payload
    assert root["outcome"]["status"] == "error" and not root["recovered"]
    assert root["outcome"]["error"]["code"] == "HTTP_ERROR"
    assert root["outcome"]["error"]["details"]["status_code"] == 429
    assert [frame["node_id"] for frame in walk(root) if frame["kind"] == "node"] == ["fetch-a", "fetch-b", "fetch-c"]
    assert all(frame["outcome"]["status"] == "error" for frame in walk(root) if frame["kind"] == "node")


@pytest.mark.parametrize("status,retryable", [(401, False), (403, False), (429, True), (503, True)])
async def test_http_error_classification_is_structured_and_hook_policy_selects_only_429(monkeypatch, status, retryable):
    async with http_providers({"a": status, "b": 200, "c": 200}) as (server, calls, _, __, ___):
        graph, clients = build_scripted(monkeypatch, fallback_graph(server))
        await asyncio.wait_for(collect(graph), 3)
    payload = tool_payload(clients["client-1"])
    root = root_record(graph, "fetch-a", clients["client-1"].llm.call_id)
    original = root["original_outcome"]["error"]
    assert original["code"] == "HTTP_ERROR"
    assert original["details"]["status_code"] == status and original["retryable"] is retryable
    assert calls == (["a", "b"] if status == 429 else ["a"])
    assert root["outcome"]["status"] == ("success" if status == 429 else "error")
    assert "execution" not in payload


@pytest.mark.parametrize("scope", ["edge", "node"])
async def test_shared_fetch_node_uses_the_selected_caller_edge_or_all_node_callers(monkeypatch, scope):
    async with http_providers({"a": 429, "b": 429, "c": 200}) as (server, calls, _, __, ___):
        graph, clients = build_scripted(monkeypatch, fallback_graph(server, scope=scope, callers=2), callers=2)
        await asyncio.wait_for(collect(graph), 3)
    _, first = assert_original_provider_result(graph, clients["client-1"], "c")
    if scope == "node":
        _, second = assert_original_provider_result(graph, clients["client-2"], "c")
        assert first["id"] != second["id"] and first["root_id"] != second["root_id"]
        assert Counter(calls) == {"a": 2, "b": 2, "c": 2}
    else:
        # The existing unhooked Fetch tool keeps its legacy HTTP-error content;
        # it cannot inherit another caller's recovered final outcome.
        other = clients["client-2"].llm.tool_messages
        assert len(other) == 1 and other[0]["tool_call_id"] == "original-call-2"
        assert "429" in other[0]["content"]
        assert Counter(calls) == {"a": 2, "b": 1, "c": 1}


async def test_external_cancellation_stops_http_fallback_without_committing_a_tool_result(monkeypatch):
    async with http_providers({"a": 429, "b": 200, "c": 200}, blocked="b") as (server, calls, _, entered, release):
        graph, clients = build_scripted(monkeypatch, fallback_graph(server))
        task = asyncio.create_task(collect(graph))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(clients["client-1"].llm.chats) == 1
            assert clients["client-1"].llm.tool_messages == []
            assert calls == ["a", "b"]
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def test_existing_linear_edge_hook_emits_before_original_target_delivery():
    template = '''async def observe(context, chat_log):
    import asyncio
    await asyncio.sleep(0)
    return emit.user("legacy hook finished")
'''
    graph = build({"type": "graph", "debug": False, "timeout": 2, "nodes": [
        {"id": "input", "type": "user_input", "data": {}},
        {"id": "legacy", "type": "hook", "data": {"function_template": template}},
        {"id": "target", "type": "parser", "data": {"text": "{{ handle_parser_input }}"}},
        {"id": "end", "type": "end", "data": {}},
    ], "edges": [edge("delivery", "input", "target", "handle_user_message", "handle_parser_input", hook="legacy"),
                 edge("result", "target", "end", "handle_parser_output", "handle_flow_input")]}, message="original payload")
    timeline = []
    hook = graph.nodes["legacy"]
    original_hook = hook.process

    async def observed_hook(chat_log):
        async for event in original_hook(chat_log):
            timeline.append("hook emit")
            yield event
        timeline.append("hook complete")

    class ObservedInputs(dict):
        def __setitem__(self, key, value):
            if key == "handle_parser_input":
                timeline.append("target delivery")
            super().__setitem__(key, value)

    hook.process = observed_hook
    graph.nodes["target"].inputs = ObservedInputs(graph.nodes["target"].inputs)
    await asyncio.wait_for(collect(graph), 2)
    assert timeline.index("hook emit") < timeline.index("hook complete") < timeline.index("target delivery")
    assert graph.nodes["target"].outputs["handle_parser_output"]["content"] == "original payload"
    assert graph.nodes["end"].inputs["handle_flow_input"] == "original payload"


def parser_graph(*, start_template, finish_template, child=False):
    nodes = [
        {"id": "input", "type": "user_input", "data": {}},
        {"id": "target", "type": "parser", "data": {"text": "core:{{ handle_parser_input }}"}},
        {"id": "prepare", "type": "hook", "data": {"function_template": start_template,
                                                       "lifecycle_event": "onStart", "target_node_id": "target"}},
        {"id": "finish", "type": "hook", "data": {"function_template": finish_template,
                                                      "lifecycle_event": "onFinish", "target_node_id": "target"}},
        {"id": "end", "type": "end", "data": {}},
    ]
    edges = [edge("input-target", "input", "target", "handle_user_message", "handle_parser_input"),
             edge("target-end", "target", "end", "handle_parser_output", "handle_flow_input")]
    if child:
        nodes.append({"id": "child", "type": "parser", "data": {"text": "child:{{ handle_parser_input }}"}})
        edges.append(edge("prepare-child", "prepare", "child", "handle-child-call", "handle_parser_input"))
    return {"type": "graph", "debug": True, "timeout": 3, "nodes": nodes, "edges": edges}


PARSER_FINISH = '''async def finish(context, chat_log):
    if context.outcome["status"] != "success":
        return {"action": "pass"}
    return {"action": "outcome", "outcome": {"status": "success", "content": {**context.outcome["content"], "handle_parser_output": "finished:" + context.outcome["content"]["handle_parser_output"]}}}
'''


async def test_non_fetch_node_admission_and_finish_transform_before_downstream_delivery(monkeypatch):
    prepare = '''async def prepare(context, chat_log):
    content = context.request["content"]
    content["handle_parser_input"] = content["handle_parser_input"].strip().lower()
    return {"action": "input", "content": content}
'''
    core_inputs = []
    original = NodeParser.process

    async def observe_core(self, chat_log):
        core_inputs.append(deepcopy(self.inputs))
        async for event in original(self, chat_log):
            yield event

    monkeypatch.setattr(NodeParser, "process", observe_core)
    graph = build(parser_graph(start_template=prepare, finish_template=PARSER_FINISH), message=" ORIGINAL ")
    await asyncio.wait_for(collect(graph), 2)
    assert core_inputs == [{"handle_parser_input": "original"}]
    assert graph.nodes["end"].inputs["handle_flow_input"] == "finished:core:original"
    frame = graph.nodes["target"]._last_invocation_record
    assert frame["input"] == {"handle_parser_input": " ORIGINAL "}
    assert frame["request"]["content"] == {"handle_parser_input": "original"}
    assert frame["original_outcome"]["status"] == "success"
    assert frame["original_outcome"]["content"]["handle_parser_output"] == "core:original"
    assert frame["outcome"] == {"status": "success", "content": {
        **frame["original_outcome"]["content"], "handle_parser_output": "finished:core:original"}}
    assert [hook["event"] for hook in frame["child"]] == ["onStart", "onFinish"]


async def test_general_hook_redirect_executes_a_real_parser_child_and_skips_original_core(monkeypatch):
    prepare = '''async def prepare(context, chat_log):
    return {"action": "redirect", "connection": "prepare-child", "content": context.request["content"]["handle_parser_input"]}
'''
    core_nodes = []
    original = NodeParser.process

    async def observe_core(self, chat_log):
        core_nodes.append(self.node_id)
        async for event in original(self, chat_log):
            yield event

    monkeypatch.setattr(NodeParser, "process", observe_core)
    graph = build(parser_graph(start_template=prepare, finish_template=PARSER_FINISH, child=True), message="payload")
    await asyncio.wait_for(collect(graph), 2)
    assert core_nodes == ["child"]
    assert graph.nodes["end"].inputs["handle_flow_input"] == "finished:child:payload"
    frame = graph.nodes["target"]._last_invocation_record
    assert not frame["executed"]
    assert frame["original_outcome"]["status"] == "success"
    assert frame["original_outcome"]["content"]["handle_parser_output"] == "child:payload"
    assert [item["node_id"] for item in walk(frame)] == ["target", "prepare", "child", "finish"]
    assert frame["child"][0]["child"][0]["caller"] == {"node_id": "prepare", "edge_id": "prepare-child"}


async def test_cancelling_general_node_admission_never_executes_or_delivers_its_core(monkeypatch):
    prepare = '''async def prepare(context, chat_log):
    import asyncio
    await asyncio.Event().wait()
    return {"action": "input", "content": context.request["content"]}
'''
    graph = build(parser_graph(start_template=prepare, finish_template=PARSER_FINISH), message="payload")
    entered = asyncio.Event()
    hook = graph.nodes["prepare"]
    original = hook.invoke_control

    async def observe_hook(context, chat_log):
        entered.set()
        return await original(context, chat_log)

    monkeypatch.setattr(hook, "invoke_control", observe_hook)
    task = asyncio.create_task(collect(graph))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "handle_parser_output" not in graph.nodes["target"].outputs
        assert "handle_flow_input" not in graph.nodes["end"].inputs
        roots = [frame for frame in graph.nodes["target"]._invocation_control.records
                 if frame["node_id"] == "target" and frame["parent_id"] is None]
        assert len(roots) == 1 and roots[0]["outcome"] == {"status": "cancelled"}
        assert not roots[0]["executed"]
        assert [child["event"] for child in roots[0]["child"]] == ["onStart", "onFinish"]
        assert roots[0]["child"][1]["decision"] == {"action": "pass"}
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("scope", ["edge", "node"])
async def test_non_fetch_python_tool_controls_keep_concurrent_callers_isolated(monkeypatch, scope):
    nodes = [{"id": "input", "type": "user_input", "data": {}},
             {"id": "compute", "type": "python_exec", "data": {
                 "code": "def run(handler): return {'echo': handler['query']}",
                 "tool_mode": True, "tool_name": "browse_primary", "safety_mode": "restricted_builtins",
                 "tool_parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
             }}]
    edges = []
    for index in (1, 2):
        client_id, llm_id, end_id = f"client-{index}", f"llm-{index}", f"end-{index}"
        nodes.extend([{"id": client_id, "type": "client", "data": {"engine": "openai", "model": "local-scripted-model"}},
                      {"id": llm_id, "type": "llm", "data": {"stream": False}},
                      {"id": end_id, "type": "end", "data": {}}])
        edges.extend([edge(f"input-{index}", "input", llm_id, "handle_user_message", "handle_user_message"),
                      edge(f"client-{index}", client_id, llm_id, "handle-client-provider", "handle-client-provider"),
                      edge(f"tool-{index}", "compute", llm_id, "handle-tool-definition", "handle-tool-definition-0",
                           hook="prepare" if scope == "edge" and index == 1 else None),
                      edge(f"result-{index}", llm_id, end_id, "handle_generated_content", "handle_flow_input")])
    prepare = '''async def prepare(context, chat_log):
    import asyncio
    await asyncio.sleep(0)
    return {"action": "input", "content": {"query": "prepared:" + context.caller["node_id"]}}
'''
    finish = '''async def finish(context, chat_log):
    import json
    if context.outcome["status"] != "success":
        return {"action": "pass"}
    content = context.outcome["content"]
    if isinstance(content, str):
        content = json.loads(content)
    return {"action": "outcome", "outcome": {"status": "success", "content": {
        **content, "caller": context.caller["node_id"]}}}
'''
    prepare_data = {"function_template": prepare, "lifecycle_event": "onStart"}
    if scope == "node":
        prepare_data["target_node_id"] = "compute"
    nodes.extend([{"id": "prepare", "type": "hook", "data": prepare_data},
                  {"id": "finish", "type": "hook", "data": {"function_template": finish,
                                                               "lifecycle_event": "onFinish", "target_node_id": "compute"}}])
    graph, clients = build_scripted(monkeypatch, {"type": "graph", "debug": True, "timeout": 5,
                                                "nodes": nodes, "edges": edges}, callers=2)
    rendezvous = {"started": 0, "entered": asyncio.Event()}
    for client in clients.values():
        client.llm.rendezvous = rendezvous
    await asyncio.wait_for(collect(graph), 3)
    assert rendezvous["started"] == 2
    one, two = tool_payload(clients["client-1"]), tool_payload(clients["client-2"])
    assert one == {"echo": "prepared:llm-1", "caller": "llm-1"}
    assert two == {"echo": "prepared:llm-2" if scope == "node" else "same query", "caller": "llm-2"}
    first = root_record(graph, "compute", "original-call-1")
    second = root_record(graph, "compute", "original-call-2")
    assert first["input"] == second["input"] == {"query": "same query"}
    assert first["id"] != second["id"]
    assert first["caller"]["tool_call_id"] == "original-call-1"
    assert second["caller"]["tool_call_id"] == "original-call-2"
    assert [frame["node_id"] for frame in second["child"]] == (["prepare", "finish"] if scope == "node" else ["finish"])


def argument_transform(phase, suffix):
    return f'''async def transform(context, chat_log):
    assert context.event["event"] == {phase!r}
    return {{"action": "input", "content": {{"query": context.request["content"]["query"] + {suffix!r}}}}}
'''


def hook_node(identifier, phase, suffix, *, target=None):
    data = {"lifecycle_event": phase, "function_template": argument_transform(phase, suffix)}
    if target:
        data["target_node_id"] = target
    return {"id": identifier, "type": "hook", "data": data}


def assert_tool_delivery_phases(graph, events, root, subject, delivery_hook, start_hook=None):
    deliveries = [frame for frame in subject["child"] if frame.get("operation") == "delivery"]
    assert len(deliveries) == 1
    delivery, = deliveries
    assert delivery["node_id"] == subject["node_id"]
    assert delivery["parent_id"] == subject["id"] and delivery["root_id"] == root["id"]
    assert delivery["id"] != subject["id"]
    assert "tool_call_id" not in delivery["caller"]
    assert [hook["node_id"] for hook in delivery["child"]] == [delivery_hook]
    assert [hook["event"] for hook in delivery["child"]] == ["onDeliver"]
    admission = [frame for frame in subject["child"] if frame["kind"] == "hook" and frame["event"] == "onStart"]
    assert [hook["node_id"] for hook in admission] == ([start_hook] if start_hook else [])
    if admission:
        assert subject["child"].index(delivery) < subject["child"].index(admission[0])
    facts = graph.nodes[subject["node_id"]]._invocation_control.events
    assert [fact["event"] for fact in facts if fact["frame_id"] == delivery["id"]] == ["onDeliver", "onFinish"]
    assert [fact["event"] for fact in facts if fact["frame_id"] == subject["id"]] == ["onStart", "onFinish"]
    # The public debug trace includes both identities and their owned Hooks.
    published = [event["content"]["data"]["execution"] for event in events
                 if event.get("type") == "debug" and event.get("content", {}).get("event_type") == "HOOK_RESULT"
                 and event["content"].get("data", {}).get("execution", {}).get("id") == root["id"]]
    assert published == [root]
    results = [event for event in events if event.get("type") == "debug"
               and event.get("content", {}).get("event_type") == "TOOL_RESULT"]
    assert len(results) == 1 and results[0]["content"]["data"]["tool_call_id"] == "original-call-1"
    return delivery


async def test_tool_edge_delivery_transforms_arguments_before_node_start_and_real_http(monkeypatch):
    async with http_providers({"a": 200, "b": 200, "c": 200}) as (server, calls, queries, _, __):
        data = fallback_graph(server, scope="node")
        data["nodes"].extend([
            hook_node("tool-deliver", "onDeliver", "|edge-delivery"),
            hook_node("tool-start", "onStart", "|node-start", target="fetch-a"),
        ])
        next(item for item in data["edges"] if item["id"] == "tool-1")["hooks"] = {
            "enabled": True, "hook_node_id": "tool-deliver"}
        graph, clients = build_scripted(monkeypatch, data)
        events = await asyncio.wait_for(collect(graph), 3)
    payload, root = assert_original_provider_result(graph, clients["client-1"], "a")
    assert calls == ["a"] and queries == ["same query|edge-delivery|node-start"]
    assert payload["query"] == queries[0]
    assert root["input"] == {"query": "same query"}
    assert root["request"]["content"] == {"query": queries[0]}
    delivery = assert_tool_delivery_phases(graph, events, root, root, "tool-deliver", "tool-start")
    assert delivery["outcome"] == {"status": "success", "content": {"query": "same query|edge-delivery"}}


async def test_node_scoped_tool_delivery_runs_without_attached_edge_hook(monkeypatch):
    async with http_providers({"a": 200, "b": 200, "c": 200}) as (server, calls, queries, _, __):
        data = fallback_graph(server, scope="node")
        data["nodes"].append(hook_node("node-deliver", "onDeliver", "|node-delivery", target="fetch-a"))
        assert not next(item for item in data["edges"] if item["id"] == "tool-1").get("hooks")
        graph, clients = build_scripted(monkeypatch, data)
        events = await asyncio.wait_for(collect(graph), 3)
    payload, root = assert_original_provider_result(graph, clients["client-1"], "a")
    assert calls == ["a"] and queries == ["same query|node-delivery"]
    assert payload["query"] == queries[0]
    assert root["input"] == {"query": "same query"}
    assert root["request"]["content"] == {"query": queries[0]}
    assert_tool_delivery_phases(graph, events, root, root, "node-deliver")


async def test_declared_hook_child_edge_delivery_precedes_child_start_and_real_http(monkeypatch):
    async with http_providers({"a": 429, "b": 200, "c": 200}) as (server, calls, queries, _, __):
        data = fallback_graph(server)
        data["nodes"].extend([
            hook_node("child-deliver", "onDeliver", "|child-delivery"),
            hook_node("child-start", "onStart", "|child-start", target="fetch-b"),
        ])
        next(item for item in data["edges"] if item["id"] == "a-hook-to-b")["hooks"] = {
            "enabled": True, "hook_node_id": "child-deliver"}
        graph, clients = build_scripted(monkeypatch, data)
        events = await asyncio.wait_for(collect(graph), 3)
    payload, root = assert_original_provider_result(graph, clients["client-1"], "b")
    assert calls == ["a", "b"] and queries == ["same query", "same query|child-delivery|child-start"]
    assert payload["query"] == queries[1]
    child = next(frame for frame in walk(root) if frame["node_id"] == "fetch-b" and frame.get("operation") != "delivery")
    assert child["caller"] == {"node_id": "recover-a", "edge_id": "a-hook-to-b"}
    assert child["input"] == {"query": "same query"}
    assert child["request"]["content"] == {"query": queries[1]}
    assert_tool_delivery_phases(graph, events, root, child, "child-deliver", "child-start")


def local_studio_browsing_catalog():
    # This cross-package check uses the Studio's actual generated graph and
    # Hook code. Standalone magic-agents checkouts may omit the Studio package.
    configured = os.environ.get("MAGIC_UI_REPO")
    roots = ([Path(configured)] if configured else []) + [Path.home() / "WebstormProjects" / "magic-ui"]
    for root in roots:
        source = root / "src/App/Docs/catalog/examples/browsing-provider-fallback.json"
        if source.is_file():
            return json.loads(source.read_text())["graph"]
    pytest.skip("Studio browsing catalog requires MAGIC_UI_REPO or the sibling magic-ui checkout")


def catalog_fallback_chain(data):
    """Provider Fetch ids in fallback order, derived from the catalog graph.

    The primary is the Fetch wired to the LLM as a tool; each next provider is
    the child-call target of the onError Hook bound to the previous one.
    Provider names are not hard-coded so the docs catalog can change them.
    """
    fetch_ids = {node["id"] for node in data["nodes"] if node["type"] == "fetch"}
    hooks = {node["id"]: node["data"] for node in data["nodes"] if node["type"] == "hook"}
    tool_edges = [item for item in data["edges"] if item["source"] in fetch_ids
                  and item["targetHandle"].startswith("handle-tool-definition")]
    assert len(tool_edges) == 1
    chain = [tool_edges[0]["source"]]
    while True:
        current = chain[-1]
        bound = [hook_id for hook_id, hook in hooks.items() if hook.get("lifecycle_event") == "onError"
                 and (hook.get("target_node_id") == current or any(
                     item["source"] == current and (item.get("hooks") or {}).get("hook_node_id") == hook_id
                     for item in data["edges"]))]
        following = [item["target"] for item in data["edges"] if item["source"] in bound
                     and item["sourceHandle"] == "handle-child-call" and item["target"] in fetch_ids]
        if not following:
            return chain
        assert len(following) == 1 and following[0] not in chain
        chain.append(following[0])


@pytest.mark.parametrize("second_status,final_index", [(200, 1), (401, 2)])
async def test_studio_catalog_explicit_policy_recovers_http401_with_one_original_tool_result(monkeypatch, second_status, final_index):
    data = local_studio_browsing_catalog()
    chain = catalog_fallback_chain(data)
    assert len(chain) == 3
    primary = chain[0]
    statuses = {chain[0]: 401, chain[1]: second_status, chain[2]: 200}
    async with http_providers(statuses) as (server, calls, queries, _, __):
        # Only connection endpoints/credentials and the model transport become
        # local fixtures; the exported Hook decisions and child bindings stay
        # exactly those authored in the Studio catalog.
        for node in data["nodes"]:
            node["data"].pop("customName", None)  # Studio runtime export omits display names.
            if node["type"] == "fetch":
                node["data"]["url"] = str(server.make_url("/" + node["id"]))
                node["data"]["headers"] = {"Authorization": "Bearer INVALID_TOKEN}" if node["id"] == primary else "Bearer local-test-key"}
                if node["id"] == primary:
                    node["data"]["tool_name"] = "browse_primary"
            elif node["type"] == "llm":
                node["data"]["stream"] = False
            elif node["type"] == "client":
                node["id"] = "client-1"
                node["data"]["api_info"] = {"api_key": "local-test-unused"}
        for connection in data["edges"]:
            if connection["source"] == "client":
                connection["source"] = "client-1"
        graph, clients = build_scripted(monkeypatch, data)
        events = await asyncio.wait_for(collect(graph), 3)
    expected = chain[:final_index + 1]
    assert calls == expected and queries == ["same query"] * len(expected)
    payload = tool_payload(clients["client-1"])
    assert payload["provider"] == chain[final_index] and "execution" not in payload
    root = root_record(graph, primary, "original-call-1")
    assert root["original_outcome"]["error"]["code"] == "HTTP_ERROR"
    assert root["original_outcome"]["error"]["details"]["status_code"] == 401
    assert root["original_outcome"]["error"]["retryable"] is False
    assert root["outcome"] == {"status": "success", "content": payload} and root["recovered"]
    attempts = [frame for frame in walk(root) if frame["kind"] == "node"]
    assert [attempt["node_id"] for attempt in attempts] == expected
    for attempt in attempts[:-1]:
        assert attempt["original_outcome"]["error"]["details"]["status_code"] == 401
    assert root["caller"]["tool_call_id"] == "original-call-1"
    assert all("tool_call_id" not in attempt["caller"] for attempt in attempts[1:])
    published = [event["content"]["data"]["execution"] for event in events
                 if event.get("type") == "debug" and event.get("content", {}).get("event_type") == "HOOK_RESULT"
                 and event["content"].get("data", {}).get("execution", {}).get("id") == root["id"]]
    assert published == [root]
    results = [event for event in events if event.get("type") == "debug"
               and event.get("content", {}).get("event_type") == "TOOL_RESULT"]
    assert len(results) == 1 and results[0]["content"]["data"]["tool_call_id"] == "original-call-1"
