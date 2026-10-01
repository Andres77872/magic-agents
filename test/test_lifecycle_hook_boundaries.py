"""Reliability boundaries on the real graph executor and existing node classes.

There is no second Hook engine in this suite. The model transport is scripted
only in the observer/tool test; production graph, Hook, LLM loop, and Python
tool execution perform the operations under test.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
import json

import pytest

from magic_agents.agt_flow import build, run_agent
from magic_agents.hooks.flow_hooks import FlowHooks
from magic_agents.hooks.runtime_config import RuntimeConfig
from magic_agents.node_system.NodeClientLLM import NodeClientLLM
from magic_agents.node_system.NodeHook import NodeHook
from magic_agents.node_system.NodeParser import NodeParser
from magic_llm import MagicLLM
from magic_llm.model.ModelChatResponse import Choice, FunctionCall, Message, ModelChatResponse, ToolCall


pytestmark = pytest.mark.asyncio


PASS = 'async def control(context, chat_log): return {"action": "pass"}'


def edge(identifier, source, target, source_handle, target_handle):
    return {"id": identifier, "source": source, "target": target,
            "sourceHandle": source_handle, "targetHandle": target_handle}


def parser_graph(template=PASS, *, failure_policy="preserve", child=False, timeout=None):
    hook_data = {"function_template": template, "lifecycle_event": "onStart",
                 "target_node_id": "target", "failure_policy": failure_policy}
    if timeout is not None:
        hook_data["timeout_override"] = timeout
    nodes = [{"id": "input", "type": "user_input"},
             {"id": "target", "type": "parser", "data": {"text": "core:{{ value }}"}},
             {"id": "control", "type": "hook", "data": hook_data},
             {"id": "end", "type": "end"}]
    edges = [edge("input-target", "input", "target", "handle_user_message", "value"),
             edge("target-end", "target", "end", "handle_parser_output", "handle_flow_input")]
    if child:
        nodes.append({"id": "child", "type": "parser", "data": {"text": "child:{{ value }}"}})
        edges.append(edge("control-child", "control", "child", "handle-child-call", "value"))
    return build({"type": "graph", "debug": True, "timeout": 4,
                  "nodes": nodes, "edges": edges}, message="original")


async def collect(graph, **kwargs):
    return [event async for event in run_agent(graph, **kwargs)]


def published_traces(events, node_id):
    return [event["content"]["data"]["execution"] for event in events
            if event.get("type") == "debug"
            and event.get("content", {}).get("event_type") == "HOOK_RESULT"
            and event["content"].get("data", {}).get("execution", {}).get("node_id") == node_id]


def root_record(graph, node_id):
    records = [record for record in graph.nodes[node_id]._invocation_control.records
               if record["kind"] == "node" and record["node_id"] == node_id and record["parent_id"] is None]
    assert len(records) == 1
    return records[0]


async def test_hook_timeout_includes_redirect_child_and_joins_its_operation(monkeypatch):
    template = '''async def control(context, chat_log):
    return {"action": "redirect", "connection": "control-child", "content": "attempt"}
'''
    graph = parser_graph(template, child=True, timeout=1)
    timeline = []
    original = NodeParser.process

    async def observed_process(self, chat_log):
        if self.node_id == "child":
            timeline.append("child started")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                timeline.append("child cancelled")
                raise
            finally:
                timeline.append("child joined")
        else:
            timeline.append("original core")
            async for event in original(self, chat_log):
                yield event

    monkeypatch.setattr(NodeParser, "process", observed_process)
    events = await asyncio.wait_for(collect(graph), 3)
    assert timeline == ["child started", "child cancelled", "child joined", "original core"]
    assert graph.nodes["end"].inputs["handle_flow_input"] == "core:original"
    root = root_record(graph, "target")
    hook, = root["child"]
    assert hook["diagnostic"]["code"] == "HOOK_TIMEOUT"
    assert hook["child"][0]["outcome"]["status"] in ("error", "cancelled")
    assert root["outcome"]["status"] == "success" and not root["recovered"]
    assert published_traces(events, "target") == [root]


async def test_hook_that_swallows_timeout_cancellation_cannot_commit_late_input():
    template = '''async def control(context, chat_log):
    import asyncio
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        return {"action": "input", "content": {"value": "late replacement"}}
'''
    graph = parser_graph(template, timeout=1)
    events = await asyncio.wait_for(collect(graph), 3)
    root = root_record(graph, "target")
    assert root["child"][0]["diagnostic"]["code"] == "HOOK_TIMEOUT"
    assert "decision" not in root["child"][0]
    assert root["request"]["content"] == root["input"] == {"value": "original"}
    assert graph.nodes["end"].inputs["handle_flow_input"] == "core:original"
    assert published_traces(events, "target") == [root]


@pytest.mark.parametrize("reply", [
    {"status": "success", "content": "raw result"},
    {"action": "pass", "content": "unexpected field"},
    {"action": "outcome", "outcome": {"status": "error", "error": "untyped failure"}},
])
@pytest.mark.parametrize("failure_policy", ["preserve", "fail"])
async def test_invalid_hook_decisions_have_explicit_preserve_or_fail_behavior(reply, failure_policy):
    template = "async def control(context, chat_log):\n    return " + repr(reply)
    graph = parser_graph(template, failure_policy=failure_policy)
    events = await asyncio.wait_for(collect(graph), 2)
    root = root_record(graph, "target")
    hook, = root["child"]
    assert hook["outcome"]["status"] == "error" and hook["diagnostic"]["code"] == "HOOK_FAILED"
    assert "decision" not in hook
    assert published_traces(events, "target") == [root]
    if failure_policy == "preserve":
        assert root["executed"] and root["outcome"]["status"] == "success"
        assert graph.nodes["end"].inputs["handle_flow_input"] == "core:original"
    else:
        assert not root["executed"] and root["outcome"]["error"]["code"] == "HOOK_FAILED"
        assert root["original_outcome"] == root["outcome"]
        assert "handle_flow_input" not in graph.nodes["end"].inputs


async def test_unawaited_owned_child_is_cancelled_and_cannot_write_after_parent_commit(monkeypatch):
    # A fixture barrier enters the authored namespace solely to synchronize the
    # test. The Hook still compiles and runs through NodeHook.invoke_control.
    entered = asyncio.Event()
    release = asyncio.Event()
    timeline = []
    template = '''async def control(context, chat_log):
    import asyncio
    asyncio.create_task(context.call("control-child", "attempt"))
    await child_started.wait()
    return {"action": "pass"}
'''
    compile_hook = NodeHook._compile_hook_function

    def compile_with_barrier(self, source):
        function = compile_hook(self, source)
        if self.node_id == "control" and function is not None:
            function.__globals__["child_started"] = entered
        return function

    monkeypatch.setattr(NodeHook, "_compile_hook_function", compile_with_barrier)
    original = NodeParser.process

    async def observed_process(self, chat_log):
        if self.node_id == "child":
            entered.set()
            timeline.append("child entered")
            try:
                await release.wait()
                timeline.append("late write")
                yield self.yield_static("late result", content_type=self.OUTPUT_HANDLE)
            except asyncio.CancelledError:
                timeline.append("child cancelled")
                raise
            finally:
                timeline.append("child joined")
        else:
            timeline.append("parent core")
            async for event in original(self, chat_log):
                yield event

    monkeypatch.setattr(NodeParser, "process", observed_process)
    graph = parser_graph(template, child=True)
    events = await asyncio.wait_for(collect(graph), 2)
    assert timeline == ["child entered", "child cancelled", "child joined", "parent core"]
    root = root_record(graph, "target")
    owned_child = root["child"][0]["child"][0]
    assert owned_child["outcome"] == {"status": "cancelled"}
    before = deepcopy(root)
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert root == before and "late write" not in timeline
    assert published_traces(events, "target") == [root]
    assert graph.nodes["end"].inputs["handle_flow_input"] == "core:original"


async def test_generic_input_replacement_removes_omitted_json_handle(monkeypatch):
    graph = parser_graph('''async def control(context, chat_log):
    return {"action": "input", "content": {"value": "replacement"}}
''')
    graph.nodes["target"].inputs["omitted"] = "must be removed"
    core_inputs = []
    original = NodeParser.process

    async def observed_process(self, chat_log):
        if self.node_id == "target":
            core_inputs.append(deepcopy(self.inputs))
        async for event in original(self, chat_log):
            yield event

    monkeypatch.setattr(NodeParser, "process", observed_process)
    events = await asyncio.wait_for(collect(graph), 2)
    root = root_record(graph, "target")
    assert root["input"] == {"value": "original", "omitted": "must be removed"}
    assert root["request"]["content"] == {"value": "replacement"}
    assert core_inputs == [{"value": "replacement"}]
    assert graph.nodes["end"].inputs["handle_flow_input"] == "core:replacement"
    assert published_traces(events, "target") == [root]


class RecordingFlowHooks(FlowHooks):
    def __init__(self):
        self.node_events = []
        self.llm_events = []

    async def on_node_start(self, context):
        self.node_events.append(("start", context.node_id))

    async def on_node_end(self, context):
        self.node_events.append(("end", context.node_id))

    async def on_llm_start(self, context, llm_config=None):
        self.llm_events.append("start")

    async def on_llm_end(self, context, response=None):
        self.llm_events.append("end")


class ScriptedEngine:
    engine = engine_name = "openai"
    model = "local-scripted-model"

    def __init__(self):
        self.chats = []

    async def async_generate(self, chat, **kwargs):
        self.chats.append(deepcopy(chat.messages))
        if len(self.chats) == 1:
            message = Message(role="assistant", content=None, tool_calls=[ToolCall(
                id="original-call", function=FunctionCall(name="compute", arguments='{"query":"private"}'))])
            reason = "tool_calls"
        else:
            assert len(self.chats) == 2
            tool_messages = [message for message in chat.messages if message.get("role") == "tool"]
            assert len(tool_messages) == 1 and tool_messages[0]["tool_call_id"] == "original-call"
            assert json.loads(tool_messages[0]["content"]) == {"visible": "approved"}
            message = Message(role="assistant", content="answer")
            reason = "stop"
        return ModelChatResponse(id="local", object="chat.completion", created=0, model=self.model,
                                 choices=[Choice(index=0, message=message, finish_reason=reason)])


class ScriptedClient:
    run_agent_async = MagicLLM.run_agent_async
    _task_executor = None

    def __init__(self):
        self.llm = ScriptedEngine()


async def test_controlled_nodes_keep_observers_once_and_publish_tool_and_node_traces(monkeypatch):
    monkeypatch.setenv("DEBUG_ENABLED", "true")
    client = ScriptedClient()

    def initialize(self, engine, model):
        self.client = client
        self.init_error = self.init_error_type = None
        self._current_engine, self._current_model = engine, model

    monkeypatch.setattr(NodeClientLLM, "_initialize_client", initialize)
    nodes = [{"id": "input", "type": "user_input"},
             {"id": "parser", "type": "parser", "data": {"text": "{{ value }}"}},
             {"id": "client", "type": "client", "data": {"engine": "openai", "model": "local-scripted-model"}},
             {"id": "llm", "type": "llm", "data": {"stream": False}},
             {"id": "compute", "type": "python_exec", "data": {
                 "code": "def run(handler): return {'private': handler['query']}",
                 "tool_mode": True, "tool_name": "compute", "safety_mode": "restricted_builtins",
                 "tool_parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
             {"id": "end", "type": "end"}]
    for node_id in ("parser", "llm"):
        for event in ("onStart", "onFinish"):
            nodes.append({"id": node_id + event, "type": "hook", "data": {
                "lifecycle_event": event, "target_node_id": node_id, "function_template": PASS}})
    nodes.append({"id": "approve", "type": "hook", "data": {
        "lifecycle_event": "onFinish", "target_node_id": "compute", "function_template": '''async def approve(context, chat_log):
    if context.outcome["status"] != "success":
        return {"action": "pass"}
    return {"action": "outcome", "outcome": {"status": "success", "content": {"visible": "approved"}}}
'''}})
    edges = [edge("input-parser", "input", "parser", "handle_user_message", "value"),
             edge("parser-llm", "parser", "llm", "handle_parser_output", "handle_user_message"),
             edge("client-llm", "client", "llm", "handle-client-provider", "handle-client-provider"),
             edge("compute-llm", "compute", "llm", "handle-tool-definition", "handle-tool-definition-0"),
             edge("llm-end", "llm", "end", "handle_generated_content", "handle_flow_input")]
    graph = build({"type": "graph", "debug": True, "timeout": 4,
                   "nodes": nodes, "edges": edges}, message="request")
    flow_hooks = RecordingFlowHooks()
    observer_events = []

    async def record_observer(event):
        observer_events.append(event)

    try:
        events = await asyncio.wait_for(collect(graph, hooks=RuntimeConfig([flow_hooks]),
                                               debug_callback=record_observer), 3)
    except TimeoutError:
        raise AssertionError({"turns": len(client.llm.chats), "flow_hooks": flow_hooks.node_events,
                              "observers": [(event.event_type.value, event.node_id) for event in observer_events],
                              "inputs": {node: list(instance.inputs) for node, instance in graph.nodes.items()},
                              "records": [(frame["node_id"], frame["outcome"]) for frame in graph.nodes["llm"]._invocation_control.records]})
    assert graph.nodes["end"].inputs["handle_flow_input"] == "answer"
    for node_id in ("parser", "llm"):
        assert [kind for kind, node in flow_hooks.node_events if node == node_id] == ["start", "end"]
        assert Counter(event.event_type.value for event in observer_events
                       if event.node_id == node_id and event.event_type.value in ("node_start", "node_end")) == {
                           "node_start": 1, "node_end": 1}
        assert published_traces(events, node_id) == [root_record(graph, node_id)]
    # Existing per-provider observers remain active on both real LLM turns.
    assert Counter(flow_hooks.llm_events) == {"start": 2, "end": 2}
    tool_root = root_record(graph, "compute")
    original = tool_root["original_outcome"]["content"]
    if isinstance(original, str):
        original = json.loads(original)
    assert original == {"private": "private"}
    assert tool_root["outcome"] == {"status": "success", "content": {"visible": "approved"}}
    assert tool_root["caller"]["tool_call_id"] == "original-call"
    assert published_traces(events, "compute") == [tool_root]
    provider_tool = next(message for message in client.llm.chats[1] if message.get("role") == "tool")
    assert "private" not in provider_tool["content"] and "execution" not in provider_tool["content"]
