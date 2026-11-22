"""Debug SSE frames carry the timing and data the run debugger needs.

Covers the frame additions made for the playground run trace:
- ``t_ms`` on every frame: monotonic ms since the root graph started, shared
  by child graphs, taken at event time even when hooks run later (HookRelay).
- ``duration_ms`` on node_error, counts on graph_end / graph_error.
- ``tool_call_id`` / ``execution_time_ms`` / error details on tool_end.
- Per-call usage on llm_end inside a streamed tool loop, or an explicit
  ``usage_reported: False`` instead of zeros.
- Executor failure frames that only echo an ``on_node_error`` are flagged
  ``lifecycle_reported`` so debug consumers do not report them twice.
"""
from __future__ import annotations

import asyncio
import copy
import json
import time
from types import SimpleNamespace

import pytest
from magic_llm import MagicLLM
from magic_llm.agent.types import AgentState, ToolResult
from magic_llm.model.ModelChatResponse import Choice, Message, ModelChatResponse, UsageModel
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

from magic_agents.agt_flow import build
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.hooks.context_factory import HookContextFactory
from magic_agents.hooks.debug_sse import DebugSSEHook
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.hooks.hook_relay import HookRelay
from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel
from magic_agents.node_system.NodeLLM import NodeLLM


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: dict) -> None:
        self.events.append(event)

    def of(self, event_type: str) -> list[dict]:
        return [event["content"] for event in self.events if event["event_type"] == event_type]


def node_context(**overrides):
    ctx = HookContextFactory.build_node_context(
        execution_id="exec", run_id="run", node_id=overrides.pop("node_id", "n1"), node_type="text",
    )
    for name, value in overrides.items():
        setattr(ctx, name, value)
    return ctx


def graph_context(**overrides):
    ctx = HookContextFactory.build_graph_context(execution_id="exec", run_id="run", metadata={"node_count": 5})
    for name, value in overrides.items():
        setattr(ctx, name, value)
    return ctx


# ── t_ms ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_frame_carries_t_ms_from_root_graph_start():
    sink = RecordingSink()
    hook = DebugSSEHook(sink=sink, id_chat="chat")

    await hook.on_graph_start(graph_context())
    await asyncio.sleep(0.01)
    await hook.on_node_start(node_context())
    await hook.on_llm_start(node_context())
    await hook.on_llm_end(node_context())
    await hook.on_llm_loop_end(node_context())
    await hook.on_tool_start(node_context())
    await hook.on_tool_end(node_context())
    await hook.on_node_bypass(node_context(node_id="n2"), reason="condition")
    await hook.on_node_error(node_context(node_id="n3"), ValueError("x"))
    await hook.on_node_end(node_context())
    await hook.on_graph_end(graph_context())

    times = [event["content"]["t_ms"] for event in sink.events]
    assert len(times) == 11
    assert times[0] == 0.0
    assert times[1] >= 9.0, "offset must follow real elapsed time"
    assert times == sorted(times), "a monotonic clock never goes back"
    assert all(isinstance(value, float) for value in times)


@pytest.mark.asyncio
async def test_child_graphs_share_the_root_origin_and_a_new_root_run_restarts_it():
    sink = RecordingSink()
    root = DebugSSEHook(sink=sink, id_chat="chat")
    await root.on_graph_start(graph_context())
    await asyncio.sleep(0.01)

    child = root.fork_for_child(child_run_id="child-run", parent_node_id="inner")
    await child.on_graph_start(graph_context())
    await child.on_node_start(node_context(node_id="leaf"))
    child_start, leaf_start = sink.events[1]["content"]["t_ms"], sink.events[2]["content"]["t_ms"]
    assert child_start >= 9.0, "a child graph start must not reset the run clock"
    assert leaf_start >= child_start

    await root.on_graph_start(graph_context())
    assert sink.events[-1]["content"]["t_ms"] == 0.0
    await child.on_node_end(node_context(node_id="leaf"))
    assert sink.events[-1]["content"]["t_ms"] < child_start, "forks follow the root's restarted clock"


@pytest.mark.asyncio
async def test_event_time_recorded_on_the_context_wins_over_dispatch_time():
    sink = RecordingSink()
    hook = DebugSSEHook(sink=sink, id_chat="chat")
    await hook.on_graph_start(graph_context())

    happened = time.perf_counter()
    await asyncio.sleep(0.02)  # the hook runs late, as a HookRelay task does
    await hook.on_tool_start(node_context(monotonic_time=happened))
    await hook.on_tool_end(node_context())

    started, ended = (event["content"]["t_ms"] for event in sink.events[1:])
    assert ended - started >= 19.0


# ── Node / graph / tool / llm fields ────────────────────────────────────────


@pytest.mark.asyncio
async def test_node_error_reports_duration():
    sink = RecordingSink()
    await DebugSSEHook(sink=sink, id_chat="chat").on_node_error(node_context(duration_ms=41.5), KeyError("k"))
    assert sink.of("node_error")[0]["duration_ms"] == 41.5


@pytest.mark.asyncio
@pytest.mark.parametrize("method,kwargs", [("on_graph_end", {}), ("on_graph_error", {"error": RuntimeError("1 node error(s)")})])
async def test_graph_terminal_frames_report_execution_counts(method, kwargs):
    sink = RecordingSink()
    ctx = graph_context(duration_ms=250.0)
    ctx.metadata["execution_summary"] = {"total": 6, "completed": 3, "bypassed": 2, "errors": 1, "pending": 0, "states": {}}
    await getattr(DebugSSEHook(sink=sink, id_chat="chat"), method)(ctx, **kwargs)

    content = sink.events[0]["content"]
    assert sink.events[0]["type"] == "debug_summary"
    assert content["duration_ms"] == 250.0
    assert (content["node_count"], content["executed_count"], content["bypassed_count"], content["failed_count"]) == (6, 3, 2, 1)


@pytest.mark.asyncio
async def test_graph_error_without_summary_keeps_its_original_fields():
    sink = RecordingSink()
    await DebugSSEHook(sink=sink, id_chat="chat").on_graph_error(graph_context(), RuntimeError("boom"))
    content = sink.events[0]["content"]
    assert content["error_type"] == "RuntimeError" and content["error_message"] == "boom"
    assert "executed_count" not in content


@pytest.mark.asyncio
async def test_tool_end_pairs_by_call_id_and_reports_timing_and_failure():
    sink = RecordingSink()
    hook = DebugSSEHook(sink=sink, id_chat="chat")
    ok = node_context(outputs={"tool_name": "search", "tool_call_id": "call-1", "success": True, "execution_time_ms": 12.0})
    failed = node_context(
        outputs={"tool_name": "search", "tool_call_id": "call-2", "success": False, "execution_time_ms": 3.5},
        error_message="rate limited", error_type="HTTPError",
    )
    await hook.on_tool_end(ok)
    await hook.on_tool_end(failed)

    first, second = sink.of("tool_end")
    assert (first["tool_call_id"], first["execution_time_ms"], first["success"]) == ("call-1", 12.0, True)
    assert "error_message" not in first
    assert (second["tool_call_id"], second["success"]) == ("call-2", False)
    assert (second["error_type"], second["error_message"]) == ("HTTPError", "rate limited")


@pytest.mark.asyncio
async def test_tool_end_falls_back_to_the_start_inputs_for_the_call_id():
    sink = RecordingSink()
    await DebugSSEHook(sink=sink, id_chat="chat").on_tool_end(
        node_context(inputs={"tool_name": "search", "tool_call_id": "call-9"}, outputs={"success": True}))
    assert sink.of("tool_end")[0]["tool_call_id"] == "call-9"


@pytest.mark.asyncio
@pytest.mark.parametrize("outputs,reported", [
    ({"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}, True),
    ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, False),
    ({}, False),
    ({"prompt_tokens": None, "usage_reported": False}, False),
])
async def test_llm_end_says_whether_usage_was_reported(outputs, reported):
    sink = RecordingSink()
    await DebugSSEHook(sink=sink, id_chat="chat").on_llm_end(node_context(outputs=outputs))
    assert sink.of("llm_end")[0]["usage_reported"] is reported


# ── HookRelay ───────────────────────────────────────────────────────────────


class Captured:
    def __init__(self):
        self.contexts = {}

    def on_llm_end(self, context):
        self.contexts.setdefault("on_llm_end", []).append(context)

    def on_tool_end(self, context):
        self.contexts.setdefault("on_tool_end", []).append(context)


def stream_chunk(usage=None):
    chunk = ChatCompletionModel(id="chunk", model="m", choices=[ChoiceModel(delta=DeltaModel(content="x"))])
    if usage is not None:
        chunk.usage = UsageModel(**usage)
    return chunk


def synthetic_stream_response():
    """What magic-llm's streamed loop hands on_llm_response: default usage."""
    return ModelChatResponse(id="chunk", object="chat.completion", created=0, model="m",
                             choices=[Choice(index=0, message=Message(role="assistant", content="x"), finish_reason="stop")])


def test_streamed_call_usage_comes_from_its_chunks():
    hooks = Captured()
    relay = HookRelay(flow_hooks=hooks, node_id="llm", llm_config={"streaming": True})
    state = AgentState()

    relay.on_iteration_start(0, state)
    relay.observe_stream_chunk(stream_chunk())  # a content chunk with default zero usage
    relay.observe_stream_chunk(stream_chunk({"prompt_tokens": 300, "completion_tokens": 7, "total_tokens": 307, "tps": 40.0}))
    relay.observe_stream_chunk(stream_chunk())
    relay.on_llm_response(synthetic_stream_response(), state)

    relay.on_iteration_start(1, state)
    relay.observe_stream_chunk(stream_chunk({"prompt_tokens": 368, "completion_tokens": 22, "total_tokens": 390}))
    relay.on_llm_response(synthetic_stream_response(), state)

    first, second = hooks.contexts["on_llm_end"]
    assert (first.outputs["prompt_tokens"], first.outputs["completion_tokens"], first.outputs["usage_reported"]) == (300, 7, True)
    assert first.outputs["raw_usage_json"]["tps"] == 40.0
    assert (second.outputs["prompt_tokens"], second.outputs["total_tokens"]) == (368, 390)
    # The request id used for tool/persistence correlation is unchanged.
    assert first.outputs["provider_request_id"] == "chunk"
    assert first.inputs["streaming"] is True


def test_cumulative_stream_usage_keeps_the_highest_counts():
    hooks = Captured()
    relay = HookRelay(flow_hooks=hooks, node_id="llm")
    relay.on_iteration_start(0, AgentState())
    relay.observe_stream_chunk(stream_chunk({"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51}))
    relay.observe_stream_chunk(stream_chunk({"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59}))
    relay.observe_stream_chunk(stream_chunk({"prompt_tokens": 0, "completion_tokens": 4, "total_tokens": 4}))
    relay.on_llm_response(synthetic_stream_response(), AgentState())
    outputs = hooks.contexts["on_llm_end"][0].outputs
    assert (outputs["prompt_tokens"], outputs["completion_tokens"], outputs["total_tokens"]) == (50, 9, 59)
    assert hooks.contexts["on_llm_end"][0].inputs["streaming"] is True


def test_unreported_usage_is_explicit_and_never_zero():
    hooks = Captured()
    relay = HookRelay(flow_hooks=hooks, node_id="llm", llm_config={"streaming": True})
    relay.on_iteration_start(0, AgentState())
    relay.observe_stream_chunk(stream_chunk())
    relay.on_llm_response(synthetic_stream_response(), AgentState())
    outputs = hooks.contexts["on_llm_end"][0].outputs
    assert outputs["usage_reported"] is False
    assert outputs["prompt_tokens"] is None and outputs["total_tokens"] is None
    assert outputs["raw_usage_json"] is None
    assert outputs["provider_request_id"] == "chunk"


def test_stream_usage_does_not_leak_into_the_next_call():
    hooks = Captured()
    relay = HookRelay(flow_hooks=hooks, node_id="llm")
    relay.on_iteration_start(0, AgentState())
    relay.observe_stream_chunk(stream_chunk({"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}))
    relay.on_iteration_start(1, AgentState())  # the call above never finished
    relay.on_llm_response(synthetic_stream_response(), AgentState())
    assert hooks.contexts["on_llm_end"][0].outputs["usage_reported"] is False


def test_provider_usage_on_the_response_still_wins():
    hooks = Captured()
    relay = HookRelay(flow_hooks=hooks, node_id="llm", llm_config={"streaming": False})
    response = synthetic_stream_response()
    response.usage = UsageModel(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    relay.on_iteration_start(0, AgentState())
    relay.on_llm_response(response, AgentState())
    ctx = hooks.contexts["on_llm_end"][0]
    assert (ctx.outputs["total_tokens"], ctx.outputs["usage_reported"]) == (7, True)
    assert ctx.inputs["streaming"] is False


def test_relay_contexts_record_when_the_event_happened():
    hooks = Captured()
    relay = HookRelay(flow_hooks=hooks, node_id="llm")
    before = time.perf_counter()
    relay.on_tool_complete(ToolResult(tool_call_id="c1", name="search", content="", is_error=True,
                                      error="rate limited", error_type="HTTPError", duration_ms=4.0), AgentState())
    ctx = hooks.contexts["on_tool_end"][0]
    assert before <= ctx.monotonic_time <= time.perf_counter()
    assert (ctx.error_type, ctx.error_message, ctx.outputs["tool_call_id"]) == ("HTTPError", "rate limited", "c1")


# ── Real streamed tool loop through NodeLLM ─────────────────────────────────


def scripted_response(calls=None, content=None, usage=(0, 0)):
    prompt, completion = usage
    return ModelChatResponse(id=f"req-{prompt}", object="chat.completion", created=0, model="scripted",
        choices=[Choice(index=0, message=Message(role="assistant", content=content, tool_calls=calls),
                        finish_reason="tool_calls" if calls else "stop")],
        usage=UsageModel(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion))


def tool_call(identifier, value):
    return {"id": identifier, "type": "function", "function": {"name": "double", "arguments": json.dumps({"value": value})}}


class StreamingEngine:
    """Streams each scripted response the way providers do: usage on the last chunk."""

    model = "scripted"
    engine_name = "openai"

    def __init__(self, steps):
        self.steps = iter(steps)

    async def async_stream_generate(self, chat, **kwargs):
        result = next(self.steps)
        message = result.choices[0].message
        calls = [c.model_dump() for c in (message.tool_calls or [])]
        for index, item in enumerate(calls):
            item["index"] = index
        yield ChatCompletionModel(id=result.id, model=result.model, choices=[ChoiceModel(index=0,
            delta=DeltaModel(content=message.content or "", tool_calls=calls or None),
            finish_reason=result.choices[0].finish_reason)])
        yield ChatCompletionModel(id=result.id, model=result.model,
                                  choices=[ChoiceModel(index=0, delta=DeltaModel(content=""))], usage=result.usage)


@pytest.mark.asyncio
async def test_streamed_tool_loop_reports_each_calls_usage_and_pairs_parallel_tools(monkeypatch):
    schema = {"type": "function", "function": {"name": "double", "description": "Double a value",
              "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}

    async def double(value):
        await asyncio.sleep(0.005 if value == 3 else 0.02)
        return value * 2

    provider = SimpleNamespace(tool_schema=schema, tool_callable=double)
    client = MagicLLM.__new__(MagicLLM)
    client.llm = StreamingEngine([
        scripted_response(calls=[tool_call("call-a", 3), tool_call("call-b", 4)], usage=(300, 7)),
        scripted_response(content="14", usage=(368, 22)),
    ])
    client._task_executor = None
    node = NodeLLM(data=LlmNodeModel(stream=True), node_id="reasoner", node_type="llm", debug=False)
    node.inputs = {"handle-client-provider": client, "handle_user_message": "compute", "handle-tool-0": provider}

    async def no_subagents(_):
        return SimpleNamespace(tool_schemas=[], registered_count=0)
    monkeypatch.setattr(node, "_load_subagents_if_enabled", no_subagents)

    sink = RecordingSink()
    registry = HookRegistry(global_hooks=[DebugSSEHook(sink=sink, id_chat="chat")])
    async for _ in node(None, hooks=registry):
        pass

    llm_ends = sink.of("llm_end")
    assert [(e["tokens"]["prompt_tokens"], e["tokens"]["completion_tokens"]) for e in llm_ends] == [(300, 7), (368, 22)]
    assert all(e["usage_reported"] for e in llm_ends)

    starts = {e["tool_call_id"]: e for e in sink.of("tool_start")}
    ends = {e["tool_call_id"]: e for e in sink.of("tool_end")}
    assert set(starts) == set(ends) == {"call-a", "call-b"}
    for call_id in ("call-a", "call-b"):
        assert ends[call_id]["success"] is True
        assert ends[call_id]["execution_time_ms"] > 0
        assert ends[call_id]["t_ms"] >= starts[call_id]["t_ms"]

    node_start, node_end = sink.of("node_start")[0], sink.of("node_end")[0]
    all_times = [event["content"]["t_ms"] for event in sink.events]
    assert node_start["t_ms"] == min(all_times) and node_end["t_ms"] == max(all_times)


# ── Executor failure frames vs node diagnostics ─────────────────────────────


def _graph(definition, message="hi"):
    return build(copy.deepcopy(definition), message=message)


async def _run(graph):
    sink = RecordingSink()
    registry = HookRegistry(global_hooks=[DebugSSEHook(sink=sink, id_chat="chat")])
    events = [event async for event in execute_graph_reactive(graph, hooks=registry)]
    frames = [e["content"] for e in events if e.get("type") == "debug" and e.get("content", {}).get("error_type")]
    return sink, frames


@pytest.mark.asyncio
async def test_node_diagnostics_are_not_flagged_but_failure_echoes_are(monkeypatch):
    definition = {
        "nodes": [
            {"id": "input", "type": "user_input"},
            {"id": "route", "type": "conditional", "data": {
                "evaluation_mode": "jinja", "condition": "{{ missing.required }}", "output_handles": ["a", "b"]}},
            {"id": "a", "type": "send_message", "data": {"message": "A"}},
            {"id": "b", "type": "send_message", "data": {"message": "B"}},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"id": "1", "source": "input", "target": "route", "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
            {"id": "2", "source": "route", "target": "a", "sourceHandle": "a", "targetHandle": "handle_send_extra"},
            {"id": "3", "source": "route", "target": "b", "sourceHandle": "b", "targetHandle": "handle_send_extra"},
            {"id": "4", "source": "a", "target": "end", "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            {"id": "5", "source": "b", "target": "end", "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
        ],
    }
    graph = _graph(definition)
    sink, frames = await _run(graph)
    diagnostic = next(frame for frame in frames if frame.get("node_id") == "route")
    assert diagnostic["error_type"] == "TemplateError"
    assert "missing" in diagnostic["error_message"]
    assert "lifecycle_reported" not in diagnostic

    # The run's terminal frame now says how many nodes did what.
    terminal = (sink.of("graph_end") or sink.of("graph_error"))[0]
    assert terminal["node_count"] == len(graph.nodes)
    assert terminal["failed_count"] == 1, "the Conditional's BYPASS_ALL marks it failed"
    assert terminal["bypassed_count"] >= 3

    # A node that raises is reported by on_node_error; the executor's echo is flagged.
    from magic_agents.node_system.NodeSendMessage import NodeSendMessage

    async def explode(self, chat_log):
        raise RuntimeError("send failed")
        yield  # pragma: no cover
    monkeypatch.setattr(NodeSendMessage, "process", explode)
    definition["nodes"][1]["data"]["condition"] = "a"
    sink, frames = await _run(_graph(definition))
    echo = next(frame for frame in frames if frame.get("node_id") == "a")
    assert echo["lifecycle_reported"] is True
    assert sink.of("node_error")[0]["error_message"] == "send failed"
    assert sink.of("node_error")[0]["duration_ms"] is not None
    assert sink.of("graph_error")[0]["failed_count"] == 1
