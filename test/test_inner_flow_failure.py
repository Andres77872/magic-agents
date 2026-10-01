"""Inner Flow failures: child node state decides, the Inner node fails typed.

A sub-flow fails when one of its nodes ends in ERROR (it raised, timed out
waiting for inputs or signalled BYPASS_ALL). The Inner Flow node then raises
OperationFailure('INNER_FLOW_FAILED'), so lifecycle Hooks on it see a typed
error and can recover it. A diagnostic frame alone (a node that still
completed) is not a failure.
"""
import asyncio

import pytest

from magic_agents.agt_flow import build, run_agent
from magic_agents.execution.reactive_executor import execute_graph_reactive
from magic_agents.hooks.flow_hooks import FlowHooks
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.node_system.NodeParser import NodeParser

pytestmark = pytest.mark.asyncio

FAIL_ON_BAD = "{% if value == 'bad' %}{{ missing.invalid() }}{% endif %}R-{{ value }}"
RECOVER = ('async def recover(context, chat_log):\n'
           '    context.emit.debug({"event": context.event["event"], "outcome": context.outcome})\n'
           '    return {"action":"outcome","outcome":{"status":"success","content":{"handle_execution_content":"RECOVERED"}}}')
OBSERVE = ('async def observe(context, chat_log):\n'
           '    context.emit.debug({"event": context.event["event"], "outcome": context.outcome})\n'
           '    return None')


def edge(identifier, source, target, source_handle, target_handle):
    return {"id": identifier, "source": source, "target": target,
            "sourceHandle": source_handle, "targetHandle": target_handle}


def parser_subflow(prefix, template, *, extra_nodes=(), extra_edges=(), timeout=None):
    """user_input -> parser(template) -> end."""
    flow = {"type": "graph", "nodes": [
        {"id": f"{prefix}-in", "type": "user_input"},
        {"id": f"{prefix}-parser", "type": "parser", "data": {"text": template}},
        {"id": f"{prefix}-end", "type": "end"},
        *extra_nodes,
    ], "edges": [
        edge(f"{prefix}-e1", f"{prefix}-in", f"{prefix}-parser", "handle_user_message", "value"),
        edge(f"{prefix}-e2", f"{prefix}-parser", f"{prefix}-end", "handle_parser_output", "handle_flow_input"),
        *extra_edges,
    ]}
    if timeout is not None:
        flow["timeout"] = timeout
    return flow


def top_graph(subflow, *, hooks=(), message="bad"):
    """input -> inner -> fmt(parser) -> end."""
    return build({"type": "graph", "timeout": 3, "nodes": [
        {"id": "input", "type": "user_input"},
        {"id": "inner", "type": "inner", "data": {"magic_flow": subflow}},
        {"id": "fmt", "type": "parser", "data": {"text": "GOT[{{ r }}]"}},
        {"id": "end", "type": "end"},
        *hooks,
    ], "edges": [
        edge("input-inner", "input", "inner", "handle_user_message", "handle_user_message"),
        edge("inner-fmt", "inner", "fmt", "handle_execution_content", "r"),
        edge("fmt-end", "fmt", "end", "handle_parser_output", "handle_flow_input"),
    ]}, message=message)


class Recorder(FlowHooks):
    def __init__(self):
        self.node_errors, self.node_ends, self.graph_errors, self.graph_ends = [], [], [], []

    async def on_node_error(self, context, error):
        self.node_errors.append((context.node_id, error))

    async def on_node_end(self, context):
        self.node_ends.append(context.node_id)

    async def on_graph_error(self, context, error):
        self.graph_errors.append(context.execution_id)

    async def on_graph_end(self, context):
        self.graph_ends.append(context.execution_id)


async def collect(graph, hooks=None):
    if hooks is None:
        return [event async for event in run_agent(graph)]
    return [event async for event in execute_graph_reactive(graph, hooks=hooks)]


def frames(events):
    return [event for event in events if event.get("type") == "debug" and isinstance(event.get("content"), dict)]


def error_frames(events):
    return [(event.get("source_node_path") or [event["content"].get("node_id")], event["content"])
            for event in frames(events) if event["content"].get("error_type")]


def subgraph_ends(events):
    return [event["content"]["status"] for event in frames(events)
            if event["content"].get("event_type") == "SUBGRAPH_END"]


def side_events(graph, node_id):
    record = graph.nodes[node_id]._last_invocation_record
    return [event["content"] for hook in record["child"] for event in hook.get("side_events", [])]


# ─── Result sink ──────────────────────────────────────────────────────────────

async def test_result_sink_reports_node_state_for_static_loop_and_blocked_runs():
    ok = build(parser_subflow("s", "R-{{ value }}"), message="fine")
    result = {}
    [_ async for _ in execute_graph_reactive(ok, result=result)]
    assert result["has_errors"] is False and result["failed_nodes"] == []
    assert result["summary"]["errors"] == 0 and result["node_errors"] == {}

    failing = build(parser_subflow("s", FAIL_ON_BAD), message="bad")
    result = {}
    [_ async for _ in execute_graph_reactive(failing, result=result)]
    assert result["has_errors"] is True and result["failed_nodes"] == ["s-parser"]
    assert result["node_errors"]["s-parser"]["error_type"] == "UndefinedError"

    loop = build({"type": "graph", "timeout": 3, "nodes": [
        {"id": "input", "type": "user_input"}, {"id": "loop", "type": "loop"},
        {"id": "body", "type": "parser", "data": {"text": FAIL_ON_BAD}}, {"id": "end", "type": "end"},
    ], "edges": [
        edge("e1", "input", "loop", "handle_user_message", "handle_list"),
        edge("e2", "loop", "body", "handle_item", "value"),
        edge("e3", "body", "loop", "handle_parser_output", "handle_loop"),
        edge("e4", "loop", "end", "handle_end", "handle_flow_input"),
    ]}, message='["ok", "bad"]')
    result = {}
    [_ async for _ in execute_graph_reactive(loop, result=result)]
    assert result["has_errors"] is True and result["failed_nodes"] == ["body"]

    blocked = build(parser_subflow("s", "R-{{ value }}"), message="fine")
    blocked._validation_errors = [{"error_type": "GraphValidationError", "error_message": "broken"}]
    result = {}
    [_ async for _ in execute_graph_reactive(blocked, result=result)]
    assert result["has_errors"] is True and result["failed_nodes"] == []


async def test_conditional_routing_error_is_a_node_error_and_fails_a_subflow():
    routing = {"type": "graph", "nodes": [
        {"id": "r-in", "type": "user_input"},
        {"id": "r-cond", "type": "conditional", "data": {"condition": "{{ 'handle_missing' }}"}},
        {"id": "r-parser", "type": "parser", "data": {"text": "yes:{{ value }}"}},
        {"id": "r-end", "type": "end"},
    ], "edges": [
        edge("r1", "r-in", "r-cond", "handle_user_message", "handle_input"),
        edge("r2", "r-cond", "r-parser", "handle_yes", "value"),
        edge("r3", "r-parser", "r-end", "handle_parser_output", "handle_flow_input"),
    ]}
    graph = build(routing, message="x")
    recorder, result = Recorder(), {}
    registry = HookRegistry()
    registry.register_graph(recorder)
    events = [_ async for _ in execute_graph_reactive(graph, hooks=registry, result=result)]
    assert [c["error_type"] for _, c in error_frames(events)] == ["GraphRoutingError"]
    assert result["has_errors"] is True and result["failed_nodes"] == ["r-cond"]
    assert recorder.graph_errors and not recorder.graph_ends

    graph = top_graph(routing, message="x")
    events = await asyncio.wait_for(collect(graph), 5)
    inner = [c for path, c in error_frames(events) if path == ["inner"]]
    assert inner and inner[0]["error_code"] == "INNER_FLOW_FAILED"
    assert "GraphRoutingError" in inner[0]["error_message"]
    assert not graph.nodes["fmt"].outputs


# ─── Step mode ────────────────────────────────────────────────────────────────

async def test_failed_subflow_raises_typed_failure_and_never_delivers_a_partial_result():
    # Text -> End succeeds, Parser -> End fails: the partial text must not leak.
    sub = parser_subflow("p", FAIL_ON_BAD,
                         extra_nodes=[{"id": "p-text", "type": "text", "data": {"text": "PARTIAL"}}],
                         extra_edges=[edge("p-e3", "p-text", "p-end", "handle_text_output", "handle_flow_input")])
    graph = top_graph(sub)
    recorder = Recorder()
    registry = HookRegistry()
    registry.register_graph(recorder)
    events = await asyncio.wait_for(collect(graph, registry), 5)

    errors = error_frames(events)
    assert [(path, c["error_type"]) for path, c in errors] == [
        (["inner", "p-parser"], "UndefinedError"), (["inner"], "OperationFailure")]
    inner_frame = errors[1][1]
    assert inner_frame["error_code"] == "INNER_FLOW_FAILED"
    assert inner_frame["error_message"] == "Sub-flow step 'p-parser' failed: UndefinedError: 'missing' is undefined"
    assert subgraph_ends(events) == ["error"]
    assert not graph.nodes["fmt"].outputs and "r" not in graph.nodes["fmt"].inputs
    assert not any(event.get("type") == "__bypass_all__" for event in events)

    inner_errors = [error for node_id, error in recorder.node_errors if node_id == "inner"]
    assert len(inner_errors) == 1
    outcome = inner_errors[0].outcome["error"]
    assert outcome["code"] == "INNER_FLOW_FAILED" and outcome["retryable"] is False
    assert outcome["details"]["failed_nodes"] == ["p-parser"]
    assert outcome["details"]["child_errors"] == [
        {"node_path": ["p-parser"], "error_type": "UndefinedError", "error_message": "'missing' is undefined"}]
    assert outcome["details"]["partial_content"] == "PARTIAL"
    assert outcome["details"]["child_run_id"].startswith("run-")
    assert "inner" not in recorder.node_ends
    assert len(recorder.graph_errors) >= 1


async def test_recovered_child_diagnostic_is_not_a_failure(monkeypatch):
    original = NodeParser.process

    async def diagnose_then_answer(self, chat_log):
        if self.node_id == "d-parser":
            yield self.yield_debug_error(error_type="JSONParseError", error_message="not JSON")
        async for item in original(self, chat_log):
            yield item

    monkeypatch.setattr(NodeParser, "process", diagnose_then_answer)
    graph = top_graph(parser_subflow("d", "R-{{ value }}"), message="ok")
    events = await asyncio.wait_for(collect(graph), 5)
    assert [(path, c["error_type"]) for path, c in error_frames(events)] == [(["inner", "d-parser"], "JSONParseError")]
    assert subgraph_ends(events) == ["completed"]
    assert graph.nodes["fmt"].outputs["handle_parser_output"]["content"] == "GOT[R-ok]"


async def test_message_names_the_root_cause_not_downstream_timeouts(monkeypatch):
    """A silent producer's diagnostic comes first; the TimeoutErrors after it are effects."""
    original = NodeParser.process

    async def silent(self, chat_log):
        if self.node_id == "t-parser":
            yield self.yield_debug_error(error_type="NetworkError", error_message="upstream unreachable")
            return
        async for item in original(self, chat_log):
            yield item

    monkeypatch.setattr(NodeParser, "process", silent)
    sub = parser_subflow("t", "R-{{ value }}", timeout=1,
                         extra_nodes=[{"id": "t-next", "type": "parser", "data": {"text": "{{ v }}"}}],
                         extra_edges=[edge("t-e3", "t-parser", "t-next", "handle_parser_output", "v")])
    graph = top_graph(sub, message="ok")
    events = await asyncio.wait_for(collect(graph), 5)
    inner = [c for path, c in error_frames(events) if path == ["inner"]]
    assert inner[0]["error_message"] == "Sub-flow step 't-parser' failed: NetworkError: upstream unreachable"
    assert {"t-end", "t-next"} <= {path[-1] for path, c in error_frames(events) if c["error_type"] == "TimeoutError"}


def recovered_then_failed_subflow(prefix):
    """``{prefix}-ok`` reports a diagnostic and completes; ``{prefix}-bad`` then raises."""
    return {"type": "graph", "nodes": [
        {"id": f"{prefix}-in", "type": "user_input"},
        {"id": f"{prefix}-ok", "type": "parser", "data": {"text": "OK-{{ value }}"}},
        {"id": f"{prefix}-bad", "type": "parser", "data": {"text": "BAD-{{ value }}"}},
        {"id": f"{prefix}-end", "type": "end"},
        {"id": f"{prefix}-end2", "type": "end"},
    ], "edges": [
        edge(f"{prefix}-e1", f"{prefix}-in", f"{prefix}-ok", "handle_user_message", "value"),
        edge(f"{prefix}-e2", f"{prefix}-in", f"{prefix}-bad", "handle_user_message", "value"),
        edge(f"{prefix}-e3", f"{prefix}-ok", f"{prefix}-end", "handle_parser_output", "handle_flow_input"),
        edge(f"{prefix}-e4", f"{prefix}-bad", f"{prefix}-end2", "handle_parser_output", "handle_flow_input"),
    ]}


@pytest.fixture
def recovered_then_failed(monkeypatch):
    """``*-ok`` emits a non-fatal JSONParseError first; ``*-bad`` raises a typed 429 after it."""
    from magic_agents.hooks.invocation_control import OperationFailure
    original = NodeParser.process

    async def process(self, chat_log):
        if self.node_id.endswith("-ok"):
            yield self.yield_debug_error(error_type="JSONParseError", error_message="not JSON (recovered)")
        elif self.node_id.endswith("-bad"):
            await asyncio.sleep(0.05)
            raise OperationFailure("HTTP_ERROR", "HTTP 429: Too Many Requests", retryable=True)
        async for item in original(self, chat_log):
            yield item

    monkeypatch.setattr(NodeParser, "process", process)


async def test_message_names_the_failed_step_not_an_earlier_recovered_diagnostic(recovered_then_failed):
    graph = top_graph(recovered_then_failed_subflow("r"), message="ok")
    recorder = Recorder()
    registry = HookRegistry()
    registry.register_graph(recorder)
    events = await asyncio.wait_for(collect(graph, registry), 5)
    error = dict(recorder.node_errors)["inner"].outcome["error"]
    assert error["message"] == "Sub-flow step 'r-bad' failed: HTTP_ERROR: HTTP 429: Too Many Requests"
    assert error["retryable"] is True
    assert error["details"]["cause_path"] == ["r-bad"] and error["details"]["failed_nodes"] == ["r-bad"]
    # The recovered diagnostic is still listed, first, in the child errors.
    assert [entry["node_path"] for entry in error["details"]["child_errors"]] == [["r-ok"], ["r-bad"]]
    inner = [c for path, c in error_frames(events) if path == ["inner"]]
    assert inner[0]["error_message"] == error["message"]


async def test_nested_message_skips_a_recovered_grandchild(recovered_then_failed):
    middle = {"type": "graph", "nodes": [
        {"id": "m-in", "type": "user_input"},
        {"id": "m-inner", "type": "inner", "data": {"magic_flow": recovered_then_failed_subflow("x")}},
        {"id": "m-end", "type": "end"},
    ], "edges": [
        edge("m1", "m-in", "m-inner", "handle_user_message", "handle_user_message"),
        edge("m2", "m-inner", "m-end", "handle_execution_content", "handle_flow_input"),
    ]}
    graph = top_graph(middle, message="ok")
    recorder = Recorder()
    registry = HookRegistry()
    registry.register_graph(recorder)
    await asyncio.wait_for(collect(graph, registry), 5)
    error = dict(recorder.node_errors)["inner"].outcome["error"]
    assert error["message"] == "Sub-flow step 'm-inner/x-bad' failed: HTTP_ERROR: HTTP 429: Too Many Requests"
    assert error["details"]["cause_path"] == ["m-inner", "x-bad"]
    assert error["retryable"] is True


async def test_retryable_is_inherited_from_a_typed_child_outcome_through_nesting(monkeypatch):
    from magic_agents.hooks.invocation_control import OperationFailure
    original = NodeParser.process

    async def transient(self, chat_log):
        if self.node_id == "x-parser":
            raise OperationFailure("HTTP_ERROR", "HTTP 503: Service Unavailable", retryable=True)
        async for item in original(self, chat_log):
            yield item

    monkeypatch.setattr(NodeParser, "process", transient)
    inner = parser_subflow("x", "R-{{ value }}")
    middle = {"type": "graph", "nodes": [
        {"id": "m-in", "type": "user_input"},
        {"id": "m-inner", "type": "inner", "data": {"magic_flow": inner}},
        {"id": "m-end", "type": "end"},
    ], "edges": [
        edge("m1", "m-in", "m-inner", "handle_user_message", "handle_user_message"),
        edge("m2", "m-inner", "m-end", "handle_execution_content", "handle_flow_input"),
    ]}
    graph = top_graph(middle, message="ok")
    recorder = Recorder()
    registry = HookRegistry()
    registry.register_graph(recorder)
    await asyncio.wait_for(collect(graph, registry), 5)
    error = dict(recorder.node_errors)["inner"].outcome["error"]
    assert error["code"] == "INNER_FLOW_FAILED" and error["retryable"] is True
    # The forwarded grandchild frame is the root cause, named by its full path.
    assert error["message"] == "Sub-flow step 'm-inner/x-parser' failed: HTTP_ERROR: HTTP 503: Service Unavailable"
    assert error["details"]["child_errors"][0] == {
        "node_path": ["m-inner", "x-parser"], "error_type": "OperationFailure",
        "error_message": "HTTP 503: Service Unavailable", "error_code": "HTTP_ERROR"}


# ─── Lifecycle Hooks on an Inner Flow ─────────────────────────────────────────

async def test_on_error_hook_on_an_inner_recovers_it_and_keeps_the_child_trace():
    hook = {"id": "recover", "type": "hook", "data": {"lifecycle_event": "onError", "target_node_id": "inner",
                                                       "function_template": RECOVER}}
    graph = top_graph(parser_subflow("h", FAIL_ON_BAD), hooks=[hook])
    events = await asyncio.wait_for(collect(graph), 5)

    assert graph.nodes["fmt"].outputs["handle_parser_output"]["content"] == "GOT[RECOVERED]"
    record = graph.nodes["inner"]._last_invocation_record
    assert record["recovered"] is True
    assert record["original_outcome"]["error"]["code"] == "INNER_FLOW_FAILED"
    seen = side_events(graph, "inner")
    assert seen[0]["event"] == "onError" and seen[0]["outcome"]["error"]["details"]["failed_nodes"] == ["h-parser"]
    # The child trace survives the recovery: SUBGRAPH_START/END and the child error.
    kinds = [event["content"].get("event_type") or event["content"].get("error_type") for event in frames(events)
             if (event.get("source_node_path") or [event["content"].get("node_id")])[0] == "inner"]
    assert "SUBGRAPH_START" in kinds and "UndefinedError" in kinds and "SUBGRAPH_END" in kinds
    assert not [path for path, c in error_frames(events) if path == ["inner"]]


async def test_on_finish_sees_the_error_and_an_unrecovered_inner_keeps_its_trace():
    hook = {"id": "observe", "type": "hook", "data": {"lifecycle_event": "onFinish", "target_node_id": "inner",
                                                       "function_template": OBSERVE}}
    graph = top_graph(parser_subflow("h", FAIL_ON_BAD), hooks=[hook])
    events = await asyncio.wait_for(collect(graph), 5)

    seen = side_events(graph, "inner")
    assert seen[0]["event"] == "onFinish" and seen[0]["outcome"]["status"] == "error"
    assert seen[0]["outcome"]["error"]["code"] == "INNER_FLOW_FAILED"
    assert "__bypass_all__" not in str(seen)
    errors = error_frames(events)
    assert (["inner", "h-parser"], "UndefinedError") in [(path, c["error_type"]) for path, c in errors]
    inner = [c for path, c in errors if path == ["inner"]]
    assert inner[0]["error_type"] == "OperationFailure" and inner[0]["error_code"] == "INNER_FLOW_FAILED"
    assert subgraph_ends(events) == ["error"]
    assert not graph.nodes["fmt"].outputs


async def test_bypass_all_under_hook_control_is_a_typed_error_not_a_success_handle():
    """An Inner configuration error (BYPASS_ALL) seen by a Hook is NODE_ERROR."""
    hook = {"id": "observe", "type": "hook", "data": {"lifecycle_event": "onFinish", "target_node_id": "inner",
                                                       "function_template": OBSERVE}}
    graph = build({"type": "graph", "timeout": 3, "nodes": [
        {"id": "input", "type": "user_input"},
        {"id": "inner", "type": "inner", "data": {}},
        {"id": "fmt", "type": "parser", "data": {"text": "GOT[{{ r }}]"}},
        {"id": "end", "type": "end"}, hook,
    ], "edges": [
        edge("input-inner", "input", "inner", "handle_user_message", "handle_user_message"),
        edge("inner-fmt", "inner", "fmt", "handle_execution_content", "r"),
        edge("fmt-end", "fmt", "end", "handle_parser_output", "handle_flow_input"),
    ]}, message="x")
    events = await asyncio.wait_for(collect(graph), 5)
    seen = side_events(graph, "inner")
    assert seen[0]["outcome"]["status"] == "error" and seen[0]["outcome"]["error"]["code"] == "NODE_ERROR"
    assert "magic_flow" in seen[0]["outcome"]["error"]["message"]
    assert "ConfigurationError" in [c["error_type"] for _, c in error_frames(events)]
    assert not graph.nodes["fmt"].outputs


async def test_failover_to_a_failing_backup_inner_surfaces_a_typed_error():
    """provider-failover with both providers down: no silent empty success."""
    redirect = ('def fail_over(context, chat_log):\n'
                '    if context.outcome["status"] != "error":\n'
                '        return None\n'
                '    return {"action": "redirect", "connection": "failover-backup", "content": "again"}\n')
    graph = build({"type": "graph", "timeout": 3, "nodes": [
        {"id": "input", "type": "user_input"},
        {"id": "primary", "type": "parser", "data": {"text": "{{ missing.invalid() }}"}},
        {"id": "backup", "type": "inner", "data": {"magic_flow": parser_subflow("b", "{{ missing.invalid() }}")}},
        {"id": "failover", "type": "hook", "data": {"lifecycle_event": "onError", "target_node_id": "primary",
                                                    "failure_policy": "preserve", "function_template": redirect}},
        {"id": "end", "type": "end"},
    ], "edges": [
        edge("input-primary", "input", "primary", "handle_user_message", "value"),
        edge("failover-backup", "failover", "backup", "handle-child-call", "handle_user_message"),
        edge("primary-end", "primary", "end", "handle_parser_output", "handle_flow_input"),
    ]}, message="hi")
    recorder = Recorder()
    registry = HookRegistry()
    registry.register_graph(recorder)
    await asyncio.wait_for(collect(graph, registry), 5)
    record = graph.nodes["primary"]._last_invocation_record
    assert record["recovered"] is False
    assert record["outcome"]["status"] == "error"
    assert record["outcome"]["error"]["code"] == "INNER_FLOW_FAILED"
    assert "primary" in dict(recorder.node_errors)
    assert not graph.nodes["end"].inputs
    assert recorder.graph_errors and not recorder.graph_ends


async def test_budget_timeout_of_a_hook_controlled_inner_closes_its_subgraph(monkeypatch):
    """The kept trace pairs SUBGRAPH_START with an end even when the Inner is cancelled."""
    original = NodeParser.process

    async def slow(self, chat_log):
        if self.node_id == "s-parser":
            await asyncio.sleep(2)
        async for item in original(self, chat_log):
            yield item

    monkeypatch.setattr(NodeParser, "process", slow)
    hook = {"id": "observe", "type": "hook", "data": {"lifecycle_event": "onFinish", "target_node_id": "inner",
                                                       "function_template": OBSERVE}}
    spec = {"type": "graph", "timeout": 1, "nodes": [
        {"id": "input", "type": "user_input"},
        {"id": "inner", "type": "inner", "data": {"magic_flow": parser_subflow("s", "R-{{ value }}")}},
        {"id": "end", "type": "end"},
        hook,
    ], "edges": [
        edge("input-inner", "input", "inner", "handle_user_message", "handle_user_message"),
        edge("inner-end", "inner", "end", "handle_execution_content", "handle_flow_input"),
    ]}
    events = await asyncio.wait_for(collect(build(spec, message="ok")), 5)
    subgraph = [event["content"] for event in frames(events)
                if event["content"].get("event_type") in ("SUBGRAPH_START", "SUBGRAPH_END")]
    assert [frame["event_type"] for frame in subgraph] == ["SUBGRAPH_START", "SUBGRAPH_END"]
    assert subgraph[1]["child_run_id"] == subgraph[0]["child_run_id"]
    assert subgraph[1]["status"] == "error" and subgraph[1]["reason"] == "cancelled"
    assert [c["error_code"] for path, c in error_frames(events) if path == ["inner"]] == ["NODE_TIMEOUT"]


# ─── Tolerated failures inside a sub-flow ─────────────────────────────────────
# A failure the sub-flow itself absorbs (a Loop null slot, a fan-in rendering
# with the other inputs) still fails the Inner Flow, as the same failure fails
# a top-level run. What reached End is kept in error.details.partial_content,
# and an onError Hook can adopt it to keep the degraded Result.

ADOPT_PARTIAL = ('def adopt(context, chat_log):\n'
                 '    partial = context.outcome["error"]["details"]["partial_content"]\n'
                 '    return {"action": "outcome", "outcome": {"status": "success",\n'
                 '            "content": {"handle_execution_content": partial}}}\n')


def degrading_subflow():
    """Three writers fan into one Parser; ``d-b`` fails, the Parser renders the others."""
    return {"type": "graph", "nodes": [
        {"id": "d-in", "type": "user_input"},
        {"id": "d-a", "type": "parser", "data": {"text": "A"}},
        {"id": "d-b", "type": "parser", "data": {"text": "{{ missing.invalid() }}"}},
        {"id": "d-c", "type": "parser", "data": {"text": "C"}},
        {"id": "d-asm", "type": "parser", "data": {"text": "{{ a }}|{{ b }}|{{ c }}"}},
        {"id": "d-end", "type": "end"},
    ], "edges": [
        edge("d1", "d-in", "d-a", "handle_user_message", "x"),
        edge("d2", "d-in", "d-b", "handle_user_message", "x"),
        edge("d3", "d-in", "d-c", "handle_user_message", "x"),
        edge("d4", "d-a", "d-asm", "handle_parser_output", "a"),
        edge("d5", "d-b", "d-asm", "handle_parser_output", "b"),
        edge("d6", "d-c", "d-asm", "handle_parser_output", "c"),
        edge("d7", "d-asm", "d-end", "handle_parser_output", "handle_flow_input"),
    ]}


def looping_subflow():
    """A Loop inside the sub-flow whose 2nd item fails: End gets ``["R-a", null, "R-c"]``."""
    return {"type": "graph", "nodes": [
        {"id": "l-in", "type": "user_input"},
        {"id": "l-loop", "type": "loop"},
        {"id": "l-body", "type": "parser", "data": {"text": FAIL_ON_BAD}},
        {"id": "l-fmt", "type": "parser", "data": {"text": "{{ r | tojson }}"}},
        {"id": "l-end", "type": "end"},
    ], "edges": [
        edge("l1", "l-in", "l-loop", "handle_user_message", "handle_list"),
        edge("l2", "l-loop", "l-body", "handle_item", "value"),
        edge("l3", "l-body", "l-loop", "handle_parser_output", "handle_loop"),
        edge("l4", "l-loop", "l-fmt", "handle_end", "r"),
        edge("l5", "l-fmt", "l-end", "handle_parser_output", "handle_flow_input"),
    ]}


@pytest.mark.parametrize("subflow, message, partial", [
    (degrading_subflow, "go", "A||C"),
    (looping_subflow, '["a", "bad", "c"]', '["R-a", null, "R-c"]'),
], ids=["fan-in", "loop"])
async def test_a_failure_the_subflow_tolerates_still_fails_the_inner(subflow, message, partial):
    graph = top_graph(subflow(), message=message)
    recorder = Recorder()
    registry = HookRegistry()
    registry.register_graph(recorder)
    await asyncio.wait_for(collect(graph, registry), 5)
    error = dict(recorder.node_errors)["inner"].outcome["error"]
    assert error["code"] == "INNER_FLOW_FAILED"
    assert error["details"]["partial_content"] == partial
    assert not graph.nodes["fmt"].outputs  # the degraded Result is not delivered


@pytest.mark.parametrize("subflow, message, rendered", [
    (degrading_subflow, "go", "GOT[A||C]"),
    # The parent Parser reads the adopted JSON text as a list.
    (looping_subflow, '["a", "bad", "c"]', "GOT[['R-a', None, 'R-c']]"),
], ids=["fan-in", "loop"])
async def test_on_error_hook_can_adopt_the_partial_result(subflow, message, rendered):
    hook = {"id": "adopt", "type": "hook", "data": {"lifecycle_event": "onError", "target_node_id": "inner",
                                                     "function_template": ADOPT_PARTIAL}}
    graph = top_graph(subflow(), hooks=[hook], message=message)
    await asyncio.wait_for(collect(graph), 5)
    assert graph.nodes["fmt"].outputs["handle_parser_output"]["content"] == rendered
    assert graph.nodes["inner"]._last_invocation_record["recovered"] is True


# ─── Inner Flow as a Loop body ────────────────────────────────────────────────

def loop_graph(subflow, *, hooks=(), items='["a", "bad", "c"]'):
    return build({"type": "graph", "timeout": 3, "nodes": [
        {"id": "input", "type": "user_input"},
        {"id": "loop", "type": "loop"},
        {"id": "inner", "type": "inner", "data": {"magic_flow": subflow}},
        {"id": "end", "type": "end"},
        *hooks,
    ], "edges": [
        edge("e1", "input", "loop", "handle_user_message", "handle_list"),
        edge("e2", "loop", "inner", "handle_item", "handle_user_message"),
        edge("e3", "inner", "loop", "handle_execution_content", "handle_loop"),
        edge("e4", "loop", "end", "handle_end", "handle_flow_input"),
    ]}, message=items)


def loop_values(graph):
    return graph.nodes["loop"].outputs["handle_end"]["content"]


async def test_inner_loop_body_failure_leaves_an_empty_slot():
    graph = loop_graph(parser_subflow("l", FAIL_ON_BAD))
    await asyncio.wait_for(collect(graph), 5)
    assert loop_values(graph) == ["R-a", None, "R-c"]


async def test_inner_loop_body_on_error_recovery_fills_the_slot():
    hook = {"id": "recover", "type": "hook", "data": {"lifecycle_event": "onError", "target_node_id": "inner",
                                                       "function_template": RECOVER}}
    graph = loop_graph(parser_subflow("l", FAIL_ON_BAD), hooks=[hook])
    await asyncio.wait_for(collect(graph), 5)
    assert loop_values(graph) == ["R-a", "RECOVERED", "R-c"]


async def test_inner_loop_body_with_a_recovered_diagnostic_keeps_real_values(monkeypatch):
    original = NodeParser.process

    async def diagnose_then_answer(self, chat_log):
        if self.node_id == "l-parser":
            yield self.yield_debug_error(error_type="JSONParseError", error_message="not JSON")
        async for item in original(self, chat_log):
            yield item

    monkeypatch.setattr(NodeParser, "process", diagnose_then_answer)
    graph = loop_graph(parser_subflow("l", "R-{{ value }}"), items='["a", "b"]')
    await asyncio.wait_for(collect(graph), 5)
    assert loop_values(graph) == ["R-a", "R-b"]
