from __future__ import annotations

from types import SimpleNamespace

from magic_agents.hooks.hook_relay import HookRelay


class RecordingFlowHooks:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def on_llm_start(self, context, **kwargs):
        self.calls.append(("on_llm_start", context))

    def on_llm_end(self, context):
        self.calls.append(("on_llm_end", context))

    def on_tool_start(self, context):
        self.calls.append(("on_tool_start", context))

    def on_tool_end(self, context):
        self.calls.append(("on_tool_end", context))

    def on_llm_loop_end(self, context):
        self.calls.append(("on_llm_loop_end", context))


class Response:
    id = "req-1"
    model = "gpt-4"
    content = "hello"
    finish_reason = "stop"
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


class ToolResult:
    name = "search"
    content = "result"
    error = None
    is_error = False
    duration_ms = 12.5
    tool_call_id = "tool-1"


def test_relay_reads_iteration_metadata_from_agent_state_step():
    hooks = RecordingFlowHooks()
    relay = HookRelay(flow_hooks=hooks)
    state = SimpleNamespace(step=7, iteration=2)

    relay.on_iteration_start(iteration=2, state=state)
    relay.on_llm_response(Response(), state)
    relay.on_tool_start("search", "tool-1", {"q": "x"}, state)
    relay.on_tool_complete(ToolResult(), state)
    relay.on_loop_complete(Response(), state)

    by_name = {name: ctx for name, ctx in hooks.calls}
    assert by_name["on_llm_start"].inputs["llm_call_count"] == 7
    assert by_name["on_llm_end"].outputs["iteration"] == 7
    assert by_name["on_tool_start"].inputs["iteration"] == 7
    assert by_name["on_tool_end"].outputs["iteration"] == 7
    assert by_name["on_llm_loop_end"].outputs["iteration"] == 7
    assert by_name["on_llm_loop_end"].outputs["total_iterations"] == 8
    assert relay.collected_tool_calls[0]["iteration"] == 7


def test_relay_falls_back_to_legacy_iteration_attribute():
    hooks = RecordingFlowHooks()
    relay = HookRelay(flow_hooks=hooks)
    legacy_state = SimpleNamespace(iteration=3)

    relay.on_llm_response(Response(), legacy_state)


    assert hooks.calls[0][1].outputs["iteration"] == 3
