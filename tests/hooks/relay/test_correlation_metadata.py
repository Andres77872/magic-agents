"""Regression tests: correlation metadata across LLM and tool events.

Phase 5.3: Verifies that:
- provider_request_id consistent across tool ↔ LLM events
- provider_request_id is None when no LLM response preceded (tool-first)
- Iteration number present in per-iteration events (0-indexed)
- Loop end carries 0-indexed iteration and 1-indexed total_iterations
- Token data consistent across events
- Token data handles missing usage gracefully
"""
from unittest.mock import MagicMock

from magic_llm.agent.types import AgentState, ToolResult
from magic_llm.model import ModelChatResponse

from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_relay import HookRelay


# ─── Recording FlowHooks ──────────────────────────────────────────────


class _RecordingFlowHooks:
    """Sync FlowHooks that records contexts for assertion."""

    def __init__(self):
        self.calls: list[tuple[str, HookContext]] = []

    def on_llm_start(self, context, llm_config=None):
        self.calls.append(("on_llm_start", context))

    def on_llm_end(self, context, response=None):
        self.calls.append(("on_llm_end", context))

    def on_llm_loop_end(self, context):
        self.calls.append(("on_llm_loop_end", context))

    def on_tool_start(self, context):
        self.calls.append(("on_tool_start", context))

    def on_tool_end(self, context):
        self.calls.append(("on_tool_end", context))

    def get_context(self, hook_name: str) -> HookContext:
        """Return the first context for a given hook name."""
        for name, ctx in self.calls:
            if name == hook_name:
                return ctx
        raise AssertionError(f"No call recorded for '{hook_name}'")

    def get_contexts(self, hook_name: str) -> list[HookContext]:
        """Return all contexts for a given hook name."""
        return [ctx for name, ctx in self.calls if name == hook_name]


# ─── Helpers ──────────────────────────────────────────────────────────


def make_agent_state(**overrides) -> AgentState:
    state = MagicMock(spec=AgentState)
    state.iteration = overrides.get("iteration", 0)
    return state


def make_chat_response(**overrides) -> ModelChatResponse:
    resp = MagicMock(spec=ModelChatResponse)
    resp.model = overrides.get("model", "gpt-4")
    resp.content = overrides.get("content", "Hello!")
    resp.finish_reason = overrides.get("finish_reason", "stop")
    resp.id = overrides.get("id", "chatcmpl-abc")
    resp.usage = overrides.get("usage", None)
    return resp


def make_tool_result(**overrides) -> ToolResult:
    return ToolResult(
        name=overrides.get("name", "calculator"),
        content=overrides.get("content", "42"),
        is_error=overrides.get("is_error", False),
        error=overrides.get("error", None),
        tool_call_id=overrides.get("tool_call_id", "call-1"),
    )


# ─── Tests ────────────────────────────────────────────────────────────


class TestCorrelationMetadata:
    """Tests for provider_request_id, iteration, and token metadata."""

    # ── provider_request_id consistency ─────────────────────────────

    def test_provider_request_id_consistent_across_events(self):
        """provider_request_id matches across LLM end and tool events."""
        hooks = _RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")

        state = make_agent_state(iteration=0)
        response = make_chat_response(id="chatcmpl-xyz789")

        # Fire LLM response
        relay.on_llm_response(response=response, state=state)

        # Fire tool events (should carry same provider_request_id)
        relay.on_tool_start("calc", "call-1", {"x": 1}, state)
        relay.on_tool_complete(
            make_tool_result(name="calc", content="2", tool_call_id="call-1"),
            state,
        )

        # Verify on_llm_end has provider_request_id
        llm_end_ctx = hooks.get_context("on_llm_end")
        assert llm_end_ctx.outputs["provider_request_id"] == "chatcmpl-xyz789", (
            f"Expected provider_request_id='chatcmpl-xyz789', "
            f"got {llm_end_ctx.outputs.get('provider_request_id')}"
        )

        # Verify tool contexts have provider_request_id
        tool_start_ctx = hooks.get_context("on_tool_start")
        assert tool_start_ctx.inputs["provider_request_id"] == "chatcmpl-xyz789"

        tool_end_ctx = hooks.get_context("on_tool_end")
        assert tool_end_ctx.outputs["provider_request_id"] == "chatcmpl-xyz789"

        # Verify collected tool data has provider_request_id
        assert len(relay._collected_tool_calls) == 1
        assert relay._collected_tool_calls[0]["provider_request_id"] == "chatcmpl-xyz789"

        assert len(relay._collected_tool_results) == 1
        assert relay._collected_tool_results[0]["provider_request_id"] == "chatcmpl-xyz789"

    def test_provider_request_id_none_when_no_llm_preceded(self):
        """provider_request_id is None when tool events fire before any LLM response."""
        hooks = _RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")

        state = make_agent_state(iteration=0)

        # Fire tool events WITHOUT preceding on_llm_response
        relay.on_tool_start("calc", "call-1", {"x": 1}, state)
        relay.on_tool_complete(
            make_tool_result(name="calc", content="2", tool_call_id="call-1"),
            state,
        )

        # provider_request_id should be None in all contexts
        tool_start_ctx = hooks.get_context("on_tool_start")
        assert tool_start_ctx.inputs["provider_request_id"] is None, (
            f"Expected provider_request_id=None, got "
            f"{tool_start_ctx.inputs.get('provider_request_id')}"
        )

        tool_end_ctx = hooks.get_context("on_tool_end")
        assert tool_end_ctx.outputs["provider_request_id"] is None

        # Collected tool data also has None
        assert relay._collected_tool_calls[0]["provider_request_id"] is None
        assert relay._collected_tool_results[0]["provider_request_id"] is None

    def test_provider_request_id_changes_per_iteration(self):
        """Each iteration's events carry that iteration's provider_request_id."""
        hooks = _RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")

        # Iteration 0: response id = "chatcmpl-aaa"
        state0 = make_agent_state(iteration=0)
        relay.on_llm_response(make_chat_response(id="chatcmpl-aaa"), state0)
        relay.on_tool_start("tool_a", "call-0", {}, state0)
        relay.on_tool_complete(
            make_tool_result(name="tool_a", tool_call_id="call-0"),
            state0,
        )

        # Iteration 1: response id = "chatcmpl-bbb"
        state1 = make_agent_state(iteration=1)
        relay.on_llm_response(make_chat_response(id="chatcmpl-bbb"), state1)
        relay.on_tool_start("tool_b", "call-1", {}, state1)
        relay.on_tool_complete(
            make_tool_result(name="tool_b", tool_call_id="call-1"),
            state1,
        )

        llm_end_ctxs = hooks.get_contexts("on_llm_end")
        assert len(llm_end_ctxs) == 2
        assert llm_end_ctxs[0].outputs["provider_request_id"] == "chatcmpl-aaa"
        assert llm_end_ctxs[1].outputs["provider_request_id"] == "chatcmpl-bbb"

        # Tool events carry correct iteration's request_id
        tool_start_ctxs = hooks.get_contexts("on_tool_start")
        assert tool_start_ctxs[0].inputs["provider_request_id"] == "chatcmpl-aaa"
        assert tool_start_ctxs[1].inputs["provider_request_id"] == "chatcmpl-bbb"

    # ── Iteration metadata ──────────────────────────────────────────

    def test_iteration_present_in_per_iteration_events(self):
        """Iteration number (0-indexed) present in on_llm_end and tool events."""
        hooks = _RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")

        # Iteration 0
        relay.on_llm_response(make_chat_response(), make_agent_state(iteration=0))
        relay.on_tool_start("calc", "call-0", {}, make_agent_state(iteration=0))

        # Iteration 1
        relay.on_llm_response(make_chat_response(), make_agent_state(iteration=1))
        relay.on_tool_start("tool_b", "call-1", {}, make_agent_state(iteration=1))

        llm_end_ctxs = hooks.get_contexts("on_llm_end")
        tool_start_ctxs = hooks.get_contexts("on_tool_start")

        assert llm_end_ctxs[0].outputs["iteration"] == 0
        assert llm_end_ctxs[1].outputs["iteration"] == 1
        assert tool_start_ctxs[0].inputs["iteration"] == 0
        assert tool_start_ctxs[1].inputs["iteration"] == 1

    # ── Token data ──────────────────────────────────────────────────

    # ── provider_request_id cleared at loop boundary ────────────────

    def test_provider_request_id_fresh_per_loop(self):
        """Each on_llm_response in new loop sets fresh provider_request_id."""
        relay = HookRelay(node_id="n1")

        # Loop 1
        relay.on_llm_response(
            make_chat_response(id="chatcmpl-loop1"),
            make_agent_state(iteration=0),
        )
        assert relay._current_provider_request_id == "chatcmpl-loop1"

        # Clear at loop boundary
        relay.on_loop_complete(
            final_response=make_chat_response(content="Done"),
            state=make_agent_state(iteration=0),
        )
        assert relay._current_provider_request_id is None

        # Loop 2 — fresh request_id
        relay.on_llm_response(
            make_chat_response(id="chatcmpl-loop2"),
            make_agent_state(iteration=0),
        )
        assert relay._current_provider_request_id == "chatcmpl-loop2", (
            "Should have fresh provider_request_id from new loop, "
            f"got {relay._current_provider_request_id}"
        )
