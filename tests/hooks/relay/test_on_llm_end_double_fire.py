"""Regression tests: on_llm_end and on_llm_loop_end correct event counts.

Phase 5.1: Verifies that:
- N-iteration tool loop fires exactly N on_llm_end + 1 on_llm_loop_end (no N+1)
- Single-iteration (no tools) fires correct counts
- Zero iterations (error exit) fires no LLM end events
- No double-fire bug where on_llm_end fires for both per-iteration AND loop-complete
"""
from unittest.mock import MagicMock

from magic_llm.agent.types import AgentState, ToolResult
from magic_llm.model import ModelChatResponse

from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_relay import HookRelay


# ─── RecordingFlowHooks with on_llm_loop_end support ──────────────────


class _RecordingFlowHooks:
    """Sync FlowHooks that records method calls for event count assertion.

    Includes on_llm_loop_end for testing the new loop completion event.
    """

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

    def on_node_error(self, context, **kwargs):
        self.calls.append(("on_node_error", context))

    def call_counts(self) -> dict[str, int]:
        """Return dict of hook_name → call count."""
        counts: dict[str, int] = {}
        for name, _ in self.calls:
            counts[name] = counts.get(name, 0) + 1
        return counts

    def hook_order(self) -> list[str]:
        """Return ordered list of hook names as they were invoked."""
        return [name for name, _ in self.calls]


# ─── Helpers ──────────────────────────────────────────────────────────


def make_agent_state(**overrides) -> AgentState:
    """Create a minimal AgentState for testing."""
    state = MagicMock(spec=AgentState)
    state.iteration = overrides.get("iteration", 0)
    return state


def make_chat_response(**overrides) -> ModelChatResponse:
    """Create a minimal ModelChatResponse for testing."""
    resp = MagicMock(spec=ModelChatResponse)
    resp.model = overrides.get("model", "gpt-4")
    resp.content = overrides.get("content", "Hello!")
    resp.finish_reason = overrides.get("finish_reason", "stop")
    resp.id = overrides.get("id", "chatcmpl-abc")
    resp.usage = overrides.get("usage", None)
    return resp


def make_tool_result(**overrides) -> ToolResult:
    """Create a ToolResult for testing."""
    return ToolResult(
        name=overrides.get("name", "calculator"),
        content=overrides.get("content", "42"),
        is_error=overrides.get("is_error", False),
        error=overrides.get("error", None),
        tool_call_id=overrides.get("tool_call_id", "call-1"),
    )


# ─── Tests ────────────────────────────────────────────────────────────


class TestOnLLMEndDoubleFire:
    """Tests for correct on_llm_end / on_llm_loop_end event counts."""

    # ── Scenario: N-iteration tool loop ──────────────────────────────

    def test_three_iteration_tool_loop_correct_counts(self):
        """3-iteration tool loop: 3 on_llm_end + 1 on_llm_loop_end + 3 on_llm_start."""
        hooks = _RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")

        # Simulate 3 iterations with tool calls
        for i in range(3):
            state = make_agent_state(iteration=i)
            relay.on_iteration_start(iteration=i, state=state)

            response = make_chat_response(content=f"Response {i}")
            relay.on_llm_response(response=response, state=state)

            relay.on_tool_start("calc", f"call-{i}", {"x": i}, state)
            relay.on_tool_complete(
                make_tool_result(name="calc", content=str(i * 2), tool_call_id=f"call-{i}"),
                state,
            )

        # Fire loop completion
        final_state = make_agent_state(iteration=2)
        final_response = make_chat_response(content="Final aggregated")
        relay.on_loop_complete(final_response=final_response, state=final_state)

        counts = hooks.call_counts()
        order = hooks.hook_order()

        # Assert correct counts — no N+1 double-fire
        assert counts.get("on_llm_start", 0) == 3, (
            f"Expected 3 on_llm_start calls, got {counts.get('on_llm_start', 0)}"
        )
        assert counts.get("on_llm_end", 0) == 3, (
            f"Expected 3 on_llm_end calls, got {counts.get('on_llm_end', 0)}"
        )
        assert counts.get("on_llm_loop_end", 0) == 1, (
            f"Expected 1 on_llm_loop_end call, got {counts.get('on_llm_loop_end', 0)}"
        )

        # Assert ordering: on_llm_loop_end fires AFTER all per-iteration events
        loop_end_index = order.index("on_llm_loop_end")
        last_llm_end_index = max(
            i for i, name in enumerate(order) if name == "on_llm_end"
        )
        assert loop_end_index > last_llm_end_index, (
            f"on_llm_loop_end (index {loop_end_index}) should fire after "
            f"last on_llm_end (index {last_llm_end_index})"
        )

        # Verify no on_llm_end fires after on_llm_loop_end
        after_loop_end = order[loop_end_index + 1:]
        assert "on_llm_end" not in after_loop_end, (
            f"on_llm_end should NOT fire after on_llm_loop_end. Got: {after_loop_end}"
        )

    # ── Scenario: Single-iteration (no tools) ────────────────────────

    def test_single_iteration_loop_correct_counts(self):
        """1-iteration loop (no tools): 1 on_llm_end + 1 on_llm_loop_end + 1 on_llm_start."""
        hooks = _RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")

        state = make_agent_state(iteration=0)
        relay.on_iteration_start(iteration=0, state=state)

        response = make_chat_response(content="Single response")
        relay.on_llm_response(response=response, state=state)

        # No tool calls in this iteration
        relay.on_loop_complete(
            final_response=make_chat_response(content="Single final"),
            state=make_agent_state(iteration=0),
        )

        counts = hooks.call_counts()

        assert counts.get("on_llm_start", 0) == 1, (
            f"Expected 1 on_llm_start, got {counts.get('on_llm_start', 0)}"
        )
        assert counts.get("on_llm_end", 0) == 1, (
            f"Expected 1 on_llm_end, got {counts.get('on_llm_end', 0)}"
        )
        assert counts.get("on_llm_loop_end", 0) == 1, (
            f"Expected 1 on_llm_loop_end, got {counts.get('on_llm_loop_end', 0)}"
        )
        # No tool events
        assert counts.get("on_tool_start", 0) == 0
        assert counts.get("on_tool_end", 0) == 0

    # ── Scenario: Zero iterations (error exit) ───────────────────────

    def test_zero_iterations_fires_no_llm_end_events(self):
        """0 iterations (error exit): no on_llm_end or on_llm_loop_end."""
        hooks = _RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")

        # No events fired — loop exited before any iteration
        # Only on_budget_exceeded was called (simulating error at start)
        relay.on_budget_exceeded(budget_type="time", details="Startup failure")

        counts = hooks.call_counts()

        assert counts.get("on_llm_start", 0) == 0, (
            f"Expected 0 on_llm_start, got {counts.get('on_llm_start', 0)}"
        )
        assert counts.get("on_llm_end", 0) == 0, (
            f"Expected 0 on_llm_end, got {counts.get('on_llm_end', 0)}"
        )
        assert counts.get("on_llm_loop_end", 0) == 0, (
            f"Expected 0 on_llm_loop_end, got {counts.get('on_llm_loop_end', 0)}"
        )
        # on_node_error should fire (budget exceeded → node_error)
        assert counts.get("on_node_error", 0) == 1, (
            f"Expected 1 on_node_error, got {counts.get('on_node_error', 0)}"
        )

    # ── Scenario: No double-fire on loop_complete ────────────────────

    def test_loop_complete_carries_aggregated_outputs(self):
        """on_llm_loop_end carries aggregated content, iteration, and total_iterations."""
        hooks = _RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")

        state = make_agent_state(iteration=0)
        relay.on_iteration_start(iteration=0, state=state)
        relay.on_llm_response(make_chat_response(content="Iter 0"), state)
        relay.on_loop_complete(
            final_response=make_chat_response(
                content="Final aggregated",
                id="chatcmpl-final",
                usage={"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            ),
            state=make_agent_state(iteration=0),
        )

        # Get the on_llm_loop_end context
        loop_end_invocations = [inv for inv in hooks.calls if inv[0] == "on_llm_loop_end"]
        assert len(loop_end_invocations) == 1
        _, ctx = loop_end_invocations[0]

        # Verify aggregated outputs
        assert ctx.outputs["iteration"] == 0, (
            f"Expected iteration=0 (0-indexed), got {ctx.outputs.get('iteration')}"
        )
        assert ctx.outputs["total_iterations"] == 1, (
            f"Expected total_iterations=1 (1-indexed), got {ctx.outputs.get('total_iterations')}"
        )
        assert ctx.outputs["content"] == "Final aggregated"
        assert ctx.outputs["provider_request_id"] == "chatcmpl-final"

        # Token data
        assert ctx.outputs["prompt_tokens"] == 50
        assert ctx.outputs["completion_tokens"] == 100
        assert ctx.outputs["total_tokens"] == 150
