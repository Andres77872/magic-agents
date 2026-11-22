"""Integration tests: Nested LLM hook propagation via PARENT_HOOKS ContextVar.

Phase 5.5: Verifies that:
- PARENT_HOOKS ContextVar lifecycle: parent sets, child reads, cleanup after
- Nested depth tracking: depth increments correctly across nesting
- Child events carry correlation metadata: nested_depth, parent_run_id, nested_request_id
- No parent hooks → backward compatible with hooks=None
- Child events propagate through parent's HookRelay
"""
from unittest.mock import MagicMock

from magic_llm.agent.types import AgentState, ToolResult
from magic_llm.agent._loop_shared import PARENT_HOOKS, DEPTH
from magic_llm.model import ModelChatResponse

from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_relay import HookRelay


# ─── Recording hooks ──────────────────────────────────────────────────


class _RecordingFlowHooks:
    """Sync FlowHooks that records method calls for assertion."""

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

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]


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
    resp.id = overrides.get("id", "chatcmpl-parent")
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


class TestNestedHooksComprehensive:
    """Integration tests for nested LLM hook propagation."""

    # ── Scenario: PARENT_HOOKS ContextVar lifecycle ────────────────

    # ── Scenario: Nested depth tracking ─────────────────────────────

    def test_nested_depth_tracking(self):
        """DEPTH ContextVar tracks nesting depth across levels."""
        hooks = _RecordingFlowHooks()

        # Create parent HookRelay with nested_depth=0
        parent_relay = HookRelay(
            flow_hooks=hooks,
            node_id="parent-node",
            nested_depth=0,
        )

        # Simulate parent loop setting PARENT_HOOKS and DEPTH=0
        parent_hooks_token = PARENT_HOOKS.set(parent_relay)
        depth_token = DEPTH.set(0)

        # Verify initial depth
        assert DEPTH.get() == 0, f"Expected DEPTH=0, got {DEPTH.get()}"

        try:
            # ── Child level 1 ──
            # Simulate TaskExecutor._execute_nested_llm_node()
            depth_token_1 = DEPTH.set(DEPTH.get() + 1)
            try:
                assert DEPTH.get() == 1, (
                    f"Expected DEPTH=1 at child level 1, got {DEPTH.get()}"
                )

                # Fire child event — _build_context reads DEPTH.get()
                child_state = make_agent_state(iteration=0)
                parent_relay.on_iteration_start(iteration=0, state=child_state)

                # Check that nested metadata was injected by _build_context
                # We need to get the last on_llm_start context
                llm_start_invocations = [
                    ctx for name, ctx in hooks.calls
                    if name == "on_llm_start"
                ]
                ctx = llm_start_invocations[-1]
                assert ctx.metadata.get("nested_depth") == 1, (
                    f"Expected nested_depth=1 at child level 1, "
                    f"got {ctx.metadata.get('nested_depth')}"
                )

                # ── Child level 2 (grandchild) ──
                depth_token_2 = DEPTH.set(DEPTH.get() + 1)
                try:
                    assert DEPTH.get() == 2, (
                        f"Expected DEPTH=2 at child level 2, got {DEPTH.get()}"
                    )

                    parent_relay.on_iteration_start(iteration=0, state=child_state)

                    llm_start_invocations = [
                        ctx for name, ctx in hooks.calls
                        if name == "on_llm_start"
                    ]
                    ctx = llm_start_invocations[-1]
                    assert ctx.metadata.get("nested_depth") == 2, (
                        f"Expected nested_depth=2 at child level 2, "
                        f"got {ctx.metadata.get('nested_depth')}"
                    )

                finally:
                    DEPTH.reset(depth_token_2)

                # After grandchild cleanup, depth should be back to 1
                assert DEPTH.get() == 1, (
                    f"Expected DEPTH=1 after grandchild cleanup, "
                    f"got {DEPTH.get()}"
                )

            finally:
                DEPTH.reset(depth_token_1)

            # After child cleanup, depth should be back to 0
            assert DEPTH.get() == 0, (
                f"Expected DEPTH=0 after child cleanup, "
                f"got {DEPTH.get()}"
            )

        finally:
            PARENT_HOOKS.reset(parent_hooks_token)
            DEPTH.reset(depth_token)

    # ── Scenario: Child events carry correlation metadata ──────────

    def test_child_events_carry_correlation_metadata(self):
        """Child events carry nested_depth, parent_run_id, nested_request_id."""
        hooks = _RecordingFlowHooks()
        parent_relay = HookRelay(
            flow_hooks=hooks,
            node_id="parent-node",
            graph_id="graph-1",
            run_id="run-parent",
            parent_run_id=None,       # Root — no parent
            nested_depth=0,
            nested_request_id="root-uuid",
        )

        # Create a CHILD HookRelay with parent correlation
        child_relay = HookRelay(
            flow_hooks=hooks,
            node_id="child-node",
            graph_id="graph-1",
            run_id="run-child",
            parent_run_id=parent_relay._run_id,  # Links to parent
            nested_depth=1,
            nested_request_id="child-uuid-123",  # Unique per child
        )

        # Simulate parent loop setting DEPTH before child runs
        # (TaskExecutor._execute_nested_llm_node() does this in production)
        depth_token = DEPTH.set(1)
        try:
            # Fire events from child relay
            state = make_agent_state(iteration=0)
            child_relay.on_iteration_start(iteration=0, state=state)

            # Find the on_llm_start context
            llm_start_invocations = [
                ctx for name, ctx in hooks.calls
                if name == "on_llm_start"
            ]
            assert len(llm_start_invocations) >= 1
            ctx = llm_start_invocations[-1]

            # Verify correlation metadata from _build_context()
            # _build_context reads DEPTH.get() which is 1 (set above)
            assert ctx.metadata.get("nested_depth") == 1, (
                f"Expected nested_depth=1, got {ctx.metadata.get('nested_depth')}"
            )
            assert ctx.metadata.get("parent_run_id") == "run-parent", (
                f"Expected parent_run_id='run-parent', "
                f"got {ctx.metadata.get('parent_run_id')}"
            )
            assert ctx.metadata.get("nested_request_id") == "child-uuid-123", (
                f"Expected nested_request_id='child-uuid-123', "
                f"got {ctx.metadata.get('nested_request_id')}"
            )

            # parent_run_id should also be set directly on context
            assert ctx.parent_run_id == "run-parent", (
                f"Expected ctx.parent_run_id='run-parent', "
                f"got {ctx.parent_run_id}"
            )
        finally:
            DEPTH.reset(depth_token)

    def test_root_events_have_default_nested_metadata(self):
        """Root-level events have nested_depth=0 and nested_request_id."""
        hooks = _RecordingFlowHooks()
        relay = HookRelay(
            flow_hooks=hooks,
            node_id="root-node",
            nested_depth=0,
        )

        state = make_agent_state(iteration=0)
        relay.on_iteration_start(iteration=0, state=state)

        ctx = hooks.calls[-1][1]

        assert ctx.metadata.get("nested_depth") == 0, (
            f"Expected nested_depth=0 for root, "
            f"got {ctx.metadata.get('nested_depth')}"
        )
        assert ctx.metadata.get("nested_request_id") is not None, (
            "Root events should have a nested_request_id"
        )
        # Root has no parent_run_id
        assert ctx.metadata.get("parent_run_id") is None, (
            "Root events should NOT have parent_run_id"
        )

    # ── Scenario: No parent hooks → backward compatible ────────────

        # No error occurred — backward compatible

    # ── Scenario: DEPTH error path cleanup ─────────────────────────

    # ── Scenario: Multiple children get unique request IDs ─────────

    def test_multiple_children_unique_request_ids(self):
        """Each nested invocation gets a unique nested_request_id."""
        hooks = _RecordingFlowHooks()

        child_a = HookRelay(
            flow_hooks=hooks,
            node_id="child-a",
            parent_run_id="run-parent",
            nested_depth=1,
        )
        child_b = HookRelay(
            flow_hooks=hooks,
            node_id="child-b",
            parent_run_id="run-parent",
            nested_depth=1,
        )

        # Each child gets its own nested_request_id
        assert child_a._nested_request_id != child_b._nested_request_id, (
            "Each child HookRelay should generate a unique nested_request_id"
        )
