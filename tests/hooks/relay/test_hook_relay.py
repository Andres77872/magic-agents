"""Unit tests for HookRelay adapter bridging FlowHooks to AgentHooks.

Phase 9.3: Tests for:
- HookRelay implements AgentHooks Protocol
- Iteration start translates to llm start
- Tool start translates with full context
- Tool complete translates to tool end
- Sync hooks wrapped for async execution
- Error isolation in relay

Wave 3 / Phase 6 additions:
- P0-1: Sync fallback dispatch warning in _invoke_via_registry
- P1-9: Tool context routing (node_type="TOOL" verification)
"""
import asyncio
import logging
from unittest.mock import MagicMock, patch

from magic_llm.agent.hooks import AgentHooks
from magic_llm.agent.types import AgentState, ToolResult
from magic_llm.model import ModelChatResponse

from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.context_factory import HookContextFactory
from magic_agents.hooks.hook_relay import HookRelay
from magic_agents.hooks.hook_registry import HookRegistry


# ─── Helpers ──────────────────────────────────────────────────────────────────


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


class TestHookRelayProtocol:
    """Tests that HookRelay implements AgentHooks correctly."""

    def test_hook_relay_implements_agent_hooks(self):
        """HookRelay satisfies AgentHooks Protocol (isinstance check)."""
        relay = HookRelay()
        assert isinstance(relay, AgentHooks), (
            "HookRelay should implement AgentHooks Protocol"
        )

    def test_hook_relay_implements_all_agent_hooks_methods(self):
        """HookRelay defines all 6 AgentHooks Protocol methods."""
        relay = HookRelay()
        methods = [
            "on_iteration_start",
            "on_llm_response",
            "on_tool_start",
            "on_tool_complete",
            "on_loop_complete",
            "on_budget_exceeded",
        ]
        for method_name in methods:
            assert hasattr(relay, method_name), (
                f"HookRelay should implement '{method_name}'"
            )


class TestHookRelayTranslation:
    """Tests for event translation from AgentHooks to FlowHooks."""

    # ─── Helper: concrete sync FlowHooks that records calls ──────────────

    class _RecordingFlowHooks:
        """Sync FlowHooks that records method calls for assertion.

        Using sync methods avoids event loop issues — the relay calls
        sync methods directly instead of scheduling async futures.
        """

        def __init__(self):
            self.calls: list[tuple[str, HookContext]] = []

        def on_llm_start(self, context):
            self.calls.append(("on_llm_start", context))

        def on_llm_end(self, context):
            self.calls.append(("on_llm_end", context))

        def on_llm_loop_end(self, context):
            self.calls.append(("on_llm_loop_end", context))

        def on_tool_start(self, context):
            self.calls.append(("on_tool_start", context))

        def on_tool_end(self, context):
            self.calls.append(("on_tool_end", context))

        def on_node_error(self, context, **kwargs):
            self.calls.append(("on_node_error", context))

    def test_iteration_start_translates_to_llm_start(self):
        """on_iteration_start calls FlowHooks.on_llm_start with context."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="node-1", graph_id="graph-1")
        state = make_agent_state(iteration=2)

        relay.on_iteration_start(iteration=2, state=state)

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_llm_start"
        assert isinstance(ctx, HookContext)
        assert ctx.node_id == "node-1"
        assert ctx.execution_id == "graph-1"
        assert ctx.inputs.get("iteration") == 2

    def test_iteration_start_context_has_node_id(self):
        """Context injected with node_id from constructor."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(
            flow_hooks=hooks,
            node_id="llm-node-42",
            graph_id="graph-99",
        )
        state = make_agent_state()

        relay.on_iteration_start(iteration=1, state=state)

        assert len(hooks.calls) == 1
        _, ctx = hooks.calls[0]
        assert ctx.node_id == "llm-node-42"
        assert ctx.execution_id == "graph-99"
        assert ctx.node_type == "LLM"
        assert ctx.node_class == "NodeLLM"

    def test_iteration_start_sequence_number_increments(self):
        """Each relay call has an increasing sequence_number."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()

        relay.on_iteration_start(iteration=1, state=state)
        relay.on_iteration_start(iteration=2, state=state)

        assert len(hooks.calls) == 2
        assert hooks.calls[0][1].sequence_number == 1
        assert hooks.calls[1][1].sequence_number == 2

    def test_tool_start_translates_with_full_context(self):
        """on_tool_start passes tool name, id, and arguments."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()

        relay.on_tool_start(
            tool_name="calculator",
            tool_call_id="call-abc",
            arguments={"a": 1, "b": 2},
            state=state,
        )

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_tool_start"
        assert ctx.inputs["tool_name"] == "calculator"
        assert ctx.inputs["tool_call_id"] == "call-abc"
        assert ctx.inputs["arguments"] == {"a": 1, "b": 2}

    def test_tool_complete_translates_to_tool_end(self):
        """on_tool_complete calls FlowHooks.on_tool_end with result."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()
        result = make_tool_result(name="weather", content="25°C")

        relay.on_tool_complete(result=result, state=state)

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_tool_end"
        assert ctx.outputs["tool_name"] == "weather"
        assert ctx.outputs["result"] == "25°C"
        assert ctx.outputs["success"] is True

    def test_tool_complete_with_error(self):
        """Tool result with error is reflected in context."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()
        result = make_tool_result(
            name="broken_tool",
            content="",
            is_error=True,
            error="Division by zero",
        )

        relay.on_tool_complete(result=result, state=state)

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_tool_end"
        assert ctx.outputs["success"] is False
        assert ctx.error_message == "Division by zero"

    def test_llm_response_translates_to_llm_end(self):
        """on_llm_response calls FlowHooks.on_llm_end with response data."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()
        response = make_chat_response(model="gpt-4", content="Hello world")

        relay.on_llm_response(response=response, state=state)

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_llm_end"
        assert ctx.outputs["model"] == "gpt-4"
        assert ctx.outputs["content"] == "Hello world"
        assert ctx.outputs["finish_reason"] == "stop"

    def test_loop_complete_translates_to_llm_loop_end(self):
        """on_loop_complete calls FlowHooks.on_llm_loop_end with aggregation."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state(iteration=5)
        response = make_chat_response(content="Final answer")

        relay.on_loop_complete(final_response=response, state=state)

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_llm_loop_end"
        assert ctx.outputs["content"] == "Final answer"
        assert ctx.outputs["iteration"] == 5       # 0-indexed
        assert ctx.outputs["total_iterations"] == 6  # iteration+1 = 1-indexed count

    def test_budget_exceeded_translates_to_node_error(self):
        """on_budget_exceeded calls FlowHooks.on_node_error."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")

        relay.on_budget_exceeded(budget_type="time", details="30s limit hit")

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_node_error"
        assert ctx.error_message == "Budget exceeded (time): 30s limit hit"
        assert ctx.inputs["budget_type"] == "time"

    def test_no_hooks_no_error(self):
        """Relay with no FlowHooks or Registry doesn't crash."""
        relay = HookRelay()
        state = make_agent_state()

        # None of these should raise
        relay.on_iteration_start(iteration=1, state=state)
        relay.on_tool_start("calc", "c1", {"x": 1}, state)
        relay.on_llm_response(make_chat_response(), state)
        state2 = make_agent_state(iteration=1)
        relay.on_tool_complete(make_tool_result(name="calc", content="2"), state2)
        relay.on_loop_complete(make_chat_response(), state)
        relay.on_budget_exceeded("time", "exceeded")


class TestHookRelayErrorIsolation:
    """Tests that relay isolates hook errors."""

    def test_failing_method_does_not_block_subsequent_calls(self):
        """After a failing hook call, subsequent calls still work."""
        calls: list[str] = []

        class TrackingFlowHooks:
            def on_llm_start(self, context):
                msg = "intentional failure"
                raise RuntimeError(msg)

            def on_tool_start(self, context):
                calls.append("tool_start")

        relay = HookRelay(flow_hooks=TrackingFlowHooks(), node_id="n1")
        state = make_agent_state()

        # This should not raise
        relay.on_iteration_start(iteration=1, state=state)
        # Subsequent call should work
        relay.on_tool_start("calc", "c1", {"x": 1}, state)

        assert calls == ["tool_start"], (
            "Subsequent hook calls should still work after a failure"
        )

class TestHookRelayPartialImplementation:
    """Tests that relay handles partial FlowHooks implementations."""

    def test_partial_flow_hooks_implementation(self):
        """Relay only invokes methods that exist on the FlowHooks object."""

        class PartialFlowHooks:
            def on_llm_start(self, context):
                self.llm_start_called = True

        hooks = PartialFlowHooks()
        # Need to set attribute before it's called
        hooks.llm_start_called = False

        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()

        # on_llm_start exists — should be called
        relay.on_iteration_start(iteration=1, state=state)
        assert hooks.llm_start_called is True

        # No assertion needed for the missing method — it was skipped silently


class TestStandaloneHookRelayCollector:
    """Tests for HookRelay as standalone collector without registry/flow_hooks.

    This is the fix for graph-agent NodeLLM tools with stream:true, iterate:false,
    and no configured hooks. A standalone HookRelay (no registry, no flow_hooks)
    must still collect on_tool_start/on_tool_complete data so that the streaming
    tool path can emit TOOL_CALL/TOOL_RESULT debug events for persistence.
    """

    def make_agent_state(self, **overrides) -> AgentState:
        state = MagicMock(spec=AgentState)
        state.iteration = overrides.get("iteration", 0)
        return state

    def make_tool_result(self, **overrides) -> ToolResult:
        return ToolResult(
            name=overrides.get("name", "test_tool"),
            content=overrides.get("content", "result"),
            is_error=overrides.get("is_error", False),
            error=overrides.get("error", None),
            tool_call_id=overrides.get("tool_call_id", "call-1"),
        )

    def test_standalone_relay_multiple_tools_collected(self):
        relay = HookRelay()
        state = self.make_agent_state()

        relay.on_tool_start("tool_a", "call-1", {"x": 1}, state)
        relay.on_tool_complete(
            self.make_tool_result(name="tool_a", content="a", tool_call_id="call-1"),
            state,
        )
        relay.on_tool_start("tool_b", "call-2", {"y": 2}, state)
        relay.on_tool_complete(
            self.make_tool_result(name="tool_b", content="b", tool_call_id="call-2"),
            state,
        )

        events = relay.get_collected_tool_data_for_yield()
        assert len(events) == 4
        assert events[0]["data"]["tool_calls"][0]["function"]["name"] == "tool_a"
        assert events[1]["data"]["tool_calls"][0]["function"]["name"] == "tool_b"
        assert events[3]["data"]["content"] == "b"

    def test_standalone_relay_invoke_is_noop(self):
        relay = HookRelay(node_id="test-node")
        state = self.make_agent_state()

        relay.on_iteration_start(iteration=0, state=state)
        relay.on_llm_response(MagicMock(spec=ModelChatResponse), state)
        relay.on_loop_complete(MagicMock(spec=ModelChatResponse), state)
        relay.on_budget_exceeded("time", "oops")

        assert len(relay._collected_tool_calls) == 0
        assert len(relay._collected_tool_results) == 0

class TestHookRelayUsageDetailFields:
    """P1.7 (a): HookRelay.on_llm_response() forwards all 8 UsageModel fields.

    Verify that on_llm_response() passes all 8 usage fields (3 basic + 5 detail)
    in the outputs dict, for both dict-style and UsageModel object-style usage.
    """

    class _RecordingFlowHooks:
        """Sync FlowHooks that records on_llm_end context for assertion."""

        def __init__(self):
            self.last_context = None

        def on_llm_end(self, context):
            self.last_context = context

    def _make_usage_dict(self, **overrides):
        """Build a realistic usage dict with all 8 fields."""
        return {
            "prompt_tokens": overrides.get("prompt_tokens", 10),
            "completion_tokens": overrides.get("completion_tokens", 5),
            "total_tokens": overrides.get("total_tokens", 15),
            "cached_tokens_read": overrides.get("cached_tokens_read", 2),
            "cached_tokens_write": overrides.get("cached_tokens_write", 1),
            "reasoning_tokens": overrides.get("reasoning_tokens", 3),
            "audio_tokens": overrides.get("audio_tokens", 0),
            "raw_usage_json": overrides.get("raw_usage_json", {"raw": "data"}),
        }

    def test_dict_style_all_8_fields_forwarded(self):
        """Dict-style usage: all 8 fields appear in outputs dict."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()
        response = make_chat_response()

        usage = self._make_usage_dict()
        response.usage = usage
        relay.on_llm_response(response=response, state=state)

        outputs = hooks.last_context.outputs
        assert outputs.get("prompt_tokens") == 10
        assert outputs.get("completion_tokens") == 5
        assert outputs.get("total_tokens") == 15
        assert outputs.get("cached_tokens_read") == 2
        assert outputs.get("cached_tokens_write") == 1
        assert outputs.get("reasoning_tokens") == 3
        assert outputs.get("audio_tokens") == 0
        assert outputs.get("raw_usage_json") == {"raw": "data"}

    def test_dict_style_none_detail_fields_forwarded_as_none(self):
        """Dict-style: None detail fields forwarded as None, not omitted."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()
        response = make_chat_response()

        response.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        relay.on_llm_response(response=response, state=state)

        outputs = hooks.last_context.outputs
        # Detail fields should exist with None values (not missing keys)
        assert "cached_tokens_read" in outputs
        assert outputs["cached_tokens_read"] is None
        assert "cached_tokens_write" in outputs
        assert outputs["cached_tokens_write"] is None
        assert "reasoning_tokens" in outputs
        assert outputs["reasoning_tokens"] is None
        assert "audio_tokens" in outputs
        assert outputs["audio_tokens"] is None
        assert "raw_usage_json" in outputs
        assert outputs["raw_usage_json"] is None
        # Basic fields still present
        assert outputs["prompt_tokens"] == 10
        assert outputs["total_tokens"] == 15

    def test_object_style_all_8_fields_forwarded(self):
        """UsageModel object-style: all 8 fields forwarded from attributes."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()
        response = make_chat_response()

        # Simulate UsageModel object with attributes
        class FakeUsage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15
            cached_tokens_read = 2
            cached_tokens_write = 1
            reasoning_tokens = 3
            audio_tokens = 0
            raw_usage_json = '{"raw": "data"}'
            prompt_tokens_details = None
            completion_tokens_details = None

        response.usage = FakeUsage()
        relay.on_llm_response(response=response, state=state)

        outputs = hooks.last_context.outputs
        assert outputs.get("prompt_tokens") == 10
        assert outputs.get("completion_tokens") == 5
        assert outputs.get("total_tokens") == 15
        assert outputs.get("cached_tokens_read") == 2
        assert outputs.get("cached_tokens_write") == 1
        assert outputs.get("reasoning_tokens") == 3
        assert outputs.get("audio_tokens") == 0
        assert outputs.get("raw_usage_json") == '{"raw": "data"}'

    def test_object_style_nested_detail_fields_fallback(self):
        """UsageModel with nested prompt_tokens_details/completion_tokens_details."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()
        response = make_chat_response()

        class FakeCompletionDetails:
            reasoning_tokens = 7
            audio_tokens = 2

        class FakePromptDetails:
            cached_tokens = 5
            audio_tokens = 3

        class FakeUsageWithDetails:
            prompt_tokens = 20
            completion_tokens = 15
            total_tokens = 35
            cached_tokens_read = None   # triggers fallback to details
            cached_tokens_write = 0
            reasoning_tokens = None     # triggers fallback to details
            audio_tokens = None         # triggers fallback to sum of details
            raw_usage_json = None
            prompt_tokens_details = FakePromptDetails()
            completion_tokens_details = FakeCompletionDetails()

        response.usage = FakeUsageWithDetails()
        relay.on_llm_response(response=response, state=state)

        outputs = hooks.last_context.outputs
        # Fallback: cached_tokens_read from prompt_tokens_details.cached_tokens
        assert outputs.get("cached_tokens_read") == 5
        # Fallback: reasoning_tokens from completion_tokens_details.reasoning_tokens
        assert outputs.get("reasoning_tokens") == 7
        # Fallback: audio_tokens = prompt_details.audio + completion_details.audio
        assert outputs.get("audio_tokens") == 5

    def test_usage_is_none_does_not_crash(self):
        """response.usage=None edge case: no crash, outputs have expected default fields."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()
        response = make_chat_response()
        response.usage = None

        relay.on_llm_response(response=response, state=state)

        outputs = hooks.last_context.outputs
        # Basic fields still present (from chat response)
        assert outputs.get("model") == "gpt-4"
        assert outputs.get("content") == "Hello!"
        # The relay preserves a stable telemetry schema even without usage.
        assert outputs["prompt_tokens"] is None
        assert outputs["completion_tokens"] is None
        assert outputs["total_tokens"] is None


# ── P0-1: Sync Fallback Warning (6.1) ──────────────────────────────────


# ── P1-9: Tool Context Routing ────────────────────────────────────────────


class TestToolContextRouting:
    """P1-9: Tests for _build_tool_context() node_type routing.

    Verifies that tool events carry node_type='TOOL' and LLM events
    continue to carry node_type='LLM'.
    """

    class _RecordingFlowHooks:
        def __init__(self):
            self.calls: list[tuple[str, HookContext]] = []

        def on_tool_start(self, context):
            self.calls.append(("on_tool_start", context))

        def on_tool_end(self, context):
            self.calls.append(("on_tool_end", context))

        def on_llm_start(self, context):
            self.calls.append(("on_llm_start", context))

        def on_llm_end(self, context):
            self.calls.append(("on_llm_end", context))

    def test_tool_start_context_has_node_type_tool(self):
        """on_tool_start context carries node_type='TOOL' and tool-specific fields."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state(iteration=2)
        relay._current_provider_request_id = "req-llm-1"

        relay.on_tool_start(
            tool_name="calculator",
            tool_call_id="call-abc",
            arguments={"a": 1, "b": 2},
            state=state,
        )

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_tool_start"
        assert ctx.node_type == "TOOL", (
            f"Tool start context should have node_type='TOOL', got '{ctx.node_type}'"
        )
        assert ctx.inputs["tool_name"] == "calculator"
        assert ctx.inputs["tool_call_id"] == "call-abc"
        assert ctx.inputs["arguments"] == {"a": 1, "b": 2}
        assert ctx.metadata.get("provider_request_id") == "req-llm-1" or ctx.inputs.get("provider_request_id") == "req-llm-1", (
            "Tool context should carry provider_request_id from cached LLM response"
        )
        assert ctx.metadata.get("nested_depth") is not None, (
            "Tool context should have nested_depth metadata"
        )

    def test_tool_complete_context_has_node_type_tool(self):
        """on_tool_complete calls on_tool_end with node_type='TOOL'."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state(iteration=2)
        result = make_tool_result(
            name="weather",
            content="25°C",
            tool_call_id="call-123",
        )
        relay._current_provider_request_id = "req-llm-1"

        relay.on_tool_complete(result=result, state=state)

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_tool_end"
        assert ctx.node_type == "TOOL", (
            f"Tool end context should have node_type='TOOL', got '{ctx.node_type}'"
        )
        assert ctx.outputs["tool_name"] == "weather"
        assert ctx.outputs["success"] is True

    def test_llm_end_unaffected(self):
        """on_llm_response still produces node_type='LLM'."""
        hooks = self._RecordingFlowHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()
        response = make_chat_response()

        relay.on_llm_response(response=response, state=state)

        assert len(hooks.calls) == 1
        name, ctx = hooks.calls[0]
        assert name == "on_llm_end"
        assert ctx.node_type == "LLM", (
            f"LLM end context should have node_type='LLM', got '{ctx.node_type}'"
        )


class TestHookRelaySyncDispatchWarning:
    """P0-1 (Task 6.1.3): Sync dispatch warning when no event loop available.

    When _invoke_via_registry() or _invoke_async_method() catches RuntimeError
    from asyncio.get_running_loop(), a WARNING must be logged containing the
    hook_name so users can identify which hooks are being dropped.
    """

    def test_sync_dispatch_via_registry_logs_warning_with_hook_name(self, caplog):
        """WARNING with hook_name logged when _invoke_via_registry has no loop."""
        registry = HookRegistry()
        relay = HookRelay(registry=registry, node_id="n1")

        ctx = HookContextFactory.build_node_context(
            execution_id="e1", run_id="r1", node_id="n1"
        )

        caplog.set_level(logging.WARNING)

        # Mock get_running_loop to raise RuntimeError (simulate thread context)
        with patch.object(asyncio, "get_running_loop", side_effect=RuntimeError("no loop")):
            relay._invoke_via_registry("on_llm_start", ctx)

        # Verify warning contains the hook name
        assert len(caplog.records) >= 1, "Expected at least one WARNING record"
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        hook_warnings = [m for m in warning_messages if "on_llm_start" in m]
        assert len(hook_warnings) >= 1, (
            f"Expected warning containing 'on_llm_start' in: {warning_messages}"
        )
        assert any(
            "will NOT be delivered" in m or "cannot invoke" in m
            for m in hook_warnings
        ), f"Warning should indicate hook is dropped: {hook_warnings}"

    def test_sync_dispatch_async_method_logs_warning_with_hook_name(self, caplog):
        """WARNING with hook_name logged when _invoke_async_method has no loop."""
        relay = HookRelay(flow_hooks=HookRegistry(), node_id="n1")
        ctx = HookContextFactory.build_node_context(
            execution_id="e1", run_id="r1", node_id="n1"
        )

        # Create an async method to trigger _invoke_async_method path
        async def fake_method(ctx):
            pass

        caplog.set_level(logging.WARNING)

        with patch.object(asyncio, "get_running_loop", side_effect=RuntimeError("no loop")):
            relay._invoke_async_method(fake_method, "on_llm_end", ctx)

        assert len(caplog.records) >= 1, "Expected at least one WARNING record"
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        hook_warnings = [m for m in warning_messages if "on_llm_end" in m]
        assert len(hook_warnings) >= 1, (
            f"Expected warning containing 'on_llm_end' in: {warning_messages}"
        )
        assert any(
            "will NOT be delivered" in m or "cannot invoke" in m
            for m in hook_warnings
        ), f"Warning should indicate hook is dropped: {hook_warnings}"
