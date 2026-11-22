"""
Integration Tests: NodeLLM → magic-llm HookRelay (Task 9.6).

Tests that:
1. HookRelay implements magic-llm's AgentHooks Protocol (isinstance check)
2. on_iteration_start translates to on_llm_start with correct context
3. on_tool_start translates with full arguments (not empty dict)
4. on_tool_complete translates to on_tool_end with result context
5. on_llm_response translates to on_llm_end with response context
6. Context injection includes node_id, graph_id, run_id
7. Error isolation: relay exceptions are logged, not propagated
8. on_loop_complete and on_budget_exceeded translate correctly
"""
import pytest
from unittest.mock import MagicMock

from magic_agents.hooks.hook_relay import HookRelay
from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_registry import HookRegistry


# ─── Helper: Recording FlowHooks ──────────────────────────────────────────────

class RecordingFlowHooks:
    """FlowHooks implementation that records all invocations for verification.

    Methods are SYNC (not async) because HookRelay invokes them from a sync
    context (AgentHooks Protocol is sync). Sync methods are called directly
    without needing asyncio.ensure_future scheduling.
    """

    def __init__(self):
        self.calls: list[dict] = []  # [{hook_name, context, kwargs}]

    def on_llm_start(self, context: HookContext, **kwargs) -> None:
        self.calls.append({"hook_name": "on_llm_start", "context": context, "kwargs": kwargs})

    def on_llm_end(self, context: HookContext, **kwargs) -> None:
        self.calls.append({"hook_name": "on_llm_end", "context": context, "kwargs": kwargs})

    def on_llm_loop_end(self, context: HookContext, **kwargs) -> None:
        self.calls.append({"hook_name": "on_llm_loop_end", "context": context, "kwargs": kwargs})

    def on_tool_start(self, context: HookContext, **kwargs) -> None:
        self.calls.append({"hook_name": "on_tool_start", "context": context, "kwargs": kwargs})

    def on_tool_end(self, context: HookContext, **kwargs) -> None:
        self.calls.append({"hook_name": "on_tool_end", "context": context, "kwargs": kwargs})

    def on_node_error(self, context: HookContext, **kwargs) -> None:
        self.calls.append({"hook_name": "on_node_error", "context": context, "kwargs": kwargs})


# ─── Helpers ───────────────────────────────────────────────────────────────────

def make_agent_state(iteration: int = 0) -> MagicMock:
    """Create a mock AgentState with the given iteration."""
    state = MagicMock()
    state.iteration = iteration
    return state


class _MockToolResult:
    """Simple mock for ToolResult to avoid MagicMock attribute behavior.

    ToolResult in magic-llm uses attribute name 'content' (not 'output').
    Production hook_relay.py reads: getattr(result, 'content', '').
    """
    def __init__(self, name: str = "test_tool", content: str = "result", error: str = None,
                 is_error: bool = False, tool_call_id: str = "", duration_ms: float | None = None):
        self.name = name
        self.content = content
        self.error = error
        self.is_error = is_error
        self.tool_call_id = tool_call_id
        self.duration_ms = duration_ms


def make_tool_result(name: str = "test_tool", content: str = "result", error: str = None,
                     tool_call_id: str = "", duration_ms: float | None = None) -> _MockToolResult:
    """Create a mock ToolResult with proper attribute values."""
    return _MockToolResult(name=name, content=content, error=error,
                          tool_call_id=tool_call_id, duration_ms=duration_ms)


def make_model_response(content: str = "hello", model: str = "gpt-4") -> MagicMock:
    """Create a mock ModelChatResponse with proper attribute values."""
    resp = MagicMock(spec=['content', 'model', 'finish_reason'])
    resp.content = content
    resp.model = model
    resp.finish_reason = "stop"
    return resp


# ─── Tests ─────────────────────────────────────────────────────────────────────

class TestHookRelayImplementsAgentHooks:
    """9.6.2: HookRelay must implement magic-llm's AgentHooks Protocol."""

    def test_hook_relay_is_agent_hooks(self):
        """HookRelay must pass isinstance check for AgentHooks Protocol."""
        from magic_llm.agent.hooks import AgentHooks
        relay = HookRelay(flow_hooks=RecordingFlowHooks())
        assert isinstance(relay, AgentHooks), (
            "HookRelay must implement AgentHooks Protocol for magic-llm compatibility"
        )

class TestIterationStartTranslation:
    """9.6.3: on_iteration_start translates to on_llm_start."""

    @pytest.mark.asyncio
    async def test_iteration_start_calls_llm_start(self):
        """Iteration start must invoke on_llm_start on the FlowHooks."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        relay.on_iteration_start(0, make_agent_state(iteration=0))

        # Flush pending async hook tasks instead of race-condition sleep
        await relay.flush_pending_hooks()

        assert len(hooks.calls) >= 1
        llm_start_calls = [c for c in hooks.calls if c["hook_name"] == "on_llm_start"]
        assert len(llm_start_calls) == 1

        ctx = llm_start_calls[0]["context"]
        assert ctx.node_id == "llm-1"
        assert ctx.execution_id == "g-1"
        assert ctx.run_id == "r-1"
        assert ctx.inputs.get("iteration") == 0

    @pytest.mark.asyncio
    async def test_iteration_start_injects_sequence_number(self):
        """Each invocation must get an incrementing sequence_number."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        relay.on_iteration_start(0, make_agent_state())
        relay.on_iteration_start(1, make_agent_state(iteration=1))

        await relay.flush_pending_hooks()

        llm_starts = [c for c in hooks.calls if c["hook_name"] == "on_llm_start"]
        assert len(llm_starts) == 2
        assert llm_starts[0]["context"].sequence_number == 1  # first one (0 + _next)
        assert llm_starts[1]["context"].sequence_number == 2  # second one


class TestToolStartTranslation:
    """9.6.4/9.6.5: on_tool_start translates with full arguments."""

    @pytest.mark.asyncio
    async def test_tool_start_translates_with_arguments(self):
        """Tool start must pass actual arguments (NOT empty dict)."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        tool_args = {"query": "weather in Tokyo", "units": "metric"}
        relay.on_tool_start("weather", "call_abc123", tool_args, make_agent_state())

        await relay.flush_pending_hooks()

        tool_starts = [c for c in hooks.calls if c["hook_name"] == "on_tool_start"]
        assert len(tool_starts) == 1

        ctx = tool_starts[0]["context"]
        assert ctx.inputs.get("tool_name") == "weather"
        assert ctx.inputs.get("tool_call_id") == "call_abc123"
        assert ctx.inputs.get("arguments") == tool_args
        assert ctx.inputs["arguments"] != {}, "Arguments must NOT be empty dict"
        assert ctx.node_id == "llm-1"

    @pytest.mark.asyncio
    async def test_multiple_tool_starts_before_execution(self):
        """Multiple tool calls must each produce a separate on_tool_start."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        relay.on_tool_start("search", "call_1", {"q": "a"}, make_agent_state())
        relay.on_tool_start("search", "call_2", {"q": "b"}, make_agent_state())
        relay.on_tool_start("calc", "call_3", {"expr": "1+1"}, make_agent_state())

        await relay.flush_pending_hooks()

        tool_starts = [c for c in hooks.calls if c["hook_name"] == "on_tool_start"]
        assert len(tool_starts) == 3
        contexts = [event["context"] for event in tool_starts]
        assert len({id(context) for context in contexts}) == 3
        # HookRegistry intentionally dispatches independent callbacks in
        # parallel, so completion order is not a lifecycle-order contract.
        # HookRelay's monotonic sequence number is the authoritative ordering.
        assert len({context.sequence_number for context in contexts}) == 3
        ordered_contexts = sorted(contexts, key=lambda context: context.sequence_number)
        assert [context.inputs["tool_call_id"] for context in ordered_contexts] == [
            "call_1", "call_2", "call_3",
        ]
        assert [context.inputs["arguments"] for context in ordered_contexts] == [
            {"q": "a"}, {"q": "b"}, {"expr": "1+1"},
        ]


class TestToolCompleteTranslation:
    """on_tool_complete translates to on_tool_end."""

    @pytest.mark.asyncio
    async def test_tool_complete_calls_tool_end(self):
        """Tool complete must invoke on_tool_end with result context."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        result = make_tool_result(name="search", content="Tokyo: 25°C")
        relay.on_tool_complete(result, make_agent_state())

        await relay.flush_pending_hooks()

        tool_ends = [c for c in hooks.calls if c["hook_name"] == "on_tool_end"]
        assert len(tool_ends) == 1
        ctx = tool_ends[0]["context"]
        assert ctx.outputs.get("tool_name") == "search"
        assert ctx.outputs.get("result") == "Tokyo: 25°C"

    @pytest.mark.asyncio
    async def test_tool_complete_with_error(self):
        """Tool complete with error must include error_message in context."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        result = make_tool_result(name="search", content="", error="API timeout")
        relay.on_tool_complete(result, make_agent_state())

        await relay.flush_pending_hooks()

        tool_ends = [c for c in hooks.calls if c["hook_name"] == "on_tool_end"]
        assert len(tool_ends) == 1
        ctx = tool_ends[0]["context"]
        assert ctx.error_message == "API timeout"
        assert ctx.outputs.get("success") is False


class TestLLMResponseTranslation:
    """on_llm_response translates to on_llm_end."""

    @pytest.mark.asyncio
    async def test_llm_response_calls_llm_end(self):
        """LLM response must invoke on_llm_end with response context."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        response = make_model_response(content="Hello!", model="gpt-4")
        relay.on_llm_response(response, make_agent_state())

        await relay.flush_pending_hooks()

        llm_ends = [c for c in hooks.calls if c["hook_name"] == "on_llm_end"]
        assert len(llm_ends) == 1
        ctx = llm_ends[0]["context"]
        assert ctx.outputs.get("content") == "Hello!"
        assert ctx.outputs.get("model") == "gpt-4"
        assert ctx.node_type == "LLM"


class TestLoopCompleteTranslation:
    """on_loop_complete translates to on_llm_loop_end."""

    @pytest.mark.asyncio
    async def test_loop_complete_fires_llm_loop_end(self):
        """Loop complete must invoke on_llm_loop_end with loop metadata."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        response = make_model_response(content="Final answer")
        relay.on_loop_complete(response, make_agent_state(iteration=3))

        await relay.flush_pending_hooks()

        loop_ends = [c for c in hooks.calls if c["hook_name"] == "on_llm_loop_end"]
        assert len(loop_ends) == 1
        ctx = loop_ends[0]["context"]
        assert ctx.outputs.get("iteration") == 3       # 0-indexed
        assert ctx.outputs.get("total_iterations") == 4  # iteration+1 = 1-indexed count
        assert ctx.outputs.get("content") == "Final answer"


class TestBudgetExceededTranslation:
    """on_budget_exceeded translates to on_node_error."""

    @pytest.mark.asyncio
    async def test_budget_exceeded_fires_node_error(self):
        """Budget exceeded must invoke on_node_error with error context."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(registry=registry, node_id="llm-1", graph_id="g-1", run_id="r-1")

        relay.on_budget_exceeded("token", "Exceeded 4000 tokens")

        await relay.flush_pending_hooks()

        node_errors = [c for c in hooks.calls if c["hook_name"] == "on_node_error"]
        assert len(node_errors) == 1
        ctx = node_errors[0]["context"]
        assert "Budget exceeded" in ctx.error_message
        assert ctx.inputs.get("budget_type") == "token"
        error = node_errors[0]["kwargs"]["error"]
        assert isinstance(error, RuntimeError)
        assert str(error) == ctx.error_message
        assert ctx.error is error
        assert ctx.error_type == "RuntimeError"


class TestErrorIsolation:
    """9.6.3: Hook exceptions must be isolated (logged, not propagated)."""

    async def test_raising_async_hook_is_isolated_and_later_hooks_still_run(self):
        """An async hook that raises is awaited at flush, logged, and never propagates."""
        calls = []

        class RaisingHooks:
            async def on_llm_start(self, context, **kwargs):
                calls.append("llm_start")
                raise ValueError("Hook failed intentionally")

            async def on_tool_start(self, context, **kwargs):
                calls.append("tool_start")

        relay = HookRelay(flow_hooks=RaisingHooks(), node_id="llm-1", graph_id="g-1", run_id="r-1")

        relay.on_iteration_start(0, make_agent_state())
        relay.on_tool_start("calc", "c1", {"x": 1}, make_agent_state())
        await relay.flush_pending_hooks()

        assert calls == ["llm_start", "tool_start"]


class TestContextInjection:
    """All relayed events must inject node_id, graph_id, run_id into context."""

    @pytest.mark.asyncio
    async def test_all_methods_inject_execution_identity(self):
        """Every relay method must inject node_id, execution_id, run_id."""
        hooks = RecordingFlowHooks()
        registry = HookRegistry()
        registry.register_graph(hooks)
        relay = HookRelay(
            registry=registry,
            node_id="test-llm-node",
            graph_id="test-graph-123",
            run_id="test-run-456",
        )

        # Trigger multiple events
        relay.on_iteration_start(0, make_agent_state())
        relay.on_tool_start("search", "call_1", {"q": "hello"}, make_agent_state())
        relay.on_llm_response(make_model_response(), make_agent_state())
        relay.on_tool_complete(make_tool_result(), make_agent_state())
        relay.on_loop_complete(make_model_response(), make_agent_state())

        await relay.flush_pending_hooks()

        assert len(hooks.calls) >= 5, "Expected at least 5 hook invocations"
        for call in hooks.calls:
            ctx = call["context"]
            assert ctx.node_id == "test-llm-node", f"{call['hook_name']}: wrong node_id"
            assert ctx.execution_id == "test-graph-123", f"{call['hook_name']}: wrong execution_id"
            assert ctx.run_id == "test-run-456", f"{call['hook_name']}: wrong run_id"
            # Tool events correctly carry node_type="TOOL" (P1-9); LLM events carry "LLM"
            if call["hook_name"] in ("on_tool_start", "on_tool_end"):
                assert ctx.node_type == "TOOL", f"{call['hook_name']}: expected TOOL got {ctx.node_type}"
            else:
                assert ctx.node_type == "LLM", f"{call['hook_name']}: wrong node_type"


# ─── Task 4.8: Tool Collection Clear Tests (F18/M5) ─────────────────────


class TestToolCollectionClear:
    """Tool collection clear behavior (F18/M5)."""

    def make_agent_state(self):
        from unittest.mock import MagicMock
        state = MagicMock()
        state.iteration = 0
        return state

    def test_clear_true_empties_after_read(self):
        """clear=True (default) empties internal lists after read."""
        relay = HookRelay()
        state = self.make_agent_state()

        # Accumulate tool data
        relay.on_tool_start("search", "call-1", {"q": "hello"}, state)
        relay.on_tool_complete(
            make_tool_result(name="search", content="result", tool_call_id="call-1"),
            state,
        )

        # First read with clear=True (default)
        events1 = relay.get_collected_tool_data_for_yield()
        assert len(events1) == 2, "First read must return all accumulated data"

        # Second read should be empty
        events2 = relay.get_collected_tool_data_for_yield()
        assert len(events2) == 0, (
            "Second read must return empty after clear=True"
        )

    def test_clear_false_preserves_data(self):
        """clear=False preserves data for re-read."""
        relay = HookRelay()
        state = self.make_agent_state()

        relay.on_tool_start("search", "call-1", {"q": "hello"}, state)
        relay.on_tool_complete(
            make_tool_result(name="search", content="result", tool_call_id="call-1"),
            state,
        )

        # Read with clear=False
        events1 = relay.get_collected_tool_data_for_yield(clear=False)
        assert len(events1) == 2, "First read must return all data"

        # Second read with clear=False — same data
        events2 = relay.get_collected_tool_data_for_yield(clear=False)
        assert len(events2) == 2, (
            "Second read with clear=False must return same data"
        )
        assert events1 == events2, "Both reads must return identical data"

    def test_new_events_after_clear_tracked_correctly(self):
        """New tool events after clear are tracked correctly (no duplication)."""
        relay = HookRelay()
        state = self.make_agent_state()

        # First batch
        relay.on_tool_start("tool_a", "call-1", {"x": 1}, state)
        relay.on_tool_complete(
            make_tool_result(name="tool_a", content="a", tool_call_id="call-1"),
            state,
        )

        events1 = relay.get_collected_tool_data_for_yield()  # clear=True
        assert len(events1) == 2

        # Second batch — new data
        relay.on_tool_start("tool_b", "call-2", {"y": 2}, state)
        relay.on_tool_complete(
            make_tool_result(name="tool_b", content="b", tool_call_id="call-2"),
            state,
        )

        events2 = relay.get_collected_tool_data_for_yield()
        assert len(events2) == 2, "New events after clear must be tracked correctly"
        assert events2[0]["data"]["tool_calls"][0]["function"]["name"] == "tool_b", (
            "Only new tool data should be in the second read"
        )
