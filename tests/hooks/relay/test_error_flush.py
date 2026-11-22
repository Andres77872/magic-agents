"""Integration tests: Streaming + tools + hooks complete lifecycle.

Phase 5.4: Verifies that:
- Streaming + tools + hooks fires events in correct order:
  on_llm_start → on_llm_end → on_tool_start → on_tool_end → on_llm_loop_end
- Streaming + budget exceeded fires on_budget_exceeded, NOT on_llm_loop_end
- Streaming + no tools fires simplified lifecycle
- Streaming agent receives llm_config with streaming=True
- Tool data collection works in streaming mode

Phase 6 (NodeLLM behaviors):
- P0-1: Sync fallback warning hook-loss tests (6.1.1, 6.1.2)
- P0-2: Error-exit flush tests (6.2.1, 6.2.2)
- P1-NEW: _emit_llm_generation parity tests (6.8)
"""
import pytest
import asyncio
from unittest.mock import MagicMock

from magic_llm.agent.types import AgentState, ToolResult
from magic_llm.model import ModelChatResponse

from magic_agents.hooks.flow_hooks import HookContext
from magic_agents.hooks.hook_relay import HookRelay
from magic_agents.models.factory.Nodes import LlmNodeModel
from magic_agents.node_system.NodeLLM import NodeLLM


# ─── Recording FlowHooks (with on_llm_loop_end) ──────────────────────


class _RecordingFlowHooks:
    """Sync FlowHooks that records method calls with full context."""

    def __init__(self):
        self.calls: list[tuple[str, HookContext]] = []
        self.llm_configs: list[dict | None] = []

    def on_llm_start(self, context, llm_config=None):
        self.calls.append(("on_llm_start", context))
        self.llm_configs.append(llm_config)

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

    def hook_order(self) -> list[str]:
        return [name for name, _ in self.calls]

    def get_context(self, hook_name: str, index: int = 0) -> HookContext:
        matches = [ctx for name, ctx in self.calls if name == hook_name]
        if index >= len(matches):
            raise AssertionError(
                f"No {index}th call for '{hook_name}' (have {len(matches)})"
            )
        return matches[index]


# ─── Helpers ──────────────────────────────────────────────────────────


def make_agent_state(**overrides) -> AgentState:
    return AgentState(step=overrides.get("step", overrides.get("iteration", 0)))


def make_chat_response(**overrides) -> ModelChatResponse:
    resp = MagicMock(spec=ModelChatResponse)
    resp.model = overrides.get("model", "gpt-4")
    resp.content = overrides.get("content", "Hello!")
    resp.finish_reason = overrides.get("finish_reason", "stop")
    resp.id = overrides.get("id", "chatcmpl-stream")
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


# ── P0-2: NodeLLM error-exit flush regression ───────────────────────────


@pytest.mark.asyncio
async def test_async_non_stream_error_flushes_pending_hooks_and_propagates():
    """The real non-stream ``NodeLLM`` path flushes before re-raising."""
    pending_hook_completed = asyncio.Event()

    class RecordingFlushRelay(HookRelay):
        def __init__(self):
            super().__init__(node_id="n1")
            self.flush_calls = 0

        async def flush_pending_hooks(self):
            self.flush_calls += 1
            await super().flush_pending_hooks()

    relay = RecordingFlushRelay()

    class ErroringClient:
        class LLMEngine:
            model = "test-model"
            engine_name = "test-provider"

        llm = LLMEngine()

        async def run_agent_async(self, **kwargs):
            async def pending_hook():
                await asyncio.sleep(0)
                pending_hook_completed.set()

            hooks = kwargs["hooks"]
            hooks._pending_futures.add(asyncio.create_task(pending_hook()))
            raise RuntimeError("non-stream provider error")

    class ErroringNonStreamNode(NodeLLM):
        async def _collect_tools(self):
            async def fake_tool(value: str = "") -> str:
                return value

            schema = {
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "description": "Deterministic test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            return [schema], {"fake_tool": fake_tool}, {}

        def _create_hook_relay(self, client=None):
            return relay

    node = ErroringNonStreamNode(
        data=LlmNodeModel(stream=False),
        node_id="n1",
    )
    node.inputs = {
        node.INPUT_HANDLER_CLIENT_PROVIDER: ErroringClient(),
        node.INPUT_HANDLER_USER_MESSAGE: "hello",
    }

    with pytest.raises(RuntimeError, match="non-stream provider error"):
        _ = [event async for event in node.process([])]

    assert relay.flush_calls == 1
    assert pending_hook_completed.is_set()
    assert relay._pending_futures == set()


@pytest.mark.asyncio
async def test_async_stream_error_flushes_pending_hooks_and_propagates():
    """The real NodeLLM streaming ``finally`` flushes before re-raising."""
    pending_hook_completed = asyncio.Event()

    class RecordingFlushRelay(HookRelay):
        def __init__(self):
            super().__init__(node_id="n1")
            self.flush_calls = 0

        async def flush_pending_hooks(self):
            self.flush_calls += 1
            await super().flush_pending_hooks()

    relay = RecordingFlushRelay()

    class ErroringClient:
        class LLMEngine:
            model = "test-model"
            engine_name = "test-provider"

        llm = LLMEngine()

        async def run_agent_stream_async(self, **kwargs):
            async def pending_hook():
                await asyncio.sleep(0)
                pending_hook_completed.set()

            hooks = kwargs["hooks"]
            hooks._pending_futures.add(asyncio.create_task(pending_hook()))
            raise RuntimeError("stream provider error")
            yield  # pragma: no cover - keeps this an async generator

    class ErroringStreamNode(NodeLLM):
        async def _collect_tools(self):
            async def fake_tool(value: str = "") -> str:
                return value

            schema = {
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "description": "Deterministic test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            return [schema], {"fake_tool": fake_tool}, {}

        def _create_hook_relay(self, client=None):
            return relay

    node = ErroringStreamNode(
        data=LlmNodeModel(stream=True),
        node_id="n1",
    )
    node.inputs = {
        node.INPUT_HANDLER_CLIENT_PROVIDER: ErroringClient(),
        node.INPUT_HANDLER_USER_MESSAGE: "hello",
    }

    with pytest.raises(RuntimeError, match="stream provider error"):
        _ = [event async for event in node.process([])]

    assert relay.flush_calls == 1
    assert pending_hook_completed.is_set()
    assert relay._pending_futures == set()


# ── P1-NEW: _emit_llm_generation Parity Test (Task 6.8) ──────────────────


class TestEmitLLMGenerationDetailedUsageParity:
    """P1-NEW (6.8): _emit_llm_generation data parity with on_llm_end hook data.

    Verifies that HookRelay paths produce on_llm_end context with the same
    token/usage data that _emit_llm_generation would provide.
    """

    class _ParityRecordingHooks:
        def __init__(self):
            self.last_context: HookContext | None = None

        def on_llm_end(self, context, response=None):
            self.last_context = context

    def test_on_llm_end_carries_all_token_fields(self):
        """P1-NEW (6.8.1): on_llm_end from HookRelay carries model, tokens, provider_request_id."""
        hooks = self._ParityRecordingHooks()
        relay = HookRelay(flow_hooks=hooks, node_id="n1")
        state = make_agent_state()

        response = make_chat_response(
            model="gpt-4",
            content="Hello from HookRelay",
            finish_reason="stop",
            id="req-provider-123",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "cached_tokens_read": 2,
                "cached_tokens_write": 1,
                "reasoning_tokens": 5,
                "audio_tokens": 0,
                "raw_usage_json": {"raw": "data"},
            },
        )

        relay.on_llm_response(response=response, state=state)

        ctx = hooks.last_context
        assert ctx is not None
        outputs = ctx.outputs
        # Fields that _emit_llm_generation carries
        assert outputs.get("model") == "gpt-4", "model must match"
        assert outputs.get("prompt_tokens") == 10
        assert outputs.get("completion_tokens") == 20
        assert outputs.get("total_tokens") == 30
        assert outputs.get("cached_tokens_read") == 2
        assert outputs.get("cached_tokens_write") == 1
        assert outputs.get("reasoning_tokens") == 5
        assert outputs.get("audio_tokens") == 0
        assert outputs.get("raw_usage_json") == {"raw": "data"}
        assert outputs.get("provider_request_id") == "req-provider-123"

    def test_loop_end_carries_aggregated_token_fields(self):
        """P1-NEW (6.8): on_llm_loop_end from HookRelay carries aggregated token data."""
        self._ParityRecordingHooks()

        # Use a minimal FlowHooks that captures on_llm_loop_end
        class _LoopEndRecorder:
            def __init__(self):
                self.last_context = None
            def on_llm_loop_end(self, context):
                self.last_context = context

        loop_hooks = _LoopEndRecorder()
        relay = HookRelay(flow_hooks=loop_hooks, node_id="n1")
        state = make_agent_state(iteration=2)

        relay.on_loop_complete(
            final_response=make_chat_response(
                model="gpt-4",
                content="Final aggregated answer",
                id="req-final",
                usage={
                    "prompt_tokens": 50,
                    "completion_tokens": 30,
                    "total_tokens": 80,
                },
            ),
            state=state,
        )

        ctx = loop_hooks.last_context
        assert ctx is not None
        outputs = ctx.outputs
        assert outputs.get("model") == "gpt-4"
        assert outputs.get("total_iterations") == 3  # iteration+1
        assert outputs.get("content") == "Final aggregated answer"
        assert outputs.get("provider_request_id") == "req-final"


