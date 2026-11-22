from __future__ import annotations

import asyncio

import pytest

from magic_agents.hooks.debug_sse import DebugSSEHook
from magic_agents.hooks.context_factory import HookContextFactory
from magic_agents.hooks.flow_hooks import HookContext

# Re-usable helpers

def context(**overrides) -> HookContext:
    inputs = overrides.pop("inputs", {})
    outputs = overrides.pop("outputs", {})
    ctx = HookContextFactory.build_node_context(
        execution_id=overrides.pop("execution_id", "graph-exec"),
        run_id=overrides.pop("run_id", "run-1"),
        node_id=overrides.pop("node_id", None),
        node_type=overrides.pop("node_type", None),
        node_class=overrides.pop("node_class", None),
        inputs=inputs,
        outputs=outputs,
    )
    # Keep this general-purpose fixture's legacy permissiveness while using the
    # validated construction path. Some edge-case tests intentionally pass None.
    ctx.inputs = inputs
    ctx.outputs = outputs
    for name, value in overrides.items():
        setattr(ctx, name, value)
    return ctx


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record(self, event: dict) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_debug_sse_hook_uses_injected_sink_without_api_imports():
    sink = RecordingSink()
    hook = DebugSSEHook(sink=sink, id_chat="chat-1")

    await hook.on_graph_start(context(inputs={"node_count": 2}))

    assert sink.events == [
        {
            "type": "debug",
            "event_type": "graph_start",
            "source_node_path": [],
            "id_chat": "chat-1",
            "content": {"execution_id": "graph-exec", "run_id": "run-1", "node_count": 2, "t_ms": 0.0},
        }
    ]


@pytest.mark.asyncio
async def test_debug_sse_hook_emits_node_error_envelope():
    sink = RecordingSink()
    hook = DebugSSEHook(sink=sink, id_chat="chat-1")

    await hook.on_node_error(
        context(node_id="node-1", node_type="llm", duration_ms=12.5),
        ValueError("bad node"),
    )

    assert sink.events[0]["type"] == "debug"
    assert sink.events[0]["event_type"] == "node_error"
    assert sink.events[0]["content"] == {
        "node_id": "node-1",
        "node_type": "llm",
        "error_type": "ValueError",
        "error_message": "bad node",
        "duration_ms": 12.5,
        "t_ms": 0.0,
    }


@pytest.mark.asyncio
async def test_debug_sse_node_error_carries_the_typed_error_code():
    from magic_agents.hooks.invocation_control import OperationFailure
    sink = RecordingSink()
    hook = DebugSSEHook(sink=sink, id_chat="chat-1")

    await hook.on_node_error(
        context(node_id="inner", node_type="inner", duration_ms=1.0),
        OperationFailure("INNER_FLOW_FAILED", "Sub-flow step 'llm' failed: ChatException: 429",
                         details={"failed_nodes": ["llm"]}),
    )

    content = sink.events[0]["content"]
    assert content["error_type"] == "OperationFailure"
    assert content["error_code"] == "INNER_FLOW_FAILED"
    assert "context" not in content


@pytest.mark.asyncio
async def test_debug_sse_edge_cases():
    """P1-4: Edge cases — empty inputs, None values, zero tokens."""
    sink = RecordingSink()
    hook = DebugSSEHook(sink=sink, id_chat="chat-1")

    # on_llm_start with empty inputs
    await hook.on_llm_start(context(), llm_config=None)
    ev = sink.events[0]
    assert ev["event_type"] == "llm_start"
    assert ev["content"]["model"] == ""
    assert ev["content"]["provider"] == ""
    assert ev["content"]["llm_config"] is None

    # on_llm_end with zero tokens
    await hook.on_llm_end(
        context(outputs={"finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    )
    ev = sink.events[1]
    assert ev["event_type"] == "llm_end"
    assert ev["content"]["tokens"]["prompt_tokens"] == 0

    # on_llm_loop_end with None content_preview
    await hook.on_llm_loop_end(context(outputs={"total_iterations": 1}))
    ev = sink.events[2]
    assert ev["event_type"] == "llm_loop_end"
    assert ev["content"]["content_preview"] == ""

    # on_tool_start with None inputs
    await hook.on_tool_start(context(inputs=None))
    ev = sink.events[3]
    assert ev["event_type"] == "tool_start"
    assert ev["content"]["tool_name"] is None


@pytest.mark.asyncio
async def test_debug_sse_queue_full_non_fatal():
    """P1-4: QueueFull is caught and logged (existing behavior preserved)."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    hook = DebugSSEHook(sink=queue, id_chat="chat-1")

    await hook.on_llm_start(context(node_id="n1", inputs={"model": "gpt-4"}))
    # Queue is now full — next emit should be caught
    with pytest.raises(asyncio.QueueFull):
        queue.put_nowait({"overflow": True})  # Fill the remaining slot

    # This should NOT raise — QueueFull is caught in _emit
    await hook.on_llm_end(context(outputs={"finish_reason": "stop"}))

    # First event still present
    assert queue.get_nowait()["event_type"] == "llm_start"


# ── P1-4: build_debug_sse_hook tests ────────────────────────────────────


def test_build_debug_sse_hook_returns_none_when_disabled():
    """P1-4: build_debug_sse_hook returns None when debug_sse is disabled (default)."""
    from magic_agents.hooks.runtime_config import RuntimeConfig

    config = RuntimeConfig()
    hook = config.build_debug_sse_hook(id_chat="chat-1")
    assert hook is None, "Should return None when disabled by default"


@pytest.mark.asyncio
async def test_build_debug_sse_hook_enabled_with_sink():
    """P1-4: build_debug_sse_hook returns DebugSSEHook when enabled+sink configured."""
    from magic_agents.hooks.runtime_config import RuntimeConfig

    sink = RecordingSink()
    config = RuntimeConfig()
    config.enable_debug_sse(sink)

    hook = config.build_debug_sse_hook(id_chat="chat-1")
    assert hook is not None, "Should return DebugSSEHook when enabled+sink"
    assert hook._id_chat == "chat-1"

    # Verify it fires correctly
    await hook.on_graph_start(context(inputs={"node_count": 2}))
    assert sink.events[0]["event_type"] == "graph_start"


# ── P1-4: 5 new FlowHooks methods ────────────────────────────────────────────


def _make_hook(**hook_kwargs) -> tuple[DebugSSEHook, RecordingSink]:
    """Create a DebugSSEHook with a RecordingSink and return both."""
    sink = RecordingSink()
    hook = DebugSSEHook(sink=sink, id_chat=hook_kwargs.pop("id_chat", "chat-1"), **hook_kwargs)
    return hook, sink


@pytest.mark.asyncio
async def test_on_llm_start_emits_correctly():
    """P1-4: on_llm_start emits 'llm_start' with node_id, model, provider, llm_config."""
    hook, sink = _make_hook()

    ctx = context(
        node_id="node-1",
        inputs={"model": "gpt-4", "provider": "openai"},
    )
    await hook.on_llm_start(
        ctx,
        llm_config={"temperature": 0.7, "model": "gpt-4"},
    )

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["type"] == "debug"
    assert event["event_type"] == "llm_start"
    assert event["content"]["node_id"] == "node-1"
    assert event["content"]["model"] == "gpt-4"
    assert event["content"]["provider"] == "openai"
    assert event["content"]["llm_config"] == {
        "temperature": 0.7,
        "model": "gpt-4",
    }


@pytest.mark.asyncio
async def test_on_llm_end_emits_correctly():
    """P1-4: on_llm_end emits 'llm_end' with finish_reason and tokens dict."""
    hook, sink = _make_hook()

    ctx = context(
        node_id="llm-1",
        outputs={
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "provider_request_id": "req-1",
        },
    )
    await hook.on_llm_end(ctx)

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["event_type"] == "llm_end"
    assert event["content"]["finish_reason"] == "stop"
    assert event["content"]["node_id"] == "llm-1"
    assert event["content"]["tokens"]["prompt_tokens"] == 10
    assert event["content"]["tokens"]["completion_tokens"] == 20
    assert event["content"]["tokens"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_on_llm_end_includes_detail_tokens_in_tokens_dict():
    """P1-4: on_llm_end tokens dict includes detail token fields when present."""
    hook, sink = _make_hook()

    ctx = context(
        node_id="llm-1",
        outputs={
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "cached_tokens_read": 5,
            "cached_tokens_write": 2,
            "reasoning_tokens": 8,
            "audio_tokens": 0,
            "raw_usage_json": {"cache_read": 5},
        },
    )
    await hook.on_llm_end(ctx)

    tokens = sink.events[0]["content"]["tokens"]
    assert tokens["cached_tokens_read"] == 5
    assert tokens["cached_tokens_write"] == 2
    assert tokens["reasoning_tokens"] == 8
    assert tokens["audio_tokens"] == 0
    assert tokens["raw_usage_json"] == {"cache_read": 5}


@pytest.mark.asyncio
async def test_on_llm_loop_end_emits_correctly():
    """P1-4: on_llm_loop_end emits with total_iterations and truncated content_preview."""
    hook, sink = _make_hook()

    ctx = context(
        node_id="llm-1",
        outputs={"total_iterations": 3, "content_preview": "A" * 200},
    )
    await hook.on_llm_loop_end(ctx)

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["event_type"] == "llm_loop_end"
    assert event["content"]["total_iterations"] == 3
    assert event["content"]["node_id"] == "llm-1"
    # content_preview must be truncated to max 100 chars
    assert len(event["content"]["content_preview"]) <= 100
    assert event["content"]["content_preview"] == "A" * 100


@pytest.mark.asyncio
async def test_on_tool_start_emits_correctly():
    """P1-4: on_tool_start emits 'tool_start' with tool_name and tool_call_id."""
    hook, sink = _make_hook()

    ctx = context(
        node_id="tool-1",
        inputs={"tool_name": "weather", "tool_call_id": "call-123"},
    )
    await hook.on_tool_start(ctx)

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["event_type"] == "tool_start"
    assert event["content"]["node_id"] == "tool-1"
    assert event["content"]["tool_name"] == "weather"
    assert event["content"]["tool_call_id"] == "call-123"


@pytest.mark.asyncio
async def test_on_tool_end_emits_correctly():
    """P1-4: on_tool_end emits 'tool_end' with tool_name and success=True."""
    hook, sink = _make_hook()

    ctx = context(
        node_id="tool-1",
        outputs={"tool_name": "weather", "success": True},
    )
    await hook.on_tool_end(ctx)

    assert len(sink.events) == 1
    event = sink.events[0]
    assert event["event_type"] == "tool_end"
    assert event["content"]["node_id"] == "tool-1"
    assert event["content"]["tool_name"] == "weather"
    assert event["content"]["success"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("method,kwargs", [
    ("on_node_start", {}), ("on_node_end", {}),
    ("on_node_error", {"error": ValueError("fixture")}),
    ("on_node_bypass", {"reason": "fixture"}),
    ("on_llm_start", {}), ("on_llm_end", {}), ("on_llm_loop_end", {}),
    ("on_tool_start", {}), ("on_tool_end", {}),
])
async def test_all_node_lifecycles_keep_immutable_sibling_ancestry(method, kwargs):
    from magic_agents.hooks.hook_registry import HookRegistry

    root, sink = _make_hook()
    registry = HookRegistry(global_hooks=[root])
    left = registry.fork_for_child(child_run_id="left-run", parent_node_id="left")
    right = registry.fork_for_child(child_run_id="right-run", parent_node_id="right")
    nested = left.fork_for_child(child_run_id="nested-run", parent_node_id="inner")
    await asyncio.gather(*(
        hooks.invoke(method, context(node_id="shared"), **kwargs)
        for hooks in (registry, left, right, nested)
    ))
    assert sorted(event["source_node_path"] for event in sink.events) == sorted([
        ["shared"], ["left", "shared"], ["right", "shared"], ["left", "inner", "shared"],
    ])
    assert all(event["content"]["node_id"] == "shared" for event in sink.events)
    assert all(event["id_chat"] == "chat-1" for event in sink.events)
    sink.events[0]["source_node_path"].append("sink-mutation")
    await registry.invoke(method, context(node_id="fresh"), **kwargs)
    assert sink.events[-1]["source_node_path"] == ["fresh"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,kwargs", [
    ("on_graph_start", {}), ("on_graph_end", {}),
    ("on_graph_error", {"error": ValueError("fixture")}),
])
async def test_graph_lifecycle_paths_name_the_graph_scope_only(method, kwargs):
    root, sink = _make_hook()
    child = root.fork_for_child(child_run_id="child-run", parent_node_id="outer")
    nested = child.fork_for_child(child_run_id="nested-run", parent_node_id="inner")
    for hook in (root, child, nested):
        await getattr(hook, method)(context(), **kwargs)
    assert [event["source_node_path"] for event in sink.events] == [[], ["outer"], ["outer", "inner"]]
    assert all("node_id" not in event["content"] for event in sink.events)
