# HookRelay — magic-llm Bridge

## Purpose

`HookRelay` bridges magic-llm's **sync** `AgentHooks` protocol to magic-agents' **async** `FlowHooks` protocol. This gives graph-level visibility into LLM calls and tool executions inside magic-llm without modifying magic-llm's contract.

**File**: `magic_agents/hooks/hook_relay.py:1-764`

**Status**: ✅ Implemented.

## Architecture

```
magic-llm AgentHooks (sync)
    ↕ HookRelay implements AgentHooks
    ↕ Translates each event to FlowHooks
magic-agents FlowHooks (async)
    ↕ HookRegistry.invoke() dispatches to all tiers
```

## Event Mapping

| magic-llm AgentHooks Method | FlowHooks Target | File:Line |
|-----------------------------|------------------|-----------|
| `on_iteration_start` | `on_llm_start(context, llm_config=...)` | `hook_relay.py:262-299` |
| `on_llm_response` | `on_llm_end(context)` | `hook_relay.py:301-379` |
| `on_tool_start` | `on_tool_start(context)` | `hook_relay.py:381-425` |
| `on_tool_complete` | `on_tool_end(context)` | `hook_relay.py:427-478` |
| `on_loop_complete` | **`on_llm_loop_end(context)`** | `hook_relay.py:480-526` |
| `on_budget_exceeded` | `on_node_error(context)` | `hook_relay.py:528-544` |

**Key change**: `on_loop_complete` now maps to `on_llm_loop_end`, NOT to `on_llm_end`. Loop completion is signaled by a distinct method. There is no `loop_complete=True` discriminator — this was removed in a clean cut with no deprecation window.

## Observability Levels

HookRelay provides two levels of granularity for LLM observability:

| Level | Event | Fires | Data |
|-------|-------|-------|------|
| **Per-provider-request** | `on_llm_end` | N times for N-iteration loop | Per-iteration response: model, content, finish_reason, provider_request_id, token_usage, iteration (0-indexed) |
| **Aggregate (loop completion)** | `on_llm_loop_end` | Once per loop | Accumulated content from ALL iterations: model, content, content_preview, iteration + total_iterations, provider_request_id, token_usage |

This allows consumers to choose their level of detail:
- **Real-time per-iteration telemetry**: Subscribe to `on_llm_end` events.
- **Final aggregate metrics/audit**: Subscribe to `on_llm_loop_end`.
- **Both**: Subscribe to both — they are independent events.

## Dedup Contract: `on_llm_end` vs `on_llm_loop_end`

There is **no double-fire** between `on_llm_end` and `on_llm_loop_end`. They are distinct events:

| Aspect | `on_llm_end` | `on_llm_loop_end` |
|--------|-------------|-------------------|
| Fires per | Provider request (per LLM call) | Loop completion (once per loop) |
| Content scope | Single iteration's response | ALL iterations aggregated |
 | Contains | `provider_request_id`, `iteration` (0-indexed), per-request tokens | `provider_request_id`, `iteration` (final, 0-indexed), `total_iterations` (count, 1-indexed), accumulated tokens |
| Loop completion? | NO | YES — this IS the loop completion signal |

For an N-iteration tool loop: `on_llm_end` fires N times, `on_llm_loop_end` fires exactly 1 time after the last `on_llm_end`.

## Context Payload Fields

### `on_llm_end` (per-provider-request)

`context.outputs` contains:

| Field | Type | Always Present? | Source |
|-------|------|-----------------|--------|
| `model` | `str` | Yes | `response.model` |
| `content` | `str` | Yes | `response.content` |
| `content_preview` | `str` | Yes | First 200 chars of content |
| `finish_reason` | `Optional[str]` | Yes | `response.finish_reason` |
| `provider_request_id` | `Optional[str]` | Yes | `response.id` — `None` when absent |
| `prompt_tokens` | `Optional[int]` | Yes | `response.usage.prompt_tokens` — `None` when absent |
| `completion_tokens` | `Optional[int]` | Yes | `response.usage.completion_tokens` — `None` when absent |
| `total_tokens` | `Optional[int]` | Yes | `response.usage.total_tokens` — `None` when absent |
| `iteration` | `int` | Yes | `state.iteration` — 0-indexed |

### `on_llm_loop_end` (aggregate loop completion)

`context.outputs` contains:

| Field | Type | Always Present? | Source |
|-------|------|-----------------|--------|
| `model` | `str` | Yes | `final_response.model` |
| `content` | `str` | Yes | Accumulated content across ALL iterations |
| `content_preview` | `str` | Yes | First 200 chars of accumulated content |
| `finish_reason` | `Optional[str]` | Yes | `final_response.finish_reason` |
| `iteration` | `int` | Yes | Final iteration number — **0-indexed** |
| `total_iterations` | `int` | Yes | Total iterations executed — **1-indexed count** |
| `provider_request_id` | `Optional[str]` | Yes | `final_response.id` — `None` when absent |
| `prompt_tokens` | `Optional[int]` | Yes | `final_response.usage.prompt_tokens` — `None` when absent |
| `completion_tokens` | `Optional[int]` | Yes | `final_response.usage.completion_tokens` — `None` when absent |
| `total_tokens` | `Optional[int]` | Yes | `final_response.usage.total_tokens` — `None` when absent |

### `on_llm_start` (before each LLM call)

`context.inputs` contains:

| Field | Type | Always Present? | Source |
|-------|------|-----------------|--------|
| `iteration` | `int` | Yes | 0-indexed iteration number |
| `llm_call_count` | `int` | Yes | Running count of LLM calls |
| `model` | `Optional[str]` | Yes | Best-effort from `_llm_config` — `None` when unavailable |
| `provider` | `Optional[str]` | Yes | Best-effort from `_llm_config` — `None` when unavailable |
| `streaming` | `Optional[bool]` | Yes | Best-effort from `_llm_config` — `None` when unavailable |
| `tools` | `Optional[list]` | Yes | Best-effort from `_llm_config` — `None` when unavailable |
| `tool_choice` | `Optional[str \| dict]` | Yes | Best-effort from `_llm_config` — `None` when unavailable |
| `deduplicate` | `Optional[bool]` | Yes | Best-effort from `_llm_config` — `None` when unavailable |

The `llm_config` extra kwarg on `on_llm_start` carries the full dict of available config fields (not just those in `context.inputs`). It is `None` when no config data is accessible at all.

## Nested Hook Correlation

When nested LLM nodes execute (via `PARENT_HOOKS` ContextVar in `TaskExecutor._execute_nested_llm_node()`), ALL events carry correlation metadata injected by `HookRelay._build_context()` (`hook_relay.py:176-190`) and `_build_tool_context()` (`hook_relay.py:246-257`):

| Field | Location | Type | Description |
|-------|----------|------|-------------|
| `nested_depth` | `context.metadata` | `int` | Runtime nesting depth from `DEPTH` ContextVar. Root = 0, first child = 1, etc. |
| `parent_run_id` | `context.metadata`, `context.parent_run_id` | `Optional[str]` | The `run_id` of the immediate parent loop. `None` for root loops. |
| `nested_request_id` | `context.metadata` | `str` | Unique UUID hex per nested invocation. Consistent across all events of one child. |

This makes ALL events from nested loops distinguishable from parent events without modifying the event payload schema.

## `llm_config` Configuration

HookRelay accepts an `llm_config` dict at construction time (`hook_relay.py:69, 108`). This config is built by `NodeLLM._create_hook_relay()` from available client and node state, and is delivered to `on_llm_start` via the `llm_config` extra kwarg.

The config dict contains best-effort data — all fields are optional and may be `None` when data is unavailable. See [PROTOCOL.md](./PROTOCOL.md#llm_config-parameter) for the full field list.

## Async Bridge

`HookRelay` methods are **sync** (implements `AgentHooks` protocol), but `FlowHooks` methods are **async**. The bridge at `hook_relay.py:582-716` handles the mismatch:

1. **Via HookRegistry** (preferred): `asyncio.create_task(registry.invoke(...))` schedules the async call in the running event loop (`hook_relay.py:635-676`).
2. **Via single FlowHooks**: `getattr` + `iscoroutinefunction` check → `create_task` for async, direct call for sync (`hook_relay.py:613-625`, `678-716`).
3. **No running loop**: Logs warning, hook is skipped (`hook_relay.py:664-676`, `702-716`).

## Flushing Behavior

```python
await hook_relay.flush_pending_hooks()  # hook_relay.py:548-578
```

Call at the end of a NodeLLM iteration to await all pending async tasks. Uses `asyncio.wait()` with configurable timeout (default 5s). Pending tasks beyond timeout are **cancelled** via `task.cancel()` and the `_pending_futures` set is cleared entirely.

**Fix applied**: `_pending_futures` changed from `List` to `Set` (no duplicate tracking). Uses snapshot pattern (`list(self._pending_futures)`) before `asyncio.wait()` to prevent list mutation race. Removed `done_callback` pattern entirely — no orphaned task references remain after flush.

## Tool Data Collection

`HookRelay` accumulates tool calls and results during LLM iterations:

```python
hook_relay.get_collected_tool_data_for_yield(clear=True)  # hook_relay.py:717-759
```

Returns yieldable events (`tool_call`, `tool_result`). `clear=True` (default) empties internal lists after read to prevent double-accumulation.

Each collected entry now carries correlation metadata:

- `provider_request_id`: Maps tool calls/results to the preceding LLM response (cached from `_current_provider_request_id`).
- `iteration`: 0-indexed iteration number from `state.iteration`.

## Risks

1. **Sync-only fallback**: When `asyncio.create_task` cannot be called (no running event loop), hook methods are skipped with a `logger.warning` at `hook_relay.py:664-676`, `702-716`.
2. **`_current_provider_request_id` stale window**: Between loop boundary (`on_loop_complete` clears it to `None`) and the next `on_llm_response`, tool events have `provider_request_id = None`. This is documented behavior — tool events before the first LLM response are inherently uncorrelated.
3. **`on_llm_loop_end` not implemented by existing consumers**: All existing `FlowHooks` subclasses continue working since the protocol provides a default no-op. New consumers should implement `on_llm_loop_end` for loop-level observability.
