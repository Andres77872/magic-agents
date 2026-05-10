# FlowHooks Protocol Reference

## Purpose

Reference for the 12 `FlowHooks` lifecycle methods — signatures, invocation semantics, and payload.

## Protocol Definition

**File**: `magic_agents/hooks/flow_hooks.py:22-154`

```python
@runtime_checkable
class FlowHooks(Protocol):
```

## Methods

### Graph Lifecycle

| Method | Signature | Invoked | Channels |
|--------|-----------|---------|----------|
| `on_graph_start` | `(self, context: HookContext) -> None` | After validation, before task creation (`reactive_executor.py:391`) | Debug, observer, and hook channels participate; see implementation-specific ordering below for node events. |
| `on_graph_end` | `(self, context: HookContext) -> None` | After all node tasks finish successfully | Hook, observer, and debug channels participate. |
| `on_graph_error` | `(self, context: HookContext, error: Exception) -> None` | When a blocking error occurs | Hook, observer, and debug channels participate. |

`on_graph_end` is NOT invoked for failed executions (`flow_hooks.py:63`).

### Node Lifecycle

| Method | Signature | Invoked |
|--------|-----------|---------|
| `on_node_start` | `(self, context: HookContext) -> None` | Before `_start_debug_tracking()` and `process()` (`Node.py:237`) |
| `on_node_end` | `(self, context: HookContext) -> None` | After `process()` yields outputs and `_end_debug_tracking()` |
| `on_node_error` | `(self, context: HookContext, error: Exception) -> None` | In exception catch block, BEFORE `yield_debug_error()` |
| `on_node_bypass` | `(self, context: HookContext, reason: str) -> None` | When bypass signal propagates (`reactive_executor.py:586-628`) |

Implementation ordering for node lifecycle events in `Node.py`:

- Start events: hook invocation → observer callback → debug tracking.
- End events: observer callback → debug tracking → hook invocation.

Channel ordering for BYPASS: Phase 1 (observer) → Phase 4 (hook), no debug yield (`flow_hooks.py:37-38`).

Reason for `on_node_bypass`: one of `"upstream_error"`, `"condition"`, `"not_ready"` (see contracts.py:32-38).

### LLM Lifecycle

| Method | Signature | Invoked |
|--------|-----------|---------|
| `on_llm_start` | `(self, context: HookContext, llm_config: Optional[Dict[str, Any]] = None) -> None` | Before each LLM call in the agent loop (`hook_relay.py:195-233`). `llm_config` IS populated with real config data when available (model, provider, streaming, tools, tool_choice, deduplicate). May be `None` when config is unavailable. |
| `on_llm_end` | `(self, context: HookContext) -> None` | After each LLM response, per provider request (`hook_relay.py:234-268`). Carries per-iteration data only. |
| `on_llm_loop_end` | `(self, context: HookContext) -> None` | **NEW** — After loop completes, aggregated once per loop (`hook_relay.py:354-400`). Carries accumulated content from ALL iterations. NOT the same as `on_llm_end`. |

`on_llm_end` now fires **once per provider request** (N times for an N-iteration loop). Loop completion is signaled exclusively by the separate `on_llm_loop_end` method. There is **no** double-fire: `on_llm_end` and `on_llm_loop_end` are distinct events with distinct semantics.

### Tool Lifecycle (relayed from magic-llm)

| Method | Signature | Invoked |
|--------|-----------|---------|
| `on_tool_start` | `(self, context: HookContext) -> None` | Before tool execution (`hook_relay.py:270-311`) |
| `on_tool_end` | `(self, context: HookContext) -> None` | After tool execution (`hook_relay.py:313-352`) |

`context.inputs` for `on_tool_start`: `tool_name`, `tool_call_id`, `arguments`, `provider_request_id`, `iteration`.
`context.outputs` for `on_tool_end`: `tool_name`, `success`, `execution_time_ms`, `provider_request_id`, `iteration`.
`context.error_message` for `on_tool_end`: error string if tool failed.

## Nested Hook Correlation Metadata

When nested LLM nodes execute (via `TaskExecutor._execute_nested_llm_node()`), ALL events carry correlation metadata in `context.metadata`:

| Field | Type | Description |
|-------|------|-------------|
| `nested_depth` | `int` | Nesting depth: 0 = root loop, 1 = first child, 2 = grandchild, etc. Read from the `DEPTH` ContextVar at runtime. |
| `parent_run_id` | `Optional[str]` | The `run_id` of the immediate parent loop. `None` for root loops. |
| `nested_request_id` | `str` | Unique UUID hex per nested invocation, consistent across all events of one child. |

These are injected at `_build_context()` level in `hook_relay.py:176-190`, so ALL hook events (LLM and tool) carry them.

## `llm_config` Parameter

The `llm_config` parameter on `on_llm_start(context, llm_config=...)` is now **populated with real config data** when available from the HookRelay path. It is NOT deprecated and NOT always `None`.

**Populated fields** (best-effort, may be `None` when unavailable):

| Key | Type | Source |
|-----|------|--------|
| `model` | `Optional[str]` | `client.llm.model` |
| `provider` | `Optional[str]` | `client.llm.engine_name` |
| `streaming` | `Optional[bool]` | `NodeLLM.stream` |
| `json_output` | `Optional[bool]` | `NodeLLM.json_output` |
| `tool_choice` | `Optional[str \| dict]` | Agent loop `tool_choice` config |
| `deduplicate` | `Optional[bool]` | Agent loop `deduplicate` config |
| `tools_count` | `Optional[int]` | `len(self._tools)` |

The dict is `None` when no config data is accessible at all (distinguishes "no data" from "all fields are None").

## Unimplemented NodeInner-Specific Hooks

**Status**: ❌ Not implemented.

These 5 hooks are designed in `.dev/rdd/changes/agent-hooks-architecture/` but **not implemented in code**:

| Hook | Purpose |
|------|---------|
| `on_subgraph_start` | Called when NodeInner starts its child graph |
| `on_subgraph_end` | Called when NodeInner's child graph ends |
| `on_child_event` | Called for each event emitted by the child graph |
| `on_child_chunk` | Called for each streaming chunk from the child graph |
| `on_subgraph_error` | Called when NodeInner's child graph errors |

Inner/outer graph differentiation currently relies on `parent_run_id` filtering.
