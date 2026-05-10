# Hook Contracts

## Purpose

Defines the data contracts for hook invocations: `HookContext` dataclass, `HookContextFactory` builders, `TypedDict` schemas, and canonical bypass reasons.

## HookContext Dataclass

**File**: `magic_agents/hooks/flow_hooks.py:157-218`

**Status**: ✅ Implemented. Direct construction deprecated in favor of `HookContextFactory`.

```python
@dataclass
class HookContext:
    execution_id: str              # Unique execution trace ID
    sequence_number: int = 0       # Order within execution
    run_id: str = ''               # Run ID from chat_log
    node_id: Optional[str] = None  # Node context (empty for graph-level)
    node_type: Optional[str] = None
    node_class: Optional[str] = None
    inputs: Dict[str, Any]         # Event-specific input data
    outputs: Dict[str, Any]        # Event-specific output data
    timestamp: datetime            # Creation time (UTC)
    start_time: Optional[datetime] # Execution start time
    end_time: Optional[datetime]   # Execution end time
    duration_ms: Optional[float]   # Computed duration
    error: Optional[Exception]     # Error object (for error hooks)
    error_type: Optional[str]
    error_message: Optional[str]
    metadata: Dict[str, Any]       # Additional context
    parent_run_id: Optional[str]   # Nested execution parent
    emit: Optional[Any] = None     # EmitInterface (injected at runtime)
```

`to_dict()` at `flow_hooks.py:220-247` serializes to a JSON-safe dict. Error objects are excluded; only `error_type`/`error_message` are serialized.

## HookContextFactory

**File**: `magic_agents/hooks/context_factory.py:52-419`

**Status**: ✅ Implemented (6 builder methods).

All factory methods:
- Accept `**extra` — unknown kwargs absorbed into `inputs` (forward compatibility).
- Accept `warn_on_missing: bool = False` — logs warnings for missing required fields without blocking.
- Suppress the `DeprecationWarning` from direct `HookContext()` construction (`context_factory.py:39-47`).

| Method | For | Key Parameters |
|--------|-----|----------------|
| `build_graph_context` | `on_graph_start/end/error` | `execution_id`, `metadata` (graph_id, node_count, edge_count) |
| `build_node_context` | `on_node_start/end/error` | `execution_id`, `node_id`, `node_type`, `node_class`, `inputs`, `outputs` |
| `build_edge_context` | Edge dispatch → NodeHook | `execution_id`, `source`, `target`, `content`, handles |
| `build_bypass_context` | `on_node_bypass` | `execution_id`, `node_id`, `reason` (BypassReason), `metadata` |
| `build_llm_context` | `on_llm_start/end` | `execution_id`, `model`, `streaming`, `iteration` |
| `build_tool_context` | `on_tool_start/end` | `execution_id`, `tool_name`, `tool_call_id`, `arguments`, `result` |

## TypedDict Schemas

**File**: `magic_agents/hooks/contracts.py:1-202`

**Status**: ✅ Implemented as `total=False` TypedDicts. NOT enforced at the Protocol level — serve as documentation and optional factory validation.

| Schema | Event | Required Fields | Notes |
|--------|-------|-----------------|-------|
| `GraphStartInputs` | `on_graph_start` | graph_id, node_count, edge_count | |
| `GraphEndInputs` | `on_graph_end` | graph_id, execution_time_ms | |
| `GraphErrorInputs` | `on_graph_error` | graph_id, error_message, error_type | |
| `NodeStartInputs` | `on_node_start` | node_id, node_type, node_class, input_keys | |
| `NodeEndInputs` | `on_node_end` | node_id, duration_ms, output_keys | |
| `NodeBypassInputs` | `on_node_bypass` | node_id, reason (BypassReason) | |
| `LLMStartInputs` | `on_llm_start` | model†, provider†, streaming†, tools†, tool_choice†, deduplicate†, iteration, llm_call_count | † Populated best-effort from HookRelay path when config data is available (may be `None`) |
| `LLMEndInputs` | `on_llm_end` | model, content, content_preview, finish_reason, provider_request_id‡, prompt_tokens‡, completion_tokens‡, total_tokens‡, iteration | ‡ Optional — may be `None` when response data is unavailable. Per-provider-request (N times for N-iteration loop) |
| `LLMLoopEndInputs` | `on_llm_loop_end` | model, content, content_preview, iteration, total_iterations, provider_request_id‡, prompt_tokens‡, completion_tokens‡, total_tokens‡ | **NEW** — aggregated loop completion. Fires once per loop. `iteration` is 0-indexed final, `total_iterations` is 1-indexed count |
| `ToolStartInputs` | `on_tool_start` | tool_name, tool_call_id, arguments | |
| `ToolEndInputs` | `on_tool_end` | tool_name, success | |
| `EdgeHookInputs` | Edge → NodeHook | content, source, target | |

## Canonical Bypass Reasons

**File**: `magic_agents/hooks/contracts.py:32-38`

```python
BypassReason = Literal["upstream_error", "condition", "not_ready"]
```

| Value | Meaning |
|-------|---------|
| `"upstream_error"` | Downstream node skipped because an upstream node failed |
| `"condition"` | Node's condition evaluated to false (static or iteration) |
| `"not_ready"` | Node inputs were not ready for execution (single-node bypass) |

## Deprecated Context Subclasses

**File**: `magic_agents/hooks/flow_hooks.py:253-292`

**Status**: ⚠️ Deprecated. Use `HookContext` with `metadata` fields instead.

| Subclass | Fields | Replacement |
|----------|--------|-------------|
| `NodeLLMHookContext` | model, provider, streaming, token_count, tool_calls, chunk_index | `HookContext.metadata` |
| `NodeMcpHookContext` | session_id, tools_discovered, bundle | `HookContext.metadata` |
| `NodeLoopHookContext` | items_count, item_index, aggregated_result | `HookContext.metadata` |
