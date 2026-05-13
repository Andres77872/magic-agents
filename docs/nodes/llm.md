# `llm`

## Purpose

Run LLM generation in batch or streaming mode, optionally with tools or JSON parsing.

## Runtime class

- `NodeLLM`
- model: `LlmNodeModel`

## Default inputs

- `handle-client-provider`
- `handle-chat`
- `handle-system-context`
- `handle_user_message`
- dynamic tool handles with prefix `handle-tool-`

## Runtime-overridable inputs

The runtime also supports input handles that can override selected generation settings at execution time:

- `handle-llm-temperature`
- `handle-llm-top_p`
- `handle-llm-max_tokens`
- `handle-llm-stream`
- `handle-llm-iterate`
- `handle-llm-json_output`

## Default outputs

- `handle_streaming_content` — streaming chunks during generation
- `handle_generated_content` — complete response after generation
- `handle-tool-calls` — tool call requests (when tools are present)

## Canonical output

`handle_generated_content` is the canonical routed output for downstream nodes. Use this handle for edge connections.

## Important behavior

- supports streaming and non-streaming execution
- supports `json_output` with code-block extraction before JSON parsing
- supports `iterate: true` so the node re-runs on each loop iteration
- collects tools from `fetch`, `python_exec`, `mcp`, and task-subagent bundles
- warns for engines known to have weak/no tool support

## Gotchas

- if debug is enabled, consumers must handle non-content debug events too
- `handle-tool-calls` is only emitted when tools are present

## Example

```json
{
  "id": "answer",
  "type": "llm",
  "data": {
    "stream": true,
    "temperature": 0.2,
    "max_tokens": 512
  }
}
```

## Hook Lifecycle

NodeLLM is the primary driver of LLM and tool hook events in the hooks system. It bridges magic-llm's agent loop to magic-agents' `FlowHooks` protocol through two paths: `HookRelay` (for callable tools) and direct `self._hooks.invoke()` (for schema-only tools and no-tools paths).

### Hook Events Fired by NodeLLM

| Hook Event | Trigger | Source | Path |
|------------|---------|--------|------|
| `on_llm_start` | LLM generation begins | HookRelay translation or direct `self._hooks.invoke()` | HookRelay + Direct |
| `on_llm_end` | LLM generation completes | HookRelay translation or direct `self._hooks.invoke()` | HookRelay + Direct |
| `on_llm_loop_end` | Agent loop completes (or single call finishes) | HookRelay translation (callable tools) or direct invoke (schema/no tools) | HookRelay + Direct |
| `on_tool_start` | Tool execution begins | HookRelay translation | HookRelay only |
| `on_tool_end` | Tool execution completes | HookRelay translation | HookRelay only |

### Path Distinction

1. **HookRelay path** (callable tools — `tool_functions` present): `NodeLLM` creates a `HookRelay` instance at `NodeLLM.py:292` and passes it as `hooks=` to magic-llm's `run_agent_async()` / `run_agent_stream_async()`. HookRelay bridges magic-llm's sync `AgentHooks` protocol to magic-agents' async `FlowHooks`. All 5 LLM/tool events fire through HookRelay translation. Tool events carry `node_type="TOOL"` via `_build_tool_context()`.

2. **Direct path** (schema-only tools or no tools): `NodeLLM` calls `self._hooks.invoke()` directly for `on_llm_start`, `on_llm_end`, and `on_llm_loop_end`. Tool events (`on_tool_start`, `on_tool_end`) only fire from the HookRelay path (they require an agent loop). The direct path sets `total_iterations=1` for `on_llm_loop_end` since it's a single-call path (no multi-iteration agent loop).

### flush_pending_hooks() Behavior

- Called at the **end of every execution** (both streaming and non-streaming)
- For **streaming paths**: called in a `finally` block (`NodeLLM.py:648`, `NodeLLM.py:683`) — ensures hooks are flushed even on error exit
- For **non-streaming paths**: called immediately after the agent loop or LLM call completes (`NodeLLM.py:450`, `NodeLLM.py:461`)
- Drains all pending async futures in `HookRelay._pending_futures` — prevents `Task was destroyed but it is pending!` warnings

### Sync Fallback Path Limitation

When `run_agent_async()` or `run_agent_stream_async()` are NOT available in the installed magic-llm version (very old versions), `NodeLLM` falls back to synchronous execution via `asyncio.to_thread()`:

- **Hook events are NOT delivered on sync fallback paths.** The `HookRelay` methods called from the thread pool thread cannot use `asyncio.create_task()`, so all hook events are silently dropped.
- A `WARNING`-level log message is emitted: *"Hook events will NOT be delivered via this path. Upgrade magic-llm for native async hook support."*
- **Recommendation**: Upgrade magic-llm to a version that provides `run_agent_async()` / `run_agent_stream_async()` to enable full hook delivery.

### Debug Emission Consolidation

`NodeLLM` previously emitted parallel debug events via `_emit_llm_generation()` (yield-based) alongside hook events. This has been consolidated:

- **HookRelay paths** (callable tools): `_emit_llm_generation` has been removed — `on_llm_end` carries the same data (`model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `provider_request_id`)
- **Direct paths** (schema-only/no tools): `_emit_llm_generation` retained as fallback with TODO — detail token fields (`cached_tokens_read`, `cached_tokens_write`, `reasoning_tokens`, `audio_tokens`) are not yet in the direct-path `on_llm_end` context
