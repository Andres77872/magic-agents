# `hook`

## Purpose

Execute a user-defined Python function template at a lifecycle point in graph execution.

## Runtime class

- `NodeHook`
- model: `HookNodeModel`

## Model fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `function_template` | `string` | Optional | `""` | Python code with `def` or `async def` entry point |
| `timeout_override` | `integer` | Optional | `null` | Per-hook timeout in seconds (global default 30s) |
| `hook_type` | `string` | Optional | `"custom"` | Legacy label; does not select a lifecycle event |
| `lifecycle_event` | `string or null` | Optional | `null` | `onStart`, `onDeliver`, `onError`, `onFinish`, or `onCancel`; null retains the connection observer |
| `target_node_id` | `string or null` | Optional | `null` | Node scope; otherwise use the enabled edge attachment |
| `failure_policy` | `string` | Optional | `"preserve"` | `preserve` current request/outcome or `fail` when a control fails |

## Lifecycle controls

Controls execute through `magic_agents/hooks/invocation_control.py` around the real node or registered tool operation. Node scope applies to all callers; edge scope selects the exact caller connection. For a Fetch → LLM connection, controls wrap the actual tool call, not tool registration. There is no provider-specific fallback behavior.

The function receives `(context, chat_log)`. `context.request["content"]` is an ordinary node's JSON input-handle map, or the model's tool arguments. Client objects and callable resources stay on the operational node rather than entering this JSON map. `context.outcome` is null before execution, then success (`status`, `content`), error (`status`, `error`), or cancelled (`status`). A typed error contains `code`, `message`, `retryable`, and `details`. Fetch HTTP errors use `HTTP_ERROR` with `details.http_status` and `details.status_code`; 429 and 5xx are retryable. Text that mentions an HTTP error is still ordinary successful content.

Return `None`/`{"action": "pass"}`, `{"action": "input", "content": ...}` before execution/delivery, `{"action": "outcome", "outcome": ...}`, or `{"action": "redirect", "connection": "edge-id", "content": ...}`. Ordinary node results map output handles to content; tool results contain the native tool return. Cancellation is pass-only and cannot start recovery work.

Wire `handle-child-call` (alias `handles.child_call`) to the child's data input. This is an on-demand capability, not a normal scheduler dependency. `await context.call(edge_id, content)` returns the completed child record; redirect adopts its outcome. The child runs its own controls. Nested calls share a bounded deadline and invocation budget. Original failures and child records remain in the execution trace; only the adopted tool result returns under the model's original tool-call ID.

Controls are awaited. For controlled ordinary nodes, processor output is buffered until finish controls complete; nodes without controls keep their existing streaming behavior. Pass preserves unchanged operational values, including client objects and tool definitions. Hook code has a fresh execution namespace per invocation. The existing observer registry remains responsible for run persistence and accounting.

Real integration coverage lives in `test/test_lifecycle_hook_fallback.py` and `test/test_lifecycle_delivery.py`, with `magic-llm/test/test_tool_executor_invocation_controls.py` checking that caching cannot skip per-call controls. These use the actual node classes and tool loop; the HTTP integration test uses a loopback server.

## Existing connection observer

The following behavior applies when `lifecycle_event` is unset.

## Default input

- `handle-hook-context` — receives a `HookContext` at runtime with `emit` helpers

## Default outputs

- `handle-user-output` — for `emit.user()`
- `handle-debug-output` — for `emit.debug()`
- `handle-feedback-output` — for `emit.feedback()`

## Important behavior

- receives a `HookContext` on its input handle at runtime, injected by the event dispatcher when the edge is traversed
- an enabled edge Hook runs before the original payload reaches the edge's target; the executor waits for its callback to finish before forwarding the payload
- the Hook's returned events are emitted separately and do not replace the original payload
- the template function signature is `def my_hook(hook_context, chat_log)` (sync) or `async def` variant
- timeout enforced via `asyncio.wait_for` (30s default, overridable)
- error isolation: exceptions and timeouts are caught and logged; the original payload is still forwarded and execution continues
- constrained exec namespace: `emit`, `logger`, `datetime`, `UTC`
- can be invoked directly as a graph node or triggered automatically via edge-level `hooks` config

## Current safety

Phase 1 safety (timeout + error isolation). No sandboxing yet — the template runs via `exec()` with a constrained namespace. Restricted globals and subprocess isolation are planned for follow-up.

## Example

```json
{
  "id": "log_hook",
  "type": "hook",
  "data": {
    "function_template": "def on_traversed(ctx, log):\n    return ctx.emit.user(\"edge traversed\")",
    "timeout_override": 10,
    "hook_type": "post"
  }
}
```
