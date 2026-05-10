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
| `hook_type` | `string` | Optional | `"custom"` | Lifecycle marker: `"pre"`, `"post"`, `"error"`, `"custom"` |

## Default input

- `handle-hook-context` — receives a `HookContext` at runtime with `emit` helpers

## Default outputs

- `handle-user-output` — for `emit.user()`
- `handle-debug-output` — for `emit.debug()`
- `handle-feedback-output` — for `emit.feedback()`

## Important behavior

- receives a `HookContext` on its input handle at runtime, injected by the event dispatcher when the edge is traversed
- the template function signature is `def my_hook(hook_context, chat_log)` (sync) or `async def` variant
- timeout enforced via `asyncio.wait_for` (30s default, overridable)
- error isolation: exceptions are caught and logged; execution continues
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
    "function_template": "def on_traversed(ctx, log):\n    ctx.emit.user(\"edge traversed\")\n    return None",
    "timeout_override": 10,
    "hook_type": "post"
  }
}
```
