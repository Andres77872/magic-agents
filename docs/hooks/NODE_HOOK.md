# NodeHook — Python Function Template Node

## Purpose

`NodeHook` is a node type (`type: "hook"`) that executes user-defined Python function templates via `exec()` at lifecycle points. It receives a `HookContext` via its input handle and can emit messages through `emit.user()`, `emit.debug()`, and `emit.feedback()`.

**File**: `magic_agents/node_system/NodeHook.py:1-303`

**Status**: ✅ Implemented (Phase 1 safety only).

## Model

**File**: `magic_agents/models/factory/Nodes/HookNodeModel.py:1-51`

```python
class HookNodeModel(BaseNodeModel):
    function_template: str    # Python code string (e.g., "def my_hook(ctx, log): ...")
    timeout_override: Optional[int]  # Per-hook timeout (default: 30s global)
    hook_type: str = "custom"  # 'pre', 'post', 'error', 'custom'
```

## Function Template Contract

The template must contain a `def` or `async def` function declaration (`NodeHook.py:270-294`):

```python
def my_hook(ctx: HookContext, log: ModelAgentRunLog):
    emit.user("hook fired!")
```

Or async variant:
```python
async def my_hook(ctx: HookContext, log: ModelAgentRunLog):
    emit.user("async hook!")
```

**Signature**: `func(hook_context: HookContext, chat_log)` — the second parameter receives the agent run log.

## Execution Flow

1. `NodeHook.process()` at `NodeHook.py:101-192` — entry point
2. Retrieves `HookContext` from `INPUT_HANDLE_HOOK_CONTEXT` input handle (`NodeHook.py:116-133`)
3. Creates `EmitInterface` and injects into `hook_context.emit` (`NodeHook.py:136-137`)
4. Compiles function template via `exec()` at `NodeHook.py:214-268`
5. Executes with `asyncio.wait_for(func(...), timeout=self._timeout_seconds)` at `NodeHook.py:149-154`
6. Yields result dict or debug error event

## Emit API

**File**: `magic_agents/hooks/emit_context.py:1-170`

| Method | Status | Behavior |
|--------|--------|----------|
| `emit.user(message, extras=None)` — `emit_context.py:48-95` | ✅ Concrete | Creates `ChatCompletionModel`, wraps via `node.prep()`, yields `SYSTEM_EVENT_STREAMING` |
| `emit.debug(event_type, payload)` — `emit_context.py:97-134` | ❌ Structure-only | Returns dict with `SYSTEM_EVENT_DEBUG` type. Integration with `EmitterRegistry` is TBD |
| `emit.feedback(event_data)` — `emit_context.py:136-170` | ❌ Placeholder | Returns dict. Extras propagation mechanism TBD |

Output routing via handle names (`HookNodeModel.py:18-21`):
- `emit.user()` → `handle-user-output`
- `emit.debug()` → `handle-debug-output`
- `emit.feedback()` → `handle-feedback-output`

Handle names can be overridden via the `handles` dict in JSON (`NodeHook.py:87-99`):
```json
{
  "type": "hook",
  "handles": {
    "hook_context": "custom-input",
    "user_output": "custom-output"
  }
}
```

## Timeout

- **Global default**: 30 seconds (`NodeHook.py:58`)
- **Per-hook override**: `timeout_override` field on `HookNodeModel` (`NodeHook.py:81-83`)
- Timeout produces `HookTimeout` error event via `yield_debug_error()` (`NodeHook.py:160-176`)

## Safety — Phase 1 Only

**⚠️ WARNING**: NodeHook uses `exec()` with a **constrained but unsandboxed** namespace (`NodeHook.py:238-244`):

```python
namespace = {
    'emit': None,       # Injected at runtime
    'logger': logger,
    'datetime': datetime,
    'UTC': UTC,
}
```

| Phase | Safety | Status |
|-------|--------|--------|
| 1 | Timeout + error isolation + constrained exec namespace | ✅ Implemented |
| 2 | Restricted globals | ❌ Not implemented |
| 3 | Full sandboxing (subprocess, RestrictedPython) | ❌ Not implemented |

**Risk**: Arbitrary Python code execution in hook templates. Only use with trusted templates.

## Handle Names

Default input/output handles (`HookNodeModel.py:17-21`):

```
INPUT:  handle-hook-context
OUTPUT: handle-user-output
OUTPUT: handle-debug-output
OUTPUT: handle-feedback-output
```
