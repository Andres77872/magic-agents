# EdgeHookConfig — Edge-Level Hook Dispatch

## Purpose

`EdgeHookConfig` attaches a `NodeHook` node to an edge. When output propagates through that edge, the dispatcher builds an edge `HookContext` and delivers it to the configured hook node.

**Files**:
- Model: `magic_agents/models/factory/EdgeNodeModel.py:24-52`
- Dispatch: `magic_agents/execution/event_dispatcher.py:245-275`

**Status**: ✅ Implemented.

## EdgeHookConfig Model

```python
class EdgeHookConfig(BaseModel):
    hook_node_id: Optional[str]   # NodeHook node ID to invoke (REQUIRED)
    hook_type: str = "on_edge_traversed"  # [DEPRECATED]
    timeout_override: Optional[int]  # Timeout override in seconds
    enabled: bool = True           # Whether this hook is active
```

### JSON Usage

```json
{
  "source": "node-1",
  "target": "node-2",
  "hooks": {
    "hook_node_id": "my-hook-node",
    "enabled": true
  }
}
```

## Dispatch Flow

At `event_dispatcher.py:251-275`:

1. Check: `edge.hooks and edge.hooks.enabled and edge.hooks.hook_node_id`
2. Look up the hook node by `hook_node_id` in `self.nodes`
3. Verify the node has `INPUT_HANDLE_HOOK_CONTEXT` attribute
4. Build `HookContext` via `HookContextFactory.build_edge_context()` with routing data:
   - `execution_id`, `run_id`, `source`, `target`
   - `source_handle`, `target_handle`
   - `content` (the payload traversing the edge)
5. Set `hook_node.inputs[INPUT_HANDLE_HOOK_CONTEXT] = _hook_ctx`
6. Call `self.dispatch_input(hook_node_id, handle, ctx)` to trigger `NodeHook.process()`

## Edge Hook Context Payload

`context.inputs` contains (`contracts.py:130-140`):

| Field | Type | Description |
|-------|------|-------------|
| `content` | Any | The payload traversing the edge |
| `source` | str | Source node ID |
| `target` | str | Target node ID |
| `source_handle` | Optional[str] | Source handle name |
| `target_handle` | Optional[str] | Target handle name |

## `hook_type` Deprecation

**Status**: ⚠️ Deprecated.

`hook_type` at `EdgeNodeModel.py:40-43` is marked with `deprecated=True`. The field still exists but emits a warning on non-default value (`EdgeNodeModel.py:54-62`). Edge hook dispatch is determined **solely by `hook_node_id`**.

```python
if self.hook_type is not None and self.hook_type != "on_edge_traversed":
    logger.warning("EdgeHookConfig.hook_type is deprecated...")
```

Migration: use `hook_node_id` only.
