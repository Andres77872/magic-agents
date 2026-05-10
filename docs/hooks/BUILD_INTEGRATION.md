# Build Integration — JSON/Declarative Hook Support

## Purpose

Documents how `build()` and JSON agent definitions interact with the hooks system — what works, what is stripped, and supported declarative patterns.

## Graph-Level Dict Hooks Are Stripped

**File**: `magic_agents/agt_flow.py:663-672`

```python
if 'hooks' in agt_data and isinstance(agt_data['hooks'], dict):
    logger.warning(
        "Graph-level hooks cannot be set from JSON. "
        "Use type: 'hook' nodes or EdgeHookConfig instead. "
        "The 'hooks' key in agent data will be ignored."
    )
    del agt_data['hooks']
```

**Status**: ✅ Implemented.

When building from JSON:
- If `"hooks"` key is a `dict`, it is **deleted** with a `WARNING` log.
- Graph-level hooks require Python `FlowHooks` instances (not JSON-serializable).
- Only **programmatic** `AgentFlowModel(hooks=my_hook)` can set graph-level hooks.

## Supported Declarative Hook Patterns

### 1. Hook Nodes (`type: "hook"`)

```json
{
  "id": "hook-1",
  "type": "hook",
  "function_template": "def my_hook(ctx, log): emit.user('processed')"
}
```

Works in JSON because it's a regular node with a string `function_template` field. No serialization of Python objects needed.

### 2. EdgeHookConfig (`edges[].hooks`)

```json
{
  "source": "node-1",
  "target": "node-2",
  "hooks": {
    "hook_node_id": "hook-1",
    "enabled": true
  }
}
```

Works in JSON as a declarative dict on each edge. `EdgeHookConfig` is a Pydantic model that can be deserialized from JSON.

## Programmatic-Only Hook Registration

These patterns cannot be expressed in JSON:

| Pattern | Reason |
|---------|--------|
| `RuntimeConfig.register_global_hook(hook)` | Needs Python FlowHooks instance |
| `RuntimeConfig(graph_hooks=[hook])` | Needs Python FlowHooks instance |
| `AgentFlowModel(hooks=hook_instance)` | Needs Python FlowHooks instance |
| `registry.register_node(node_id, hook)` | Needs Python FlowHooks instance |
| `registry.register_graph(hook)` | Needs Python FlowHooks instance |

## NodeInner Hook Propagation

**Status**: ✅ Implemented (`NodeInner.py:143-171`).

`NodeInner` propagates hooks to child graphs by creating a child `HookRegistry` with identity copied from the parent. When building from JSON, this works automatically if the inner graph's `AgentFlowModel.hooks` is set programmatically.
