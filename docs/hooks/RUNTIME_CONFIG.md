# RuntimeConfig & HookRegistry

## Purpose

Reference for programmatic hook registration at global, graph, and node levels — including `RuntimeConfig` (application-scoped) and `HookRegistry` (execution-scoped).

## RuntimeConfig

**File**: `magic_agents/hooks/runtime_config.py:1-295`

**Status**: ✅ Implemented.

### Class-Level (Application-Scoped) Hooks

```python
RuntimeConfig.register_global_hook(my_hook)     # runtime_config.py:50-72
RuntimeConfig.clear_global_hooks()              # runtime_config.py:74-87
RuntimeConfig.get_global_hooks()                # runtime_config.py:89-98
RuntimeConfig.has_global_hooks()                # runtime_config.py:280-287
```

Global hooks persist as class-level mutable state (`_global_hooks` at `runtime_config.py:48`). They fire for **all** executions regardless of graph instance.

**⚠️ Test pollution risk**: `_global_hooks` is class-level mutable state. Tests that exercise global hooks must call `RuntimeConfig.clear_global_hooks()` in setup/teardown paths to prevent class-level state pollution.

### Instance-Level (Config-Scoped) Hooks

```python
config = RuntimeConfig(graph_hooks=[my_hook])   # runtime_config.py:105-122
config.register_graph_hook(hook)                # runtime_config.py:124-137
```

Instance hooks fire only when this specific `RuntimeConfig` is used.

### Factory Method

```python
registry = config.create_registry()             # runtime_config.py:143-172
```

Combines global hooks + instance graph hooks. Node-level hooks must be registered directly on the registry after creation.

## HookRegistry

**File**: `magic_agents/hooks/hook_registry.py:1-286`

**Status**: ✅ Implemented.

### 3-Tier Registration

| Method | Tier | Scope |
|--------|------|-------|
| `registry.register_global(hook)` — `hook_registry.py:49-59` | 1 (Global) | ALL executions |
| `registry.register_graph(hook)` — `hook_registry.py:61-71` | 2 (Graph) | All nodes in this execution |
| `registry.register_node(node_id, hook)` — `hook_registry.py:73-89` | 3 (Node) | Only the specified node |

All registration methods emit a warning if the object does not implement `FlowHooks` protocol.

### Execution Identity

The executor sets identity fields before invoking hooks (`reactive_executor.py:439-441`):

```python
if hooks is not None:
    hooks.execution_id = _execution_id
    hooks.run_id = run_id or ''
```

### Invocation

```python
await hooks.invoke("on_node_start", context)
await hooks.invoke("on_node_error", context, error=exception)
await hooks.invoke("on_node_bypass", context, reason="upstream_error")
```

- Execution order: Node → Graph → Global (`hook_registry.py:119-161`)
- Parallel execution via `asyncio.gather(*tasks, return_exceptions=True)` (`hook_registry.py:163-195`)
- Sync hooks wrapped in `asyncio.to_thread()` (`hook_registry.py:179-191`)
- Errors logged with `logger.warning()`, never propagated (`hook_registry.py:197-255`)

### Empty Check

```python
registry.is_empty()  # hook_registry.py:257-270
```

Empty registry = no behavior change. Used for lazy `HookContext` construction optimization.

### Query Counters

```python
registry.get_global_hooks_count()  # hook_registry.py:272-274
registry.get_graph_hooks_count()   # hook_registry.py:276-278
registry.get_node_hooks_count()    # hook_registry.py:280-282
registry.get_total_hooks_count()   # hook_registry.py:284-286
```

## Graph-Level Hooks on AgentFlowModel

**File**: `magic_agents/models/factory/AgentFlowModel.py:155-159`

```python
class AgentFlowModel(BaseModel):
    hooks: Optional[FlowHooks] = Field(default=None)
```

Merged with `RuntimeConfig` at execution time in `agt_flow.py:396-405`:

```python
_registry = None
if hooks is not None and not hooks.is_empty():
    _registry = hooks.create_registry()
    if graph.hooks is not None:
        _registry.register_graph(graph.hooks)
elif graph.hooks is not None:
    _registry = HookRegistry()
    _registry.register_graph(graph.hooks)
```

## NodeInner Hook Propagation

**File**: `magic_agents/node_system/NodeInner.py:143-171`

`NodeInner` creates a child `HookRegistry`, copies `execution_id`/`run_id`, registers `inner_graph.hooks`, and passes it to `execute_graph_reactive()`.
