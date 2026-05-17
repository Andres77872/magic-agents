# Migration Guide

## Purpose

Documents deprecations and recommended migration paths for consumers of the hooks system.

---

## 1. Direct `HookContext()` → `HookContextFactory`

**Status**: ⚠️ Deprecated since Phase 4.

**Evidence**: `flow_hooks.py:167-183` — `__post_init__` emits `DeprecationWarning` when `type(self) is HookContext`.

### Before (deprecated)

```python
from magic_agents.hooks.flow_hooks import HookContext

ctx = HookContext(
    execution_id="abc",
    node_id="node-1",
    node_type="LLM",
)
```

### After

```python
from magic_agents.hooks.context_factory import HookContextFactory

ctx = HookContextFactory.build_node_context(
    execution_id="abc",
    node_id="node-1",
    node_type="LLM",
    node_class="NodeLLM",
)
```

The factory:
- Suppresses the deprecation warning (`context_factory.py:39-47`)
- Provides per-event-type validation (`context_factory.py:52-58`)
- Absorbs unknown kwargs into `inputs` for forward compatibility

---

## 2. `hook_type` → `hook_node_id` in EdgeHookConfig

**Status**: ⚠️ Deprecated.

**Evidence**: `EdgeNodeModel.py:40-43` — `hook_type` field marked `deprecated=True`. Warning emitted at `EdgeNodeModel.py:54-62`.

### Before (deprecated)

```python
EdgeHookConfig(hook_type="my_custom_type", enabled=True)
```

### After

```python
EdgeHookConfig(hook_node_id="my-hook-node", enabled=True)
```

Edge hook dispatch is determined **solely by `hook_node_id`**. The `hook_type` field is ignored at dispatch time.

---

## 3. `NodeLLMHookContext` → `HookContext.metadata`

**Status**: ⚠️ Deprecated.

**Evidence**: `flow_hooks.py:232-246` — class docstring says "[DEPRECATED]".

### Before (deprecated)

```python
ctx = NodeLLMHookContext(
    execution_id="abc",
    model="gpt-4",
    provider="openai",
    streaming=True,
)
```

### After

```python
ctx = HookContextFactory.build_llm_context(
    execution_id="abc",
    model="gpt-4",
    streaming=True,
)
# Additional fields go into metadata or are factory parameters
```

LLM-specific fields (`model`, `streaming`, `iteration`) are now first-class factory parameters (`context_factory.py:298-303`). Node-specific fields go into `HookContext.metadata`.

---

## 4. `NodeMcpHookContext` → `HookContext.metadata`

**Status**: ⚠️ Deprecated.

**Evidence**: `flow_hooks.py:248-258` — class docstring says "[DEPRECATED]".

### Before

```python
ctx = NodeMcpHookContext(
    execution_id="abc",
    session_id="sess-1",
    tools_discovered=5,
)
```

### After

```python
ctx = HookContextFactory.build_node_context(
    execution_id="abc",
    node_id="mcp-node",
    node_type="MCP",
    node_class="NodeMcp",
    metadata={
        "session_id": "sess-1",
        "tools_discovered": 5,
    },
)
```

---

## 5. `NodeLoopHookContext` → `HookContext.metadata`

**Status**: ⚠️ Deprecated.

**Evidence**: `flow_hooks.py:260-271` — class docstring says "[DEPRECATED]".

### Before

```python
ctx = NodeLoopHookContext(
    execution_id="abc",
    items_count=10,
    item_index=3,
)
```

### After

```python
ctx = HookContextFactory.build_node_context(
    execution_id="abc",
    node_id="loop-node",
    node_type="LOOP",
    node_class="NodeLoop",
    metadata={
        "items_count": 10,
        "item_index": 3,
    },
)
```

---

## Summary Table

| Deprecated | Replacement | Migration Difficulty |
|------------|-------------|---------------------|
| `HookContext()` constructor | `HookContextFactory.build_*_context()` | Easy — find/replace pattern |
| `EdgeHookConfig.hook_type` | `EdgeHookConfig.hook_node_id` | Easy — field rename |
| `NodeLLMHookContext` subclass | `HookContextFactory.build_llm_context()` | Medium — restructure to factory |
| `NodeMcpHookContext` subclass | `HookContext` + `metadata` | Medium — move fields to metadata |
| `NodeLoopHookContext` subclass | `HookContext` + `metadata` | Medium — move fields to metadata |
