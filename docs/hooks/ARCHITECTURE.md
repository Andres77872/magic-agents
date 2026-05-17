# Hooks Architecture

## Purpose

Describes the layered architecture of the hooks system: protocol layer, registry layer, runtime layer, and the node/edge integration layer.

## Layered Architecture

```
User Code (FlowHooks impl)
        |
RuntimeConfig           → class-level global hooks, instance-level graph hooks
        |
HookRegistry            → 3-tier dispatch: Node → Graph → Global, async parallel
        |
reactive_executor.py    → drives on_graph_*, on_node_*, on_node_bypass
Node.__call__()         → drives on_node_start / end / error
NodeLLM                 → on_llm_*, on_tool_* via HookRelay bridge or direct self._hooks.invoke()
event_dispatcher.py     → edge-level: EdgeHookConfig → NodeHook dispatch
NodeHook.process()      → exec() template execution
```

### Layer 1: Protocol (`magic_agents/hooks/flow_hooks.py:23-165`)

Defines `FlowHooks` — a `@runtime_checkable` Protocol with **12 async methods** (`flow_hooks.py:23-165`). All methods are optional. Hooks MUST NOT modify execution state.

The 12 methods are:

| # | Method | Source | File:Line |
|---|--------|--------|-----------|
| 1 | `on_graph_start` | executor | `flow_hooks.py:49` |
| 2 | `on_graph_end` | executor | `flow_hooks.py:56` |
| 3 | `on_graph_error` | executor | `flow_hooks.py:63` |
| 4 | `on_node_start` | executor | `flow_hooks.py:73` |
| 5 | `on_node_end` | executor | `flow_hooks.py:80` |
| 6 | `on_node_error` | executor | `flow_hooks.py:87` |
| 7 | `on_node_bypass` | executor | `flow_hooks.py:94` |
| 8 | `on_llm_start` | HookRelay / NodeLLM direct | `flow_hooks.py:104` |
| 9 | `on_llm_end` | HookRelay / NodeLLM direct | `flow_hooks.py:114` |
| 10 | `on_llm_loop_end` | HookRelay / NodeLLM direct | `flow_hooks.py:121` |
| 11 | `on_tool_start` | HookRelay / NodeLLM direct | `flow_hooks.py:141` |
| 12 | `on_tool_end` | HookRelay / NodeLLM direct | `flow_hooks.py:148` |

**Note**: `on_llm_loop_end` (method #10) was added in Phase 0 and is distinct from `on_llm_end`. It fires once per agent loop with aggregated content, while `on_llm_end` fires per iteration.

### Layer 2: Registry (`magic_agents/hooks/hook_registry.py:19-286`)

`HookRegistry` is an execution-scoped registry (`hook_registry.py:19-286`). Dies with execution. No module-level global state.

Provides 3-tier registration: Node → Graph → Global, parallel dispatch via `asyncio.gather()`, execution identity setters, and empty-check optimization.

### Layer 3: Runtime Config (`magic_agents/hooks/runtime_config.py:22-294`)

`RuntimeConfig` provides application-scoped global hook registration (`runtime_config.py:22-294`). Class-level `_global_hooks` list persists across executions.

Supports auto-wiring of `GraphPersistenceHook` (via `persistence_enabled` + `persistence_sink`) and `DebugSSEHook` (via `debug_sse_enabled` + `debug_sse_sink`). See sections below.

### Layer 4: Context Factory (`magic_agents/hooks/context_factory.py:52-419`)

`HookContextFactory` provides 6 static factory methods for validated `HookContext` construction (`context_factory.py:52-419`). The 6 methods are:

| Method | Line | Purpose |
|--------|------|---------|
| `build_graph_context` | `context_factory.py:63` | Graph lifecycle events |
| `build_node_context` | `context_factory.py:121` | Node lifecycle events |
| `build_edge_context` | `context_factory.py:178` | Edge traversal events |
| `build_bypass_context` | `context_factory.py:247` | Node bypass events |
| `build_llm_context` | `context_factory.py:292` | LLM lifecycle events (node_type="LLM") |
| `build_tool_context` | `context_factory.py:354` | Tool lifecycle events (node_type="TOOL") |

Direct `HookContext()` construction is deprecated (`flow_hooks.py:167-183`).

## Runtime Flow

1. `agt_flow.py:359` (`execute_graph()`) or `agt_flow.py:422` (`execute_graph_loop()`) → merges `RuntimeConfig` + `AgentFlowModel.hooks` → creates `HookRegistry` (`agt_flow.py:403-411`).

2. `reactive_executor.py:308-438` → `_wire_hooks_to_registry()` at line 369 (auto-wires persistence/SSE hooks) → sets `hooks.execution_id` and `hooks.run_id` → `HookContextFactory.build_graph_context()` → `hooks.invoke("on_graph_start", ctx)` at line ~443.

3. For each node: `Node.__call__()` (`Node.py:220-237`) → stores `self._hooks` → `HookContextFactory.build_node_context()` → `hooks.invoke("on_node_start", ctx)` → `process()` → on success: `hooks.invoke("on_node_end", ctx)` → on error: `hooks.invoke("on_node_error", ctx, error=e)`.

4. `NodeLLM.process()` (`NodeLLM.py:342`) → `_create_hook_relay()` at `NodeLLM.py:292` → passes `HookRelay` as `hooks=` to magic-llm's `run_agent_async()` / `run_agent_stream_async()` for callable-tools paths. For schema-only and no-tools paths, hooks fire via direct `self._hooks.invoke()`.

5. On error/bypass: `reactive_executor.py:586-628` → `_propagate_error_bypass_with_hooks()` → `dispatcher.propagate_error_bypass()` → `HookContextFactory.build_bypass_context(reason="upstream_error")` → `hooks.invoke("on_node_bypass", ctx, reason="upstream_error")`.

## 3-Tier Dispatch Order

**Execution order**: Node → Graph → Global (innermost-first).

| Tier | Scope | Registration | Fire Condition |
|------|-------|-------------|----------------|
| 3 (Node) | Per-node | `registry.register_node(node_id, hook)` | Only when that specific node executes |
| 2 (Graph) | Per-graph | `registry.register_graph(hook)` or `AgentFlowModel.hooks` | Every node in the graph |
| 1 (Global) | Application | `RuntimeConfig.register_global_hook(hook)` or auto-wired via `_wire_hooks_to_registry()` | ALL executions in the process |

All hooks execute in parallel via `asyncio.gather(*tasks, return_exceptions=True)` (`hook_registry.py:163-195`). Errors are logged but never propagated.

## Graph Integration

- **AgentFlowModel.hooks** (`AgentFlowModel.py:156-158`): graph-level `FlowHooks` instance, merged with `RuntimeConfig` at execution time.
- **AgentFlowModel.persistence_enabled** (`AgentFlowModel.py:164-167`): graph-level persistence override. When `None` (default), defers to `RuntimeConfig.persistence_enabled`. When `True`/`False`, overrides RuntimeConfig value.
- **NodeInner hook propagation** (`NodeInner.py:143-171`): creates a child `HookRegistry`, copies `execution_id`/`run_id`, registers `inner_graph.hooks`, passes to `execute_graph_reactive()`.
- **EdgeHookConfig** (`EdgeNodeModel.py:24-52`): attaches a `NodeHook` to an edge; dispatcher invokes on traversal.

## LLM Integration

`NodeLLM` creates a `HookRelay` adapter (`NodeLLM.py:292-321`) bridging magic-llm's sync `AgentHooks` protocol to magic-agents async `FlowHooks`. HookRelay implements `on_iteration_start`, `on_llm_response`, `on_tool_start`, `on_tool_complete`, `on_loop_complete`, `on_budget_exceeded` (`hook_relay.py:36-764`).

### HookRelay vs Direct Hook Paths

`NodeLLM` has two hook invocation paths:

1. **HookRelay path** (callable tools — `tool_functions` present): `HookRelay` bridges magic-llm's `AgentHooks` to `FlowHooks`. All 5 LLM/tool events fire through HookRelay translation.
2. **Direct path** (schema-only tools or no tools): `NodeLLM` calls `self._hooks.invoke()` directly for `on_llm_start`, `on_llm_end`, `on_llm_loop_end`. Tool events (`on_tool_start`, `on_tool_end`) only fire from the HookRelay path.

### Tool Context Construction

Tool events (`on_tool_start`, `on_tool_end`) are routed through `_build_tool_context()` (`hook_relay.py:193-258`) which calls `HookContextFactory.build_tool_context()` with `node_type="TOOL"`. This is distinct from LLM events which use `_build_context()` (`hook_relay.py:134-191`) → `HookContextFactory.build_llm_context()` with `node_type="LLM"`.

## NodeHook Integration

`NodeHook.process()` (`NodeHook.py:101-192`) receives a `HookContext` via its input handle (`INPUT_HANDLE_HOOK_CONTEXT`), compiles the Python function template via `exec()`, and executes it with timeout enforcement.

---

## GraphPersistenceHook (FlowHooks-Level)

**File**: `magic_agents/hooks/persistence.py:76-370`

**Status**: ✅ Implemented and auto-wired.

`GraphPersistenceHook` is a `FlowHooks` implementation that records graph/node execution tree data, run/execution lifecycle, LLM provider/model metadata, tool calls/results, and usage facts.

### Key Details

| Aspect | Detail |
|--------|--------|
| Repository | magic-agents |
| Protocol level | FlowHooks (12-method async protocol) |
| Sink interface | `ExecutionPersistencePort` (protocol at `persistence.py:16`) |
| Scope | Graph/node execution tree, run/execution lifecycle, LLM provider/model metadata, tool calls/results, usage facts |
| Wiring | `RuntimeConfig.persistence_enabled` + `persistence_sink` via `_wire_hooks_to_registry()` in `reactive_executor.py:257-303` |
| Default | `persistence_enabled=True` (no-op without sink); `persistence_sink=None` (no sink = no hook created) |
| Graph override | `AgentFlowModel.persistence_enabled` (default `None` = defer to RuntimeConfig; `True`/`False` overrides) |

### Wiring Flow

```
RuntimeConfig(persistence_enabled=True, persistence_sink=my_sink)
  └─ _wire_hooks_to_registry() at reactive_executor.py:369
       └─ getattr(graph, 'persistence_enabled', None) — graph override check
            └─ if False: skip entirely
            └─ if True/None: check runtime_config.is_persistence_enabled()
                 └─ runtime_config.build_persistence_hook(id_chat, id_thread, id_user, ...)
                      └─ returns GraphPersistenceHook(...) or None if no sink
                 └─ registry.register_global(hook) — global tier, no new manager
```

### Ownership Boundary

`GraphPersistenceHook` operates at the `FlowHooks` level. It does NOT replicate LLM event ownership (which belongs to magic-llm's `AgentPersistenceHooks`). It does NOT directly import or wrap `AgentPersistenceHooks`. It consumes already-translated FlowHooks events that arrive via `HookRelay`.

If both `GraphPersistenceHook` (magic-agents, FlowHooks-level) and `AgentPersistenceHooks` (magic-llm, AgentHooks-level) are registered in the same execution, they fire at different protocol levels — no double-recording occurs.

See `docs/hooks/PERSISTENCE.md` for the full boundary comparison.

---

## DebugSSEHook (SSE Debug Events)

**File**: `magic_agents/hooks/debug_sse.py:22-181`

**Status**: ✅ Implemented — all 12 FlowHooks methods complete. Auto-wired via RuntimeConfig.

`DebugSSEHook` emits SSE-compatible debug event envelopes to an `asyncio.Queue` sink for development/debugging purposes.

### Implemented Methods

| Method | Line | Event Type |
|--------|------|------------|
| `on_graph_start` | `debug_sse.py:48` | `graph_start` |
| `on_graph_end` | `debug_sse.py:59` | `graph_end` |
| `on_graph_error` | `debug_sse.py:66` | `graph_error` |
| `on_node_start` | `debug_sse.py:77` | `node_start` |
| `on_node_end` | `debug_sse.py:88` | `node_end` |
| `on_node_error` | `debug_sse.py:99` | `node_error` |
| `on_node_bypass` | `debug_sse.py:110` | `node_bypass` |
| `on_llm_start` | `debug_sse.py:115` | `llm_start` |
| `on_llm_end` | `debug_sse.py:130` | `llm_end` |
| `on_llm_loop_end` | `debug_sse.py:147` | `llm_loop_end` |
| `on_tool_start` | `debug_sse.py:161` | `tool_start` |
| `on_tool_end` | `debug_sse.py:172` | `tool_end` |

### Auto-Wiring

- Default: **OFF** (`debug_sse_enabled=False`)
- Enable via `RuntimeConfig.enable_debug_sse(sink)` passing an `asyncio.Queue` or `DebugEventSink`
- Wired via `_wire_hooks_to_registry()` in `reactive_executor.py:257-303` — same pattern as `GraphPersistenceHook`
- Registered as global hook on existing `HookRegistry`

### Limitations

- **Queue-full drops**: When the SSE sink queue is full, events are dropped with a WARNING log message (`DebugSSEHook queue full; dropping %s event`). This is non-fatal — no exception propagates.
- **Not production observability**: `DebugSSEHook` is intended for development/debugging only. For production observability, use `GraphPersistenceHook` with a proper `ExecutionPersistencePort` sink.
- **No automatic runtime wiring**: SSE events are only emitted when `debug_sse_enabled=True` AND a sink is configured. There is no SSE server or endpoint integration — the consumer must read from the queue.

---

## AgentPersistenceHooks (AgentHooks-Level, magic-llm)

**File**: `magic-llm/magic_llm/agent/persistence_hooks.py`

**Status**: ✅ Implemented. Available for standalone magic-llm consumers. NOT part of this change's wiring.

`AgentPersistenceHooks` is an `AgentHooks`-level persistence implementation in magic-llm. It tracks agent loop iteration lifecycle, raw token usage, and standalone loop-level persistence.

| Aspect | Detail |
|--------|--------|
| Repository | magic-llm |
| Protocol level | AgentHooks (6-method sync protocol) |
| Sink interface | `AgentPersistenceSink` (magic-llm) |
| Primary use case | Lightweight loop-level persistence without graph context |
| Wiring status | Available for standalone magic-llm consumers (NOT wired in this change) |

### Coexistence with GraphPersistenceHook

Both hooks can coexist safely:
- `AgentPersistenceHooks` fires at AgentHooks level inside magic-llm's agent loop (sync, per-iteration)
- `GraphPersistenceHook` fires at FlowHooks level after HookRelay translation (async, aggregated)
- Different protocol levels — no double-recording, no conflicts

---

## Nested Execution Correlation

`HookRelay` supports nested LLM execution correlation via `contextvars` propagated from magic-llm:

### ContextVars

| ContextVar | Source | Purpose |
|------------|--------|---------|
| `PARENT_HOOKS` | `magic-llm/agent/_loop_shared.py` | Propagates `HookRelay` instance to child agent loops for nested LLM nodes |
| `DEPTH` | `magic-llm/agent/_loop_shared.py` | Auto-incrementing nesting depth counter |

### Metadata Fields

Every `HookContext` produced by `HookRelay` carries the following metadata:

| Field | Source | Description |
|-------|--------|-------------|
| `nested_depth` | `DEPTH` ContextVar or `_nested_depth` | Current nesting depth (0 = root, 1 = first child, etc.) |
| `nested_request_id` | `uuid.uuid4().hex` at `HookRelay` construction | Unique UUID per nested invocation chain |
| `parent_run_id` | Passed at `HookRelay` construction | Run ID of the parent loop (present only in nested contexts) |

### Implementation

- `HookRelay.__init__()` (`hook_relay.py:36-115`): accepts `parent_run_id`, `nested_depth`, `nested_request_id` constructor params
- `_build_context()` (`hook_relay.py:140-192`): injects nested correlation into `ctx.metadata` using runtime `DEPTH` ContextVar when available (more accurate) or the construction-time `_nested_depth` as fallback
- `_build_tool_context()` (`hook_relay.py:193-258`): same correlation injection pattern for tool events
- Feature flags: `ENABLE_SUBAGENTS` and `ENABLE_NESTED_LLM_NODES` both default to `False` (opt-in required for nested execution)

### Sequence

```
Root HookRelay (nested_depth=0)
  └─ PARENT_HOOKS ContextVar set → child AsyncAgentLoop
       └─ Child HookRelay (nested_depth=1, parent_run_id=<root_run_id>)
            └─ DEPTH ContextVar auto-incremented
                 └─ All HookContexts carry nested_depth=1, nested_request_id, parent_run_id
```
