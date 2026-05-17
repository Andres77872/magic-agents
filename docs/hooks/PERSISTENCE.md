# Persistence Hook Ownership Boundary

## Purpose

Documents the ownership boundary between the two persistence hook implementations across the two repositories, so consumers understand when to use each one and that they are complementary, not duplicative.

## Comparison Table

| Aspect | GraphPersistenceHook | AgentPersistenceHooks |
|--------|---------------------|----------------------|
| **Repository** | magic-agents | magic-llm |
| **File** | `magic_agents/hooks/persistence.py:76-370` | `magic_llm/agent/persistence_hooks.py` |
| **Protocol level** | FlowHooks (12-method async protocol) | AgentHooks (6-method sync protocol) |
| **Scope** | Graph/node execution tree, run/execution lifecycle, LLM provider/model metadata, tool calls/results, usage facts | Agent loop iteration lifecycle, raw token usage, standalone loop-level persistence |
| **Sink interface** | `ExecutionPersistencePort` (magic-agents, at `persistence.py:16`) | `AgentPersistenceSink` (magic-llm) |
| **Primary use case** | Full graph observability with execution tree tracking | Lightweight loop-level persistence without graph context |
| **Wiring status** | Wired via `RuntimeConfig.persistence_enabled` + `_wire_hooks_to_registry()` in `reactive_executor.py` | Available for standalone magic-llm consumers (NOT part of this change) |
| **Can coexist?** | Yes — different protocol levels, no double-recording | Yes — fires at AgentHooks level inside magic-llm's agent loop |

## Ownership Boundary

### CallbackEmitter is a separate compatibility bridge

`magic_agents.agt_flow.CallbackEmitter` is a module-level synchronous callback registry, not a `FlowHooks` persistence hook. Current source dispatches graph boundary debug envelopes (`GRAPH_START`, `GRAPH_END`) through it from the reactive executors. It is useful for lightweight host-app integration, but it does not implement `ExecutionPersistencePort`, does not receive the full `FlowHooks` lifecycle, and should not be treated as a replacement for `GraphPersistenceHook`. See [CALLBACK_EMITTER.md](CALLBACK_EMITTER.md).

### GraphPersistenceHook (magic-agents)

- Operates at the **FlowHooks** level (12-method async protocol)
- Tracks **graph/node execution tree** — run/execution lifecycle, LLM provider/model metadata, tool calls/results, usage facts
- Uses `ExecutionPersistencePort` sink protocol (injectable by API layer consumer)
- Consumes **already-translated** FlowHooks events that arrive via `HookRelay`
- **Does NOT** directly import or wrap `AgentPersistenceHooks`
- **Does NOT** create its own LLM-level agent loop hooks
- **Does NOT** replicate LLM event ownership (which belongs to magic-llm)

### AgentPersistenceHooks (magic-llm)

- Operates at the **AgentHooks** level (6-method sync protocol)
- Tracks **agent loop iteration lifecycle** — raw token usage, standalone loop-level persistence
- Uses `AgentPersistenceSink` protocol
- Remains available for **standalone magic-llm consumers** (no magic-agents dependency)
- **NOT part of this change** — no modifications and no wiring changes

## Coexistence

Both hooks can be active in the same execution:

```
magic-llm agent loop
  ├─ AgentPersistenceHooks.on_iteration_start()   ← AgentHooks level (sync)
  ├─ AgentPersistenceHooks.on_llm_response()       ← AgentHooks level (sync)
  ├─ AgentPersistenceHooks.on_loop_complete()      ← AgentHooks level (sync)
  │
  └─ via HookRelay →
       ├─ GraphPersistenceHook.on_llm_start()       ← FlowHooks level (async)
       ├─ GraphPersistenceHook.on_llm_end()         ← FlowHooks level (async)
       └─ GraphPersistenceHook.on_llm_loop_end()    ← FlowHooks level (async)
```

- No double-recording occurs — they fire at **different protocol levels**
- `AgentPersistenceHooks` fires inside magic-llm's agent loop (sync, `AgentHooks`)
- `GraphPersistenceHook` fires after `HookRelay` translation (async, `FlowHooks`)
- Both record the same underlying events at their respective abstraction levels

## Recommendations

| Use Case | Recommended Hook |
|----------|-----------------|
| Full graph observability with execution tree tracking | **GraphPersistenceHook** (magic-agents, FlowHooks-level) |
| Standalone loop-level persistence without graph context | **AgentPersistenceHooks** (magic-llm, AgentHooks-level) |
| Both (complex deployment with full observability) | **Both** — they coexist at different protocol levels |

**For magic-agents deployments**: Use `GraphPersistenceHook` configured via `RuntimeConfig.enable_persistence(sink)`. This provides full graph/node execution tree tracking, LLM provider metadata, tool calls/results, and usage facts — all at the FlowHooks abstraction level that matches graph execution semantics.

**For standalone magic-llm deployments**: Use `AgentPersistenceHooks` directly by implementing `AgentPersistenceSink` and registering the hook with the agent loop. This provides lightweight loop-level persistence without needing any magic-agents dependency.

## Configuration

### Enabling GraphPersistenceHook

```python
from magic_agents.hooks.runtime_config import RuntimeConfig
from my_sink import MyPersistenceSink

config = RuntimeConfig()
config.enable_persistence(MyPersistenceSink())
# GraphPersistenceHook auto-wired for all executions using this config
```

### Graph-Level Override

```python
# Graph JSON — disable persistence for this specific graph
agent_flow_model.persistence_enabled = False
```

### Default Behavior

- `RuntimeConfig.persistence_enabled` defaults to **True**
- `RuntimeConfig.persistence_sink` defaults to **None** (no sink)
- Without a configured sink, `build_persistence_hook()` returns `None` — effectively a no-op
- This means existing deployments see **zero behavior change** even with the `True` default
- At the graph level, `persistence_enabled=None` means "defer to RuntimeConfig" — backward compatible for all existing graphs

## Sink Protocol: ExecutionPersistencePort

`ExecutionPersistencePort` is a `@runtime_checkable` Protocol defined at `magic_agents/hooks/persistence.py:16`. It is the sink/adapter abstraction for graph persistence.

The concrete sink implementation is **injected by the API layer consumer** — it is NOT part of this change. For testing, use a mock/fake `ExecutionPersistencePort` to verify hook wiring.
