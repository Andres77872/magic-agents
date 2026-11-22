# Graph persistence hook

## Purpose

Documents `GraphPersistenceHook`, the persistence hook for graph executions,
and what it does and does not own.

The published `magic_llm` package has no `AgentPersistenceHooks` or
`AgentPersistenceSink`; earlier versions of this page described them as an
upstream counterpart. Persistence for graph runs is only provided here.

## Summary

| Aspect | GraphPersistenceHook |
|--------|---------------------|
| **File** | `magic_agents/hooks/persistence.py` (`GraphPersistenceHook`) |
| **Protocol level** | FlowHooks (12-method async protocol) |
| **Scope** | Graph/node execution tree, run/execution lifecycle, LLM provider/model metadata, tool calls/results, usage facts |
| **Sink interface** | `ExecutionPersistencePort` (`magic_agents/hooks/persistence.py`) |
| **Wiring** | `RuntimeConfig.persistence_enabled` + `_wire_hooks_to_registry()` in `reactive_executor.py` |

## Ownership Boundary

### CallbackEmitter is a separate compatibility bridge

`magic_agents.agt_flow.CallbackEmitter` is a module-level synchronous callback registry, not a `FlowHooks` persistence hook. Current source dispatches graph boundary debug envelopes (`GRAPH_START`, `GRAPH_END`) through it from the reactive executors. It is useful for lightweight host-app integration, but it does not implement `ExecutionPersistencePort`, does not receive the full `FlowHooks` lifecycle, and should not be treated as a replacement for `GraphPersistenceHook`. See [CALLBACK_EMITTER.md](CALLBACK_EMITTER.md).

### GraphPersistenceHook

- Operates at the **FlowHooks** level (12-method async protocol)
- Tracks **graph/node execution tree** — run/execution lifecycle, LLM provider/model metadata, tool calls/results, usage facts
- Uses `ExecutionPersistencePort` sink protocol (injectable by API layer consumer)
- Consumes **already-translated** FlowHooks events that arrive via `HookRelay`
- **Does NOT** create its own LLM-level agent loop hooks
- **Does NOT** replicate LLM event ownership (which belongs to magic-llm)

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

`ExecutionPersistencePort` is a `@runtime_checkable` Protocol defined at `magic_agents/hooks/persistence.py`. It is the sink/adapter abstraction for graph persistence.

The concrete sink implementation is **injected by the API layer consumer** — it is NOT part of this change. For testing, use a mock/fake `ExecutionPersistencePort` to verify hook wiring.
