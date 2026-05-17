# CallbackEmitter — Module-Level Debug Callback Registry

## Purpose

`CallbackEmitter` is a small module-level callback registry in `magic_agents/agt_flow.py:93-117`. It lets host applications register synchronous callbacks for selected structured debug events emitted by the reactive executors.

This is **not** the main hook system. It is a compatibility/debug bridge for external consumers that need coarse execution lifecycle events without implementing `FlowHooks`.

## API

```python
from magic_agents.agt_flow import CallbackEmitter

def persist_event(event: dict, chat_log) -> None:
    event_type = event.get("content", {}).get("event_type")
    # Store or forward the event. Keep this fast and non-blocking.

CallbackEmitter.register(persist_event)

# Later, when no longer needed:
CallbackEmitter.unregister(persist_event)
```

Callbacks receive two arguments:

| Argument | Type | Description |
| --- | --- | --- |
| `event` | `dict` | A yielded debug event envelope, usually `{ "type": "debug", "content": {...} }` |
| `chat_log` | `ModelAgentRunLog \| None` | Current run log object when available |

`CallbackEmitter.emit(event, chat_log)` iterates registered callbacks synchronously. Callback exceptions are caught and logged; they do not abort graph execution.

## Current Emission Points

The current source only calls `CallbackEmitter.emit()` from the reactive executors for graph boundary events:

| Event | Source |
| --- | --- |
| `GRAPH_START` | `execute_graph_reactive()` (`reactive_executor.py:417-432`) and `execute_graph_loop_reactive()` (`reactive_executor.py:869-884`) |
| `GRAPH_END` | `execute_graph_reactive()` completion path (`reactive_executor.py:771-784`) |

Other debug events such as `NODE_START`, `NODE_END`, `LLM_GENERATION`, `TOOL_CALL`, `TOOL_RESULT`, `ITERATION_START`, `ITERATION_END`, `SUBGRAPH_START`, and `SUBGRAPH_END` can be yielded by nodes/executors or delivered through the debug observer/callback system, but they are **not currently dispatched through `agt_flow.CallbackEmitter.emit()` unless code explicitly calls it**.

## Relationship to GraphPersistenceHook

| Concern | `CallbackEmitter` | `GraphPersistenceHook` |
| --- | --- | --- |
| File | `magic_agents/agt_flow.py:93-117` | `magic_agents/hooks/persistence.py:76-370` |
| Protocol | Module-level synchronous callback registry | `FlowHooks` async implementation |
| Registration | `CallbackEmitter.register(callback)` | `RuntimeConfig.enable_persistence(sink)` / registry auto-wiring |
| Scope | Coarse debug event envelopes emitted manually by executor code | Graph/node/LLM/tool lifecycle via `HookRegistry` and `HookRelay` |
| Sink contract | Callback receives `(event, chat_log)` | `ExecutionPersistencePort` methods |
| Best use | Lightweight compatibility bridge, quick debugging, host-app glue | Production graph persistence/observability |

Use `GraphPersistenceHook` for durable execution-tree persistence. Use `CallbackEmitter` only when you need the existing module-level callback bridge or a lightweight observer for the events it actually emits.

## Relationship to Debug CallbackEmitter

There is also a separate `magic_agents.debug.emitter.CallbackEmitter` class. That one is an instance-based debug emitter for `DebugEvent` objects and supports sync/async callbacks (`debug/emitter.py:405-442`). Do not confuse it with `magic_agents.agt_flow.CallbackEmitter`, which is module-level and receives raw debug-event dictionaries.

## Limitations

- Module-level mutable state persists until callbacks are unregistered. Tests and long-running hosts should unregister callbacks or isolate setup/teardown.
- Callbacks are synchronous. Keep them quick; offload slow I/O to your own queue/worker.
- Event coverage is intentionally limited by current call sites. Do not assume every yielded debug event passes through this registry.
