# Debug system

Debugging in Magic Agents is event-driven.

## How to enable it

```json
{
  "debug": true,
  "debug_config": {
    "preset": "verbose",
    "redact_sensitive": true,
    "max_payload_length": 2000
  }
}
```

Actual activation is gated in three places, not just by the graph JSON:

1. global debug must be enabled
2. `graph.debug` must be `true`
3. resolved `debug_config.enabled` must still be `true`

If any gate is off, execution uses a `NullObserver`.

## What `debug_config` supports

`DebugConfig.from_dict()` currently supports presets and overrides such as:

- `preset`
- `enabled`
- `min_severity`
- `redact_sensitive`
- `additional_redact_keys`
- `max_payload_length`
- `max_list_items`
- `capture_inputs`
- `capture_outputs`
- `capture_internal_state`
- `emit_to_log`
- `log_level`
- `log_format_json`
- `sample_rate`
- `include_nodes` / `exclude_nodes`
- `include_event_types` / `exclude_event_types`

## Presets

Current preset names:

- `default`
- `minimal`
- `verbose`
- `production`
- `errors_only`

## Runtime debug outputs

| Event type | Shape |
| --- | --- |
| `debug` | per-error or per-node structured payload |
| `debug_summary` | final `GraphDebugFeedback` payload |
| `loop_progress` | loop progress event |

## Observer architecture

Debug capture is handled through `ObserverRegistry`, not by ad-hoc prints.

- `ObserverRegistry.create(...)` decides between `NullObserver` and `DefaultObserver`
- `DefaultObserver` uses emitter registries/callback emitters to forward structured events
- if a node provides its own observer, `ObserverRegistry.observer_for(...)` can return either that observer alone or a `CompositeObserver` that runs both parent + child observers

This is why the debug system is more than "emit a few dicts": it has per-execution observer resolution, filtering, and callback integration.

## Node-level debug payloads

Node debug data can include:

- node identifiers and class name
- start/end timestamps
- execution duration
- captured inputs
- captured outputs
- internal variables
- bypass/execution state
- error message if execution failed

## Error reporting model

Nodes usually emit debug events instead of throwing hard exceptions into the caller. Examples:

- missing required input
- invalid conditional template
- JSON parsing failures
- MCP transport/protocol failures

Validation diagnostics can also be surfaced as debug payloads, but only blocking `GraphValidationError` entries abort execution immediately.

## Important nuance

`run_agent()` output is not guaranteed to be only `ChatCompletionModel` objects when debug is enabled. Debug events are plain dict events.

That is why consumers and tests must branch on the event `type` instead of blindly assuming every item has `.choices`.

## Related systems that touch debug output

- `CallbackEmitter` in `magic_agents.agt_flow` bridges selected structured execution/debug events to external callbacks; current direct emissions are graph boundary events. See [../hooks/CALLBACK_EMITTER.md](../hooks/CALLBACK_EMITTER.md).
- hooks and debug observers are separate systems: hooks observe lifecycle, observers format/filter/emit debug data
