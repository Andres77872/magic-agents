# Debug Mode Documentation

Welcome to the Magic Agents Debug Mode documentation.

## What is Debug Mode?

Debug mode is a comprehensive execution tracking system that captures detailed information about node execution, data flow, and performance metrics during graph execution.

## Architecture Overview

The debug system uses a **Capture-Transform-Emit** pipeline:

```
┌──────────────┐    ┌─────────────────┐    ┌───────────────────┐
│   CAPTURE    │───▶│   TRANSFORM     │───▶│      EMIT         │
│   (Hooks)    │    │   (Pipeline)    │    │   (Dispatcher)    │
└──────────────┘    └─────────────────┘    └───────────────────┘
```

- **Capture**: Hooks that create `DebugEvent` instances when lifecycle events occur
- **Transform**: Pipeline of transformers (redact, filter, truncate) that process events
- **Emit**: Dispatchers that deliver events to queues, logs, or callbacks

## Quick Start

Enable debug mode by setting `debug: true` in your graph JSON:

```json
{
  "type": "chat",
  "debug": true,
  "nodes": [...],
  "edges": [...]
}
```

### Debug Configuration from JSON

You can also configure debug behavior directly in the JSON:

```json
{
  "type": "chat",
  "debug": true,
  "debug_config": {
    "preset": "verbose",
    "redact_sensitive": true,
    "max_payload_length": 2000
  },
  "nodes": [...],
  "edges": [...]
}
```

**Available presets:** `default`, `minimal`, `verbose`, `production`, `errors_only`

**Common options:**
- `redact_sensitive`: Hide API keys, passwords (default: `true`)
- `max_payload_length`: Truncate long strings (default: `1000`)
- `capture_inputs`/`capture_outputs`: Toggle data capture (default: `true`)
- `include_nodes`/`exclude_nodes`: Filter specific nodes

Collect debug feedback during execution:

```python
async for result in run_agent(graph):
    if result.get("type") == "content":
        # Normal streaming content
        chat_model = result["content"]
        # Process ChatCompletionModel...
    elif result.get("type") == "debug":
        # Per-node debug info
        node_info = result["content"]
        print(f"Node {node_info['node_id']} executed")
    elif result.get("type") == "debug_summary":
        # Final summary
        summary = result["content"]
        print(f"Total execution: {summary['total_duration_ms']}ms")
```

## Documentation Files

1. **[Debug Mode Overview](./DEBUG_MODE_OVERVIEW.md)** - Introduction and use cases
2. **[Debug Feedback Structure](./DEBUG_FEEDBACK_STRUCTURE.md)** - Complete structure of debug output
3. **[Node Debug Information](./NODE_DEBUG_INFORMATION.md)** - Node-specific debug data
4. **[Debug Mode Examples](./DEBUG_MODE_EXAMPLES.md)** - Practical usage examples

## Key Features

- **Unified Event Model**: All debug events use a single `DebugEvent` structure
- **Capture-Transform-Emit Pipeline**: Clean separation of concerns
- **Comprehensive Tracking**: Inputs, outputs, and internal state for every node
- **Timing Information**: Precise execution timing at millisecond level
- **Error Capture**: Detailed error messages when nodes fail
- **Conditional Flow Tracking**: See which branches were taken
- **Loop Iteration Details**: Track each iteration
- **Safe Serialization**: JSON-compatible output with automatic redaction
- **Backward Compatibility**: Legacy format support via `to_legacy_format()`

## Debug Module Structure

```
magic_agents/debug/
├── events.py        # DebugEvent, DebugEventType, DebugEventSeverity
├── capture.py       # DefaultDebugCapture, DebugCaptureHook protocol
├── transform.py     # TransformPipeline, RedactTransformer, FilterTransformer
├── emitter.py       # QueueEmitter, LogEmitter, CallbackEmitter
├── collector.py     # DebugCollector, GraphExecutionSummary
├── context.py       # DebugContext, debug_context()
└── config.py        # DebugConfig, presets (default, minimal, verbose)
```

## Important Notes

**Performance**: Debug mode adds overhead. Disable in production unless troubleshooting.

**Privacy**: Debug mode captures all inputs/outputs. Use `RedactTransformer` to hide sensitive data.

**Output Size**: Debug feedback can be large for complex graphs. Use `TruncateTransformer` to limit payload sizes.
