# Debug Mode Usage Guide

## Overview

Debug mode in Magic Agents provides detailed execution information about your graph flows. When enabled, the system captures inputs, outputs, timing, and execution status for all nodes in your graph.

## Enabling Debug Mode

Set `"debug": true` in your graph definition:

```json
{
  "type": "chat",
  "debug": true,
  "nodes": [...],
  "edges": [...]
}
```

### Debug Configuration from JSON

You can customize debug behavior directly in the JSON using `debug_config`:

```json
{
  "type": "chat",
  "debug": true,
  "debug_config": {
    "preset": "verbose",
    "redact_sensitive": true,
    "max_payload_length": 2000,
    "capture_inputs": true,
    "capture_outputs": true
  },
  "nodes": [...],
  "edges": [...]
}
```

#### Available Presets

Use the `preset` field to start from a predefined configuration:

| Preset | Description |
|--------|-------------|
| `default` | Standard debugging with sensible defaults |
| `minimal` | Errors and warnings only, no input/output data |
| `verbose` | Captures everything including trace events |
| `production` | Production-safe with sampling and redaction |
| `errors_only` | Only error events |

#### Configuration Options

All options can be set in `debug_config`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `preset` | string | - | Base preset to start from |
| `enabled` | bool | `true` | Master switch for debug |
| `redact_sensitive` | bool | `true` | Redact API keys, passwords, etc. |
| `max_payload_length` | int | `1000` | Max string length in debug output |
| `max_list_items` | int | `20` | Max items in list output |
| `capture_inputs` | bool | `true` | Capture node input data |
| `capture_outputs` | bool | `true` | Capture node output data |
| `capture_internal_state` | bool | `true` | Capture internal node state |
| `emit_to_log` | bool | `false` | Also emit to logging |
| `log_level` | string | `"DEBUG"` | Log level for emissions |
| `sample_rate` | float | `1.0` | Fraction of events to include (1.0 = all) |
| `include_nodes` | array | - | Only debug specific nodes |
| `exclude_nodes` | array | - | Exclude specific nodes |

#### Example Configurations

**Minimal debugging for production:**
```json
{
  "debug": true,
  "debug_config": {
    "preset": "minimal",
    "redact_sensitive": true
  }
}
```

**Verbose debugging with logging:**
```json
{
  "debug": true,
  "debug_config": {
    "preset": "verbose",
    "emit_to_log": true,
    "log_level": "INFO"
  }
}
```

**Debug only specific nodes:**
```json
{
  "debug": true,
  "debug_config": {
    "include_nodes": ["llm-1", "parser-1"],
    "capture_internal_state": false
  }
}
```

## Handling Debug Output

All messages from `run_agent()` are yielded as **dictionaries with a `type` field**.

### Message Types

- `{"type": "content", "content": ChatCompletionModel}` - Streaming content chunks
- `{"type": "debug", "content": {...}}` - Per-node debug info (when debug enabled)
- `{"type": "debug_summary", "content": {...}}` - Final summary (when debug enabled)

### ⚠️ Common Mistake

```python
# ❌ WRONG - Assumes result is ChatCompletionModel directly
async for result in run_agent(graph):
    print(result.choices[0].delta.content, end='')
```

**Error:** Results are wrapped in dicts with `type` and `content` fields!

### ✅ Correct Pattern

```python
async for result in run_agent(graph):
    # Check message type first
    if result.get("type") == "content":
        # Extract ChatCompletionModel from content field
        chat_model = result["content"]
        if hasattr(chat_model, 'choices') and chat_model.choices:
            delta = chat_model.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                print(delta.content, end='')
    
    elif result.get("type") == "debug":
        # Per-node debug feedback
        debug_info = result["content"]
        print(f"Node executed in {debug_info['execution_duration_ms']:.2f}ms")
    
    elif result.get("type") == "debug_summary":
        # Final summary debug feedback
        summary = result["content"]
        print(f"Total execution: {summary['total_duration_ms']:.2f}ms")
```

## Debug Output Structure

When debug mode is enabled, a debug message is yielded at the end of execution:

```python
{
    "type": "debug",
    "content": {
        "execution_id": "abc123...",
        "graph_type": "chat",
        "start_time": "2025-10-10T06:12:21.123456",
        "end_time": "2025-10-10T06:12:23.456789",
        "total_duration_ms": 2333.333,
        "total_nodes": 5,
        "executed_nodes": 4,
        "bypassed_nodes": 1,
        "failed_nodes": 0,
        "nodes": [...],
        "edges_processed": [...]
    }
}
```

### Node Information

Each node in the `nodes` array contains:

```python
{
    "node_id": "llm-1",
    "node_type": "llm",
    "node_class": "NodeLLM",
    "start_time": "2025-10-10T06:12:21.200000",
    "end_time": "2025-10-10T06:12:22.500000",
    "execution_duration_ms": 1300.0,
    "inputs": {
        "handle_user_message": "What is AI?",
        "handle-client-provider": "<MagicLLM>"
    },
    "outputs": {
        "end": {
            "node": "NodeLLM",
            "content": "AI stands for..."
        }
    },
    "internal_variables": {
        "_response": "AI stands for...",
        "cost": 0.0,
        "stream": true,
        "json_output": false,
        "iterate": false,
        "generated": "AI stands for...",
        "extra_data": {"temperature": 0.7}
    },
    "was_executed": true,
    "was_bypassed": false,
    "error": null
}
```

## Key Features

### 1. Executed vs Bypassed Nodes

- **Executed nodes**: `was_executed = true` - Node ran and produced output
- **Bypassed nodes**: `was_bypassed = true` - Node was skipped (e.g., in conditional branches)

Only nodes that were either executed OR bypassed are included in debug output. Nodes that were never reached are excluded.

### 2. Bypassed Node Information

When a node is bypassed (e.g., in a conditional branch not taken), the debug system still captures:
- Current inputs at the time of bypass
- Current outputs (usually empty)
- Internal state
- Proper flags: `was_executed=false`, `was_bypassed=true`

### 3. Timing Information

Each node includes:
- `start_time`: ISO timestamp when execution started
- `end_time`: ISO timestamp when execution completed
- `execution_duration_ms`: Duration in milliseconds

Bypassed nodes have `null` timing values.

### 4. Edge Processing

The `edges_processed` array shows the order edges were processed:

```python
[
    {
        "source": "user-input-1",
        "target": "llm-1",
        "source_handle": "handle_user_message",
        "target_handle": "handle_user_message"
    },
    ...
]
```

## Examples

### Example 1: Performance Analysis

```python
async for result in run_agent(graph):
    if result.get("type") == "debug_summary":
        debug_info = result["content"]
        
        # Find bottlenecks
        sorted_nodes = sorted(
            [n for n in debug_info['nodes'] if n['execution_duration_ms']],
            key=lambda n: n['execution_duration_ms'],
            reverse=True
        )
        
        print("Slowest nodes:")
        for node in sorted_nodes[:3]:
            print(f"  {node['node_id']}: {node['execution_duration_ms']:.2f}ms")
```

### Example 2: Conditional Flow Analysis

```python
async for result in run_agent(graph):
    if result.get("type") == "debug_summary":
        debug_info = result["content"]
        
        # Check which branch was taken
        conditional_nodes = [
            n for n in debug_info['nodes'] 
            if n['node_class'] == 'NodeConditional'
        ]
        
        for cond_node in conditional_nodes:
            outputs = cond_node['outputs']
            branch = list(outputs.keys())[0] if outputs else "none"
            print(f"Conditional {cond_node['node_id']} took branch: {branch}")
        
        # Show bypassed nodes
        bypassed = [n for n in debug_info['nodes'] if n['was_bypassed']]
        print(f"Bypassed {len(bypassed)} nodes: {[n['node_id'] for n in bypassed]}")
```

### Example 3: Save Debug Output

```python
import json

per_node_debug = []
debug_summary = None

async for result in run_agent(graph):
    if result.get("type") == "debug":
        per_node_debug.append(result["content"])
    elif result.get("type") == "debug_summary":
        debug_summary = result["content"]

# Save to file
with open('debug_output.json', 'w') as f:
    json.dump(debug_data, f, indent=2)
```

## Best Practices

1. **Check message type first**: Always check if result is a debug message before accessing content attributes
2. **Disable in production**: Debug mode adds overhead - only enable during development/testing
3. **Filter relevant nodes**: Debug output only includes executed/bypassed nodes (not unreached nodes)
4. **Use timing data**: Identify performance bottlenecks with `execution_duration_ms`
5. **Analyze conditional flows**: Use `was_bypassed` to understand which paths were taken

## Troubleshooting

### "AttributeError: 'dict' object has no attribute 'choices'"

**Cause:** Not extracting the ChatCompletionModel from the message wrapper.

**Solution:** Check message type and extract content:
```python
if result.get("type") == "content":
    chat_model = result["content"]  # Extract ChatCompletionModel
    # Now access chat_model.choices[0].delta.content
elif result.get("type") == "debug":
    # Handle debug message
```

### "No debug information returned"

**Possible causes:**
1. `"debug": true` not set in graph definition
2. Graph execution failed before debug collection
3. Filtering out debug messages in your code

**Solution:** Verify graph definition has `"debug": true` and that you're checking for debug messages correctly.

### "Bypassed nodes have no information"

**Previous behavior:** Bypassed nodes had minimal information.

**Fixed:** Bypassed nodes now capture their current state when marked as bypassed, including inputs, outputs, and internal variables.

## See Also

- [Debug Mode Overview](./DEBUG_MODE_OVERVIEW.md)
- [Debug Feedback Structure](./DEBUG_FEEDBACK_STRUCTURE.md)
- [Debug Mode Examples](./DEBUG_MODE_EXAMPLES.md)

---

## Programmatic Debug Configuration (New Architecture)

The new debug system provides programmatic configuration through the `magic_agents.debug` module.

### Using DebugConfig

```python
from magic_agents.debug import DebugConfig

# Use a preset
config = DebugConfig.default()       # Full capture
config = DebugConfig.minimal()       # Low overhead
config = DebugConfig.verbose()       # Maximum detail
config = DebugConfig.errors_only()   # Production-safe

# Custom configuration
config = DebugConfig(
    enabled=True,
    capture_inputs=True,
    capture_outputs=True,
    capture_internal_state=False,  # Skip internal vars for performance
    max_string_length=500,
    redact_patterns=["password", "api_key", "secret"],
    include_event_types=[
        DebugEventType.NODE_START,
        DebugEventType.NODE_END,
        DebugEventType.NODE_ERROR
    ]
)
```

### Using DebugContext

```python
from magic_agents.debug import debug_context, DebugConfig

async def run_with_debug():
    config = DebugConfig.verbose()
    
    async with debug_context(config) as ctx:
        # Emit events during execution
        ctx.emit_node_start("node-1", "LLM", {"prompt": "Hello"})
        
        # ... perform node execution ...
        
        ctx.emit_node_end("node-1", "LLM", {"response": "Hi!"}, duration_ms=150.0)
        
        # Get summary at the end
        summary = ctx.get_summary()
        print(f"Executed {summary.nodes_executed} nodes")
```

### Real-time Event Streaming

```python
from magic_agents.debug import DebugContext, QueueEmitter

async def stream_debug_events():
    emitter = QueueEmitter()
    ctx = DebugContext(
        config=DebugConfig.default(),
        emitter=emitter
    )
    
    async with ctx:
        # Start listening for events in a task
        async def event_listener():
            async for event in emitter.events():
                print(f"[{event.event_type.value}] {event.node_id}: {event.payload}")
        
        listener_task = asyncio.create_task(event_listener())
        
        # ... run graph execution ...
        
        await emitter.close()
        await listener_task
```

### Transform Pipeline

Apply transformations to events before emission:

```python
from magic_agents.debug import (
    TransformPipeline, 
    RedactTransformer, 
    TruncateTransformer,
    FilterTransformer,
    DebugEventType
)

# Build a transformation pipeline
pipeline = TransformPipeline([
    # Filter to only node events
    FilterTransformer(
        include_types={
            DebugEventType.NODE_START,
            DebugEventType.NODE_END,
            DebugEventType.NODE_ERROR
        }
    ),
    # Redact sensitive data
    RedactTransformer(
        patterns=["password", "api_key", "bearer"],
        replacement="[REDACTED]"
    ),
    # Truncate long strings
    TruncateTransformer(max_length=200)
])

# Apply to events
transformed = pipeline.transform(event)
```

### Using DebugCollector for Summaries

```python
from magic_agents.debug import DebugCollector

collector = DebugCollector(execution_id="exec-123")

# Add events during execution
collector.add_event(node_start_event)
collector.add_event(node_end_event)

# Get execution summary
summary = collector.get_summary()
print(f"Total duration: {summary.total_duration_ms}ms")
print(f"Nodes executed: {summary.nodes_executed}")
print(f"Nodes bypassed: {summary.nodes_bypassed}")
print(f"Nodes failed: {summary.nodes_failed}")

# Get per-node summaries
for node_id, node_summary in summary.nodes.items():
    print(f"  {node_id}: {node_summary.duration_ms}ms")

# Convert to legacy format for backward compatibility
legacy = summary.to_legacy_format()
```

