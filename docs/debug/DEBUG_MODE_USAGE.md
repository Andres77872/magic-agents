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
