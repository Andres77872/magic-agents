# Debug Mode Documentation

Welcome to the Magic Agents Debug Mode documentation.

## What is Debug Mode?

Debug mode is a comprehensive execution tracking system that captures detailed information about node execution, data flow, and performance metrics during graph execution.

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

- **Comprehensive Tracking**: Inputs, outputs, and internal state for every node
- **Timing Information**: Precise execution timing at millisecond level
- **Error Capture**: Detailed error messages when nodes fail
- **Conditional Flow Tracking**: See which branches were taken
- **Loop Iteration Details**: Track each iteration
- **Safe Serialization**: JSON-compatible output

## Important Notes

**Performance**: Debug mode adds overhead. Disable in production unless troubleshooting.

**Privacy**: Debug mode captures all inputs/outputs including sensitive data.

**Output Size**: Debug feedback can be large for complex graphs.
