# Debug Mode Examples

This document provides practical examples of using debug mode in Magic Agents.

## Example 1: Basic Debug Mode Usage

### Graph Definition

```json
{
  "type": "chat",
  "debug": true,
  "nodes": [
    {
      "id": "user-input-1",
      "type": "USER_INPUT",
      "data": {}
    },
    {
      "id": "llm-1",
      "type": "LLM",
      "data": {
        "stream": true,
        "json_output": false,
        "temperature": 0.7
      }
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "source": "user-input-1",
      "target": "llm-1",
      "sourceHandle": "handle_user_message",
      "targetHandle": "handle_user_message"
    },
    {
      "id": "edge-2",
      "source": "llm-1",
      "target": "end-1",
      "sourceHandle": "end",
      "targetHandle": "input"
    }
  ]
}
```

### Python Code

```python
from magic_agents import build, run_agent

# Build the graph with debug enabled
graph = build(graph_data, message="What is machine learning?", load_chat=load_chat_fn)
# Execute and collect results
content_chunks = []
node_debug_info = []
debug_summary = None

async for result in run_agent(graph=graph):
    if result.get("type") == "content":
        # Regular content chunks (ChatCompletionModel)
        chat_model = result["content"]
        content_chunks.append(chat_model)
    elif result.get("type") == "debug":
        # Per-node debug feedback (yielded immediately after each node execution)
        node_info = result["content"]
        node_debug_info.append(node_info)
        print(f"[DEBUG] Node {node_info['node_id']} completed in {node_info['execution_duration_ms']:.2f}ms")
{{ ... }}
    elif result.get("type") == "debug_summary":
        # Final summary debug feedback (yielded at the end)
        debug_summary = result["content"]

# Analyze final debug summary
print(f"\n=== Execution Summary ===")
print(f"Execution ID: {debug_summary['execution_id']}")
print(f"Total duration: {debug_summary['total_duration_ms']:.2f}ms")
print(f"Nodes executed: {debug_summary['executed_nodes']}/{debug_summary['total_nodes']}")

# Inspect individual nodes from real-time debug messages
print(f"\n=== Per-Node Details ===")
for node in node_debug_info:
    print(f"\nNode: {node['node_id']} ({node['node_class']})")
    print(f"  Duration: {node['execution_duration_ms']:.2f}ms")
    print(f"  Inputs: {list(node['inputs'].keys())}")
    print(f"  Outputs: {list(node['outputs'].keys())}")
    if node['error']:
        print(f"  ERROR: {node['error']}")
```

### Expected Output

```
[DEBUG] Node user-input-1 completed in 0.50ms
[DEBUG] Node llm-1 completed in 1520.20ms
[DEBUG] Node end-1 completed in 0.10ms

=== Execution Summary ===
Execution ID: a1b2c3d4e5f6
Total duration: 1523.45ms
Nodes executed: 3/3

=== Per-Node Details ===

Node: user-input-1 (NodeUserInput)
  Duration: 0.50ms
  Inputs: []
  Outputs: ['handle_user_message']

Node: llm-1 (NodeLLM)
  Duration: 1520.20ms
  Inputs: ['handle_user_message', 'handle-client-provider']
  Outputs: ['end']

Node: end-1 (NodeEND)
  Duration: 0.10ms
  Inputs: ['input']
  Outputs: []
```

**Note**: The `[DEBUG]` messages appear in real-time as each node completes, allowing you to monitor execution progress. The summary appears at the end with aggregate statistics.

---

## Example 2: Debugging Conditional Flow

### Graph Definition

```json
{
  "type": "chat",
  "debug": true,
  "nodes": [
    {
      "id": "user-input-1",
      "type": "USER_INPUT"
    },
    {
      "id": "conditional-1",
      "type": "CONDITIONAL",
      "data": {
        "condition": "len(input_text) > 100"
      }
    },
    {
      "id": "llm-long",
      "type": "LLM",
      "data": {"temperature": 0.3}
    },
    {
      "id": "llm-short",
      "type": "LLM",
      "data": {"temperature": 0.7}
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "user-input-1",
      "target": "conditional-1",
      "sourceHandle": "handle_user_message",
      "targetHandle": "input_text"
    },
    {
      "source": "conditional-1",
      "target": "llm-long",
      "sourceHandle": "handle_true",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "conditional-1",
      "target": "llm-short",
      "sourceHandle": "handle_false",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "llm-long",
      "target": "end-1"
    },
    {
      "source": "llm-short",
      "target": "end-1"
    }
  ]
}
```

### Analyzing Debug Output

```python
async for result in run_agent(graph):
    if result.get("type") == "debug_summary":
        debug_info = result["content"]
        
        # Check which branch was taken
        conditional_node = next(
            n for n in debug_info['nodes'] 
            if n['node_class'] == 'NodeConditional'
        )
        
        branch_taken = list(conditional_node['outputs'].keys())[0]
        print(f"Conditional evaluation: {branch_taken}")
        
        # Check which LLM was executed and which was bypassed
        for node in debug_info['nodes']:
            if node['node_class'] == 'NodeLLM':
                status = "EXECUTED" if node['was_executed'] else "BYPASSED"
                print(f"{node['node_id']}: {status}")
```

### Expected Output (short input)

```
Conditional evaluation: handle_false
llm-long: BYPASSED
llm-short: EXECUTED
```

---

## Example 3: Debugging Loop Execution

### Graph Definition

```json
{
  "type": "chat",
  "debug": true,
  "nodes": [
    {
      "id": "user-input-1",
      "type": "USER_INPUT"
    },
    {
      "id": "parser-1",
      "type": "PARSER",
      "data": {
        "parse_json": true
      }
    },
    {
      "id": "loop-1",
      "type": "LOOP"
    },
    {
      "id": "llm-1",
      "type": "LLM",
      "data": {
        "iterate": true,
        "stream": false
      }
    },
    {
      "id": "end-1",
      "type": "END"
    }
  ],
  "edges": [
    {
      "source": "user-input-1",
      "target": "parser-1",
      "sourceHandle": "handle_user_message",
      "targetHandle": "input"
    },
    {
      "source": "parser-1",
      "target": "loop-1",
      "sourceHandle": "end",
      "targetHandle": "handle_list"
    },
    {
      "source": "loop-1",
      "target": "llm-1",
      "sourceHandle": "handle_item",
      "targetHandle": "handle_user_message"
    },
    {
      "source": "llm-1",
      "target": "loop-1",
      "sourceHandle": "end",
      "targetHandle": "handle_loop"
    },
    {
      "source": "loop-1",
      "target": "end-1",
      "sourceHandle": "handle_end",
      "targetHandle": "input"
    }
  ]
}
```

### Analyzing Loop Debug Output

```python
async for result in run_agent(graph):
    if result.get("type") == "debug_summary":
        debug_info = result["content"]
        
        # Find the loop node
        loop_node = next(
            n for n in debug_info['nodes'] 
            if n['node_class'] == 'NodeLoop'
        )
        
        # Get the list of items that were processed
        items_list = loop_node['inputs'].get('handle_list', [])
        results = loop_node['outputs'].get('handle_end', {}).get('content', [])
        
        print(f"Loop processed {len(items_list)} items")
        print(f"Execution time: {loop_node['execution_duration_ms']}ms")
        
        # Check if LLM was executed multiple times (once per item)
        llm_node = next(
            n for n in debug_info['nodes'] 
            if n['node_class'] == 'NodeLLM'
        )
        print(f"LLM node duration: {llm_node['execution_duration_ms']}ms")
        print(f"Iterate enabled: {llm_node['internal_variables']['iterate']}")
```

---

## Example 4: Performance Analysis

### Identifying Bottlenecks

```python
async for result in run_agent(graph):
    if result.get("type") == "debug_summary":
        debug_info = result["content"]
        
        # Sort nodes by execution time
        executed_nodes = [
            n for n in debug_info['nodes'] 
            if n['was_executed'] and n['execution_duration_ms']
        ]
        
        sorted_nodes = sorted(
            executed_nodes, 
            key=lambda n: n['execution_duration_ms'], 
            reverse=True
        )
        
        print("Performance Analysis:")
        print("-" * 60)
        total_time = debug_info['total_duration_ms']
        
        for node in sorted_nodes:
            duration = node['execution_duration_ms']
            percentage = (duration / total_time) * 100
            print(f"{node['node_id']:20} {node['node_class']:15} "
                  f"{duration:8.2f}ms ({percentage:5.1f}%)")
```

### Expected Output

```
Performance Analysis:
------------------------------------------------------------
llm-1                NodeLLM          1520.23ms ( 95.2%)
fetch-1              NodeFetch          45.67ms (  2.9%)
parser-1             NodeParser         15.23ms (  1.0%)
user-input-1         NodeUserInput       0.50ms (  0.0%)
text-1               NodeText            0.30ms (  0.0%)
end-1                NodeEND             0.10ms (  0.0%)
```

---

## Example 5: Error Debugging

### Handling Failed Nodes

```python
async for result in run_agent(graph):
    if result.get("type") == "debug_summary":
        debug_info = result["content"]
        
        # Check for failed nodes
        failed_nodes = [
            n for n in debug_info['nodes'] 
            if n['error'] is not None
        ]
        
        if failed_nodes:
            print(f"Found {len(failed_nodes)} failed node(s):")
            for node in failed_nodes:
                print(f"\nNode: {node['node_id']} ({node['node_class']})")
                print(f"Error: {node['error']}")
                print(f"Inputs: {node['inputs']}")
                print(f"Internal state: {node['internal_variables']}")
        else:
            print("All nodes executed successfully")
```

---

## Example 6: Data Flow Tracing

### Tracing Data Through the Graph

```python
async for result in run_agent(graph):
    if result.get("type") == "debug_summary":
        debug_info = result["content"]
        
        print("Data Flow Trace:")
        print("=" * 60)
        
        # Build a map of node executions in order
        executed = [n for n in debug_info['nodes'] if n['was_executed']]
        
        for i, node in enumerate(executed, 1):
            print(f"\n{i}. {node['node_id']} ({node['node_class']})")
            print(f"   Duration: {node['execution_duration_ms']}ms")
            
            # Show inputs
            if node['inputs']:
                print("   Inputs:")
                for handle, value in node['inputs'].items():
                    value_str = str(value)[:50]
                    print(f"     {handle}: {value_str}...")
            
            # Show outputs
            if node['outputs']:
                print("   Outputs:")
                for handle, value in node['outputs'].items():
                    content = value.get('content', value)
                    content_str = str(content)[:50]
                    print(f"     {handle}: {content_str}...")
        
        # Show edge processing
        print("\n" + "=" * 60)
        print("Edge Processing Order:")
        for i, edge in enumerate(debug_info['edges_processed'], 1):
            print(f"{i}. {edge['source']}[{edge['source_handle']}] -> "
                  f"{edge['target']}[{edge['target_handle']}]")
```

---

## Tips for Using Debug Mode

### 1. Selective Debugging

Only enable debug mode when needed:

```python
# Enable debug mode conditionally
graph_data['debug'] = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
```

### 2. Save Debug Output to File

```python
import json

node_debug_logs = []

async for result in run_agent(graph):
    if result.get("type") == "debug":
        # Save each node's debug info immediately (per-node)
        node_debug_logs.append(result["content"])
        with open(f'debug_node_{result["content"]["node_id"]}.json', 'w') as f:
            json.dump(result["content"], f, indent=2)
    elif result.get("type") == "debug_summary":
        # Save final summary
        with open('debug_summary.json', 'w') as f:
            json.dump(result["content"], f, indent=2)
```

### 3. Compare Executions

```python
# Save multiple debug runs to compare
debug_runs = []

for test_input in test_cases:
    graph = build(graph_data, message=test_input)
    node_info = []
    
    async for result in run_agent(graph):
        if result.get("type") == "debug":
            # Per-node debug info
            node_info.append(result["content"])
        elif result.get("type") == "debug_summary":
            # Final summary
            debug_runs.append({
                'input': test_input,
                'nodes': node_info,
                'summary': result["content"]
            })

# Compare execution times, paths taken, etc.
for run in debug_runs:
    print(f"Input: {run['input']}")
    print(f"  Total time: {run['summary']['total_duration_ms']:.2f}ms")
    print(f"  Nodes executed: {run['summary']['executed_nodes']}")
```

### 4. Custom Debug Analysis

```python
def analyze_debug_info(debug_info):
    """Custom function to analyze debug information."""
    analysis = {
        'total_time': debug_info['total_duration_ms'],
        'node_count': debug_info['total_nodes'],
        'bottleneck': max(
            debug_info['nodes'],
            key=lambda n: n.get('execution_duration_ms', 0)
        )['node_id'],
        'execution_path': [
            n['node_id'] for n in debug_info['nodes'] 
            if n['was_executed']
        ]
    }
    return analysis
```

---

## Example 3: Error Handling with Debug Mode

Debug mode provides immediate feedback when a node fails, allowing you to diagnose errors quickly.

### Python Code

```python
from magic_agents import build, run_agent

graph = build(graph_data, message="Test message", load_chat=load_chat_fn)

node_debug_info = []
error_occurred = False

try:
    async for result in run_agent(graph):
        if result.get("type") == "content":
            # Extract ChatCompletionModel and print
            chat_model = result["content"]
            if hasattr(chat_model, 'choices') and chat_model.choices:
                delta = chat_model.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    print(delta.content, end="", flush=True)
        elif result.get("type") == "debug":
            # Per-node debug feedback (includes errors)
            node_info = result["content"]
            node_debug_info.append(node_info)
            
            # Check for errors in this node
            if node_info.get('error'):
                error_occurred = True
                print(f"\n[ERROR] Node {node_info['node_id']} failed!")
                print(f"  Error: {node_info['error']}")
                print(f"  Duration before failure: {node_info['execution_duration_ms']:.2f}ms")
                print(f"  Inputs: {node_info['inputs']}")
                print(f"  Outputs: {node_info['outputs']}")
        elif result.get("type") == "debug_summary":
            debug_summary = result["content"]
            print(f"\n\nExecution completed with {debug_summary['failed_nodes']} failed nodes")
            
except Exception as e:
    print(f"\n\nExecution terminated with exception: {e}")
    
    # Review debug info collected before failure
    print(f"\n=== Debug Info Collected Before Failure ===")
    for node in node_debug_info:
        status = "FAILED" if node.get('error') else "SUCCESS"
        print(f"  {node['node_id']}: {status} ({node['execution_duration_ms']:.2f}ms)")
```

### Expected Output (When Error Occurs)

```
[ERROR] Node llm-1 failed!
  Error: API rate limit exceeded
  Duration before failure: 245.30ms
  Inputs: {'handle_user_message': 'Test message', 'handle-client-provider': '<MagicLLM>'}
  Outputs: {}

Execution terminated with exception: API rate limit exceeded

=== Debug Info Collected Before Failure ===
  user-input-1: SUCCESS (0.50ms)
  llm-1: FAILED (245.30ms)
```

---

## Example 7: Using the New Debug System Programmatically

The new debug architecture provides more control through the `magic_agents.debug` module.

### Basic Usage with DebugContext

```python
import asyncio
from magic_agents.debug import (
    DebugContext,
    DebugConfig,
    DebugCollector,
    QueueEmitter,
    DebugEventType
)

async def run_with_programmatic_debug():
    # Configure debug settings
    config = DebugConfig(
        enabled=True,
        capture_inputs=True,
        capture_outputs=True,
        capture_internal_state=True,
        max_string_length=500,
        redact_patterns=["password", "api_key"]
    )
    
    # Create debug context
    async with DebugContext(config=config) as ctx:
        # Emit events during execution
        ctx.emit_graph_start(graph_id="my-graph", graph_type="chat")
        
        ctx.emit_node_start(
            node_id="llm-1",
            node_type="LLM",
            inputs={"prompt": "Hello, world!"}
        )
        
        # Simulate node execution...
        await asyncio.sleep(0.1)
        
        ctx.emit_node_end(
            node_id="llm-1",
            node_type="LLM",
            outputs={"response": "Hi there!"},
            duration_ms=100.0
        )
        
        ctx.emit_graph_end(duration_ms=150.0)
        
        # Get execution summary
        summary = ctx.get_summary()
        print(f"Total duration: {summary.total_duration_ms}ms")
        print(f"Nodes executed: {summary.nodes_executed}")

asyncio.run(run_with_programmatic_debug())
```

### Real-Time Event Streaming

```python
import asyncio
from magic_agents.debug import (
    DebugContext,
    DebugConfig,
    QueueEmitter
)

async def stream_events():
    emitter = QueueEmitter()
    config = DebugConfig.verbose()
    
    async with DebugContext(config=config, emitter=emitter) as ctx:
        # Start event listener task
        async def listen():
            async for event in emitter.events():
                print(f"[{event.timestamp}] {event.event_type.value}: "
                      f"{event.node_id or 'graph'}")
                if event.payload:
                    print(f"  Payload: {event.payload}")
        
        listener = asyncio.create_task(listen())
        
        # Emit some events
        ctx.emit_graph_start(graph_id="test", graph_type="chat")
        ctx.emit_node_start("node-1", "TEXT", {"value": "Hello"})
        await asyncio.sleep(0.01)
        ctx.emit_node_end("node-1", "TEXT", {"result": "Hello!"}, duration_ms=10.0)
        ctx.emit_graph_end(duration_ms=50.0)
        
        # Close emitter and wait for listener
        await emitter.close()
        await listener

asyncio.run(stream_events())
```

### Using Transform Pipeline

```python
from magic_agents.debug import (
    TransformPipeline,
    RedactTransformer,
    FilterTransformer,
    TruncateTransformer,
    DebugEventType,
    node_start_event
)

# Create a pipeline
pipeline = TransformPipeline([
    # Only keep node events
    FilterTransformer(include_types={
        DebugEventType.NODE_START,
        DebugEventType.NODE_END,
        DebugEventType.NODE_ERROR
    }),
    # Redact sensitive data
    RedactTransformer(
        patterns=["password", "secret", "token", "api_key"],
        replacement="***REDACTED***"
    ),
    # Truncate long strings
    TruncateTransformer(max_length=100)
])

# Create an event
event = node_start_event(
    execution_id="exec-123",
    node_id="auth-node",
    node_type="AUTH",
    inputs={"password": "super_secret_123", "username": "user@example.com"}
)

# Transform the event
transformed = pipeline.transform(event)
print(transformed.payload)
# Output: {'inputs': {'password': '***REDACTED***', 'username': 'user@example.com'}}
```

### Custom Emitter

```python
from magic_agents.debug import Emitter, DebugEvent
import json

class FileEmitter(Emitter):
    """Custom emitter that writes events to a file."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, 'w')
    
    async def emit(self, event: DebugEvent) -> None:
        line = json.dumps({
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'node_id': event.node_id,
            'payload': event.payload
        })
        self.file.write(line + '\n')
        self.file.flush()
    
    async def close(self) -> None:
        self.file.close()

# Usage
emitter = FileEmitter('debug_events.jsonl')
# ... use with DebugContext ...
```

### Configuration Presets

```python
from magic_agents.debug import DebugConfig

# Development: Full capture
dev_config = DebugConfig.default()

# Staging: Minimal overhead
staging_config = DebugConfig.minimal()

# Troubleshooting: Maximum detail
debug_config = DebugConfig.verbose()

# Production: Only capture errors
prod_config = DebugConfig.errors_only()

# Custom: Mix and match
custom_config = DebugConfig(
    enabled=True,
    capture_inputs=True,
    capture_outputs=True,
    capture_internal_state=False,  # Skip internal vars
    max_string_length=200,
    redact_patterns=["password", "token", "secret"],
    include_event_types=[
        DebugEventType.NODE_START,
        DebugEventType.NODE_END,
        DebugEventType.NODE_ERROR,
        DebugEventType.GRAPH_END
    ]
)
```

### Converting to Legacy Format

```python
from magic_agents.debug import DebugCollector, node_start_event, node_end_event

# Collect events
collector = DebugCollector(execution_id="exec-456")

collector.add_event(node_start_event(
    execution_id="exec-456",
    node_id="text-1",
    node_type="TEXT",
    inputs={"value": "Hello"}
))

collector.add_event(node_end_event(
    execution_id="exec-456",
    node_id="text-1",
    node_type="TEXT",
    outputs={"result": "Hello, World!"},
    duration_ms=5.0
))

# Get summary in new format
summary = collector.get_summary()

# Convert to legacy GraphDebugFeedback format
legacy = summary.to_legacy_format()
print(json.dumps(legacy, indent=2))
```

---

## See Also

- [Debug Mode Overview](./DEBUG_MODE_OVERVIEW.md)
- [Debug Feedback Structure](./DEBUG_FEEDBACK_STRUCTURE.md)
- [Node Debug Information](./NODE_DEBUG_INFORMATION.md)

**Key Benefits:**
- **Immediate error notification**: Debug info is yielded as soon as the error occurs
- **Pre-exception debugging**: Get debug info before the exception is raised
- **Partial execution tracking**: See which nodes succeeded before the failure
- **Complete context**: Access inputs, outputs, and internal state at the point of failure
