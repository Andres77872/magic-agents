# Exception Handling Improvements

## Overview
All exceptions in the magic_agents system have been converted to use the debug yield mechanism instead of raising errors. This allows for graceful error handling and comprehensive error reporting without interrupting execution flow.

## Changes Summary

### 1. Base Node Class (`magic_agents/node_system/Node.py`)

#### Added `yield_debug_error()` Method
- **Purpose**: Create standardized debug error messages
- **Parameters**:
  - `error_type`: Type of error (e.g., 'ValidationError', 'InputError', 'TemplateError')
  - `error_message`: Detailed error message
  - `context`: Optional dict with additional context
- **Returns**: Dict with type='debug' and error details

#### Modified `__call__()` Method
- Exceptions no longer propagate - they're caught and yielded as debug messages
- Always yields debug info on error (not just in debug mode)
- Execution continues gracefully after error

### 2. Node Classes Updated

#### NodeLLM (`magic_agents/node_system/NodeLLM.py`)
**Errors converted to debug yields:**
- Missing required user message input
- JSON parsing failures (with position info)
- JSON extraction failures (when json_output=True)

**Context provided:**
- Available inputs
- Node configuration (stream, json_output)
- Generated content preview
- Model information
- Error position in JSON

#### NodeConditional (`magic_agents/node_system/NodeConditional.py`)
**Errors converted to debug yields:**
- Invalid configuration (empty condition, invalid merge_strategy)
- Missing inputs
- Jinja2 template errors (undefined variable, syntax error, evaluation error)
- Empty handle results

**Context provided:**
- Condition template
- Available context keys
- Merge strategy
- Template line numbers (for syntax errors)
- Rendered result preview

#### NodeFetch (`magic_agents/node_system/NodeFetch.py`)
**Errors converted to debug yields:**
- URL templating failures
- HTTP errors (with status code)
- Network errors
- Unexpected errors

**Context provided:**
- URL template and rendered URL
- HTTP method
- Status codes
- Response headers
- Exception types

#### NodeLoop (`magic_agents/node_system/NodeLoop.py`)
**Errors converted to debug yields:**
- Missing input list
- JSON parsing errors
- Type validation (expecting list)

**Context provided:**
- Available inputs
- Input value preview
- Error position in JSON
- Received type vs expected type

#### NodeInner (`magic_agents/node_system/NodeInner.py`)
**Errors converted to debug yields:**
- Missing input
- Inner graph not set

**Context provided:**
- Available inputs
- Magic flow configuration
- Inner graph status

#### NodeChat (`magic_agents/node_system/NodeChat.py`)
**Errors converted to debug yields:**
- Mixed image formats (single strings and pairs)

**Context provided:**
- Image input format details
- Validation results

### 3. Graph Execution (`magic_agents/agt_flow.py`)

#### Graph Validation (`validate_graph()`)
**Changed behavior:**
- Returns dict with `{'valid': bool, 'errors': list}` instead of raising
- Validation errors stored in `agt_data['_validation_errors']`
- Errors include detailed context

**Validation errors with debug yields:**
- Missing USER_INPUT node
- Multiple USER_INPUT nodes
- Duplicate edges

#### Graph Execution
**Errors converted to debug yields:**
- Unsupported node types (returns stub node)
- Loop input validation failures
- Circular dependency detection
- Missing nodes in graph

**Context provided:**
- Remaining edges count
- Remaining node IDs
- Executed nodes list
- Node states
- Available node types

#### Validation Error Yielding
- Both `execute_graph()` and `execute_graph_loop()` check for validation errors
- Validation errors yielded as debug messages at start of execution
- Execution continues even with validation errors (nodes handle their own errors)

## Debug Message Format

All debug error messages follow this structure:

```json
{
  "type": "debug",
  "content": {
    "node_id": "node-123",
    "node_type": "LLM",
    "node_class": "NodeLLM",
    "error_type": "InputError",
    "error_message": "Detailed error description...",
    "context": {
      "key1": "value1",
      "key2": "value2"
    },
    "timestamp": "2025-10-10T03:14:29.123456"
  }
}
```

## Benefits

1. **Non-Interrupting**: Errors don't stop execution flow
2. **Detailed Context**: Each error includes comprehensive context information
3. **Debuggable**: All errors are yielded through the same debug mechanism
4. **Traceable**: Timestamps and node information for each error
5. **Consistent**: Standardized error format across all nodes
6. **Graceful Degradation**: System continues execution where possible

## Usage Example

```python
async for result in execute_graph(graph):
    if result['type'] == 'debug':
        error_info = result['content']
        print(f"Error in {error_info['node_id']}: {error_info['error_message']}")
        print(f"Context: {error_info['context']}")
    elif result['type'] == 'content':
        # Handle normal content
        pass
```

## Migration Notes

- All exception handling now uses `yield self.yield_debug_error()` instead of `raise`
- No code should rely on exceptions being raised from node execution
- All validation errors are now yielded as debug messages
- Graph execution continues even when individual nodes fail
