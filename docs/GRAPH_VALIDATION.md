# Graph Validation Documentation

## Overview

The agent flow graph builder now includes comprehensive validation to ensure graph integrity before execution. These validations are performed in the `build()` function via the `validate_graph()` helper.

## Location

- **Module**: `magic_agents/agt_flow.py`
- **Function**: `validate_graph(nodes: list[dict], edges: list[dict])`
- **Called by**: `build()` function (line 445)

## Validations Implemented

### 1. Single User Input Node (Start Node)

**Rule**: The graph MUST contain exactly ONE `NodeUserInput` node (type: `user_input`).

**Rationale**: The `NodeUserInput` is the entry point for the agent flow. Having zero or multiple start nodes would create ambiguity in execution flow.

**Error Messages**:
- **Zero nodes**: `"Graph must contain exactly one USER_INPUT node (start node). Found: 0"`
- **Multiple nodes**: `"Graph must contain exactly one USER_INPUT node (start node). Found N nodes with IDs: [...]"`

**Example Valid Graph**:
```json
{
  "nodes": [
    {"id": "start1", "type": "user_input"},
    {"id": "llm1", "type": "llm"},
    {"id": "end1", "type": "end"}
  ],
  "edges": [...]
}
```

**Example Invalid Graph** (multiple start nodes):
```json
{
  "nodes": [
    {"id": "start1", "type": "user_input"},  // ERROR: Multiple start nodes
    {"id": "start2", "type": "user_input"},  // ERROR: Multiple start nodes
    {"id": "llm1", "type": "llm"}
  ],
  "edges": [...]
}
```

### 2. Multiple End Nodes Allowed

**Rule**: The graph CAN contain multiple `NodeEND` nodes (type: `end`).

**Rationale**: Multiple end nodes allow for branching flows where different execution paths can terminate independently.

**Status**: No validation error for multiple END nodes. This is explicitly allowed and logged.

**Example Valid Graph**:
```json
{
  "nodes": [
    {"id": "start1", "type": "user_input"},
    {"id": "conditional1", "type": "conditional"},
    {"id": "end1", "type": "end"},
    {"id": "end2", "type": "end"}
  ],
  "edges": [
    {"source": "start1", "target": "conditional1"},
    {"source": "conditional1", "target": "end1", "sourceHandle": "true"},
    {"source": "conditional1", "target": "end2", "sourceHandle": "false"}
  ]
}
```

### 3. No Duplicate Edges

**Rule**: No two edges can have identical source, target, sourceHandle, and targetHandle.

**Rationale**: Duplicate edges would create redundant data flow and ambiguous execution behavior.

**Important Note**: Edges between the same two nodes but using different handles are NOT considered duplicates, as handles represent different ports/connections.

**Error Message**: 
```
Found duplicate edges with same source, target, and handles:
  - Edge ID: edge2, Source: node1, Target: node2, Handles: out1 -> in1
```

**Example Valid Graph** (same nodes, different handles):
```json
{
  "edges": [
    {"id": "e1", "source": "n1", "target": "n2", "sourceHandle": "out1", "targetHandle": "in1"},
    {"id": "e2", "source": "n1", "target": "n2", "sourceHandle": "out2", "targetHandle": "in2"}
  ]
}
```

**Example Invalid Graph** (exact duplicate):
```json
{
  "edges": [
    {"id": "e1", "source": "n1", "target": "n2", "sourceHandle": "out1", "targetHandle": "in1"},
    {"id": "e2", "source": "n1", "target": "n2", "sourceHandle": "out1", "targetHandle": "in1"}  // ERROR: Duplicate
  ]
}
```

## Implementation Details

### Validation Flow

1. **Invocation**: `validate_graph()` is called at the beginning of `build()` function
2. **Early Exit**: If validation fails, a `ValueError` is raised before any graph building occurs
3. **Logging**: Validation results are logged at INFO level for monitoring and debugging

### Edge Signature

Edges are uniquely identified by a 4-tuple:
```python
edge_signature = (
    edge.get('source'),
    edge.get('target'),
    edge.get('sourceHandle'),
    edge.get('targetHandle')
)
```

## Testing

Comprehensive test coverage is provided in `test/test_graph_validation.py`:

- ✅ Valid graph with single USER_INPUT
- ✅ Valid graph with multiple END nodes
- ✅ Missing USER_INPUT node (should fail)
- ✅ Multiple USER_INPUT nodes (should fail)
- ✅ Duplicate edges with same source/target (should fail)
- ✅ Multiple duplicate edges detection
- ✅ Different handles NOT considered duplicate (should pass)
- ✅ Exact duplicate edges with handles (should fail)

## Usage Example

```python
from magic_agents.agt_flow import build

# Valid graph data
agt_data = {
    'nodes': [
        {'id': 'user1', 'type': 'user_input', 'data': {}},
        {'id': 'llm1', 'type': 'llm', 'data': {}},
        {'id': 'end1', 'type': 'end'}
    ],
    'edges': [
        {'id': 'e1', 'source': 'user1', 'target': 'llm1'},
        {'id': 'e2', 'source': 'llm1', 'target': 'end1'}
    ]
}

try:
    graph = build(agt_data, message="Hello", load_chat=None)
    # Validation passed, graph built successfully
except ValueError as e:
    # Validation failed
    print(f"Graph validation error: {e}")
```

## Logging Output

When validation succeeds:
```
INFO: Validation passed: Found single USER_INPUT node (id=user1)
INFO: Validation passed: No duplicate edges found (total edges: 2)
INFO: Graph contains 1 END node(s) (multiple END nodes are allowed)
```

When validation fails:
```
ValueError: Graph must contain exactly one USER_INPUT node (start node). Found 2 nodes with IDs: ['user1', 'user2']
```

## Future Enhancements

Potential additional validations to consider:

1. **Unreachable Nodes**: Detect nodes that cannot be reached from the start node
2. **Dangling Edges**: Validate that all edges reference existing nodes
3. **Circular Dependencies**: Enhanced cycle detection for non-loop graphs
4. **Required Node Types**: Ensure certain node combinations are present
5. **Handle Validation**: Verify that referenced handles exist on their respective nodes

## Related Files

- **Implementation**: `magic_agents/agt_flow.py` (lines 377-439)
- **Tests**: `test/test_graph_validation.py`
- **Node Types**: `magic_agents/models/factory/Nodes/BaseNodeModel.py`
- **User Input Node**: `magic_agents/node_system/NodeUserInput.py`
