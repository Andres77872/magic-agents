# Conditional Flow Execution & Convergence Handling

This document describes how the Magic-Agents execution engine handles **conditional branching** and **flow convergence** when using conditional nodes (NodeIf, NodeSwitch, NodeConditional).

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Current Implementation](#current-implementation)
3. [Bypass Propagation Algorithm](#bypass-propagation-algorithm)
4. [Convergence Patterns](#convergence-patterns)
5. [Execution Tracking](#execution-tracking)
6. [Best Practices](#best-practices)
7. [Advanced Scenarios](#advanced-scenarios)

---

## Problem Statement

When a conditional node branches execution into multiple paths, several challenges arise:

1. **Path Selection** – Only ONE path should execute based on the condition
2. **Bypass Propagation** – Non-selected paths must be marked as bypassed
3. **Flow Convergence** – When paths merge back together, how to handle:
   - Which branch was taken?
   - How to pass data from the active branch?
   - Should the merge node wait for all branches or just the active one?
4. **State Tracking** – The executor must know which nodes executed vs. bypassed
5. **Dependency Resolution** – Downstream nodes must handle mixed bypass states

### Example Flow Diagram

```
                    ┌──────────────┐
                    │  UserInput   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Conditional  │◄──── Evaluates condition
                    └──┬────────┬──┘
                       │        │
           true ───────┘        └─────── false
                       │        │
              ┌────────▼──┐  ┌──▼─────────┐
              │ ProcessA  │  │ ProcessB   │
              └────────┬──┘  └──┬─────────┘
                       │        │
                       └────┬───┘
                            │
                     ┌──────▼───────┐
                     │    Merge     │◄──── How does this node execute?
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │     END      │
                     └──────────────┘
```

**Key Questions:**
- Should `Merge` wait for both `ProcessA` AND `ProcessB`?
- Or should it execute as soon as ONE path completes?
- How does `Merge` know which path was taken?
- What data does `Merge` receive?

---

## Current Implementation

The Magic-Agents execution engine (`execute_graph` in `agt_flow.py`) implements a **bypass propagation** algorithm to handle conditional branching:

### Node States

Each node can be in one of three states:

```python
node_state: Dict[str, str] = {}
# Possible values:
# - "executed" → node ran successfully
# - "bypassed" → node was skipped due to conditional logic
# - None → not yet processed
```

### Edge Bypass Tracking

```python
bypass_edges: set[str] = set()
# Contains IDs of edges that should not be processed
```

### Core Algorithm Flow

```text
1. Execute conditional node
2. Conditional yields selected handle
3. Mark all non-selected edges as bypassed
4. Propagate bypass to downstream nodes
5. Continue execution, skipping bypassed paths
```

### Code Excerpt from `agt_flow.py` (lines 324-332)

```python
# If the node is a Conditional, decide bypass paths
if isinstance(source_node, NodeConditional):
    produced = set(source_node.outputs.keys()) - {"end"}
    selected_handle = next(iter(produced), None)
    logger.debug("Conditional %s produced handle=%s; bypassing non-selected paths", 
                 edge.source, selected_handle)
    for e in graph.edges:
        if e.source == edge.source and e.sourceHandle != selected_handle:
            mark_edge_bypass(e)
            propagate_bypass(e.target)
```

---

## Bypass Propagation Algorithm

### Algorithm: `propagate_bypass`

**Purpose:** Recursively mark nodes and edges as bypassed if **all** parent paths are bypassed.

```python
def propagate_bypass(node_id: str):
    """Recursively mark node and outgoing edges as bypassed if all parents bypassed."""
    if node_state.get(node_id) == "bypassed":
        return  # Already bypassed
    
    # Get all incoming edges
    incoming = [e for e in graph.edges if e.target == node_id]
    
    # Check if ALL incoming edges are bypassed
    if incoming and all(is_edge_bypassed(e) for e in incoming):
        # Mark this node as bypassed
        node_state[node_id] = "bypassed"
        
        # Mark all outgoing edges as bypassed
        for e in graph.edges:
            if e.source == node_id:
                mark_edge_bypass(e)
                # Recursively propagate to children
                propagate_bypass(e.target)
```

### Key Insight: All-or-Nothing Rule

A node is bypassed **ONLY if ALL** of its parents are bypassed. This handles convergence automatically:

```
Case 1: One parent bypassed, one executed
    → Node EXECUTES (at least one path active)

Case 2: All parents bypassed  
    → Node BYPASSED (no active paths)
```

### Visualization

```
        Conditional
           /    \
      [✓]       [✗]
     PathA     PathB
       |         |
       └────┬────┘
            ↓
      MergeNode ← Has ONE active parent (PathA)
                → EXECUTES ✓
```

```
        Conditional
           /    \
      [✗]       [✗]
     PathA     PathB
       |         |
       └────┬────┘
            ↓
      MergeNode ← ALL parents bypassed
                → BYPASSED ✗
```

---

## Convergence Patterns

### Pattern 1: Simple If-Else Merge

**Scenario:** Two branches converge to a single node

```json
{
  "nodes": [
    {"id": "cond", "type": "conditional"},
    {"id": "pathA", "type": "llm"},
    {"id": "pathB", "type": "llm"},
    {"id": "merge", "type": "parser"}
  ],
  "edges": [
    {"source": "cond", "sourceHandle": "true", "target": "pathA"},
    {"source": "cond", "sourceHandle": "false", "target": "pathB"},
    {"source": "pathA", "target": "merge", "targetHandle": "handle_input"},
    {"source": "pathB", "target": "merge", "targetHandle": "handle_input"}
  ]
}
```

**Execution:**
1. `cond` evaluates to `true`
2. `pathA` executes, `pathB` bypassed
3. `merge` receives input from `pathA` only
4. `merge` executes with `handle_input` = output from `pathA`

**Data Flow:**
```
merge.inputs = {
    "handle_input": pathA.outputs["handle_generated_end"]
}
```

### Pattern 2: Multiple Branches (Switch/Case)

```json
{
  "nodes": [
    {"id": "switch", "type": "conditional"},
    {"id": "case_success", "type": "text"},
    {"id": "case_error", "type": "text"},
    {"id": "case_timeout", "type": "text"},
    {"id": "merge", "type": "parser"}
  ],
  "edges": [
    {"source": "switch", "sourceHandle": "success", "target": "case_success"},
    {"source": "switch", "sourceHandle": "error", "target": "case_error"},
    {"source": "switch", "sourceHandle": "timeout", "target": "case_timeout"},
    {"source": "case_success", "target": "merge", "targetHandle": "success_msg"},
    {"source": "case_error", "target": "merge", "targetHandle": "error_msg"},
    {"source": "case_timeout", "target": "merge", "targetHandle": "timeout_msg"}
  ]
}
```

**Execution (when `switch` selects "error"):**
1. `case_success` → bypassed
2. `case_error` → executed
3. `case_timeout` → bypassed
4. `merge` receives input ONLY on `error_msg` handle
5. `merge` executes with partial inputs

**Data Flow:**
```
merge.inputs = {
    "error_msg": case_error.outputs["handle_void"]
    # Note: success_msg and timeout_msg are NOT set
}
```

⚠️ **Important:** The merge node must handle **partial inputs**. Use template defaults:

```jinja2
{{ success_msg | default("") }}
{{ error_msg | default("") }}
{{ timeout_msg | default("") }}
```

### Pattern 3: Nested Conditionals

```
       Conditional1
         /     \
    [true]   [false]
       |         |
   Conditional2  PathC
     /    \      |
  PathA  PathB   |
     |     |     |
     └──┬──┴─────┘
        ↓
      Merge
```

**Bypass Logic:**
- If Conditional1 = true:
  - Conditional2 executes → one of PathA/PathB executes
  - PathC bypassed
- If Conditional1 = false:
  - PathC executes
  - Conditional2, PathA, PathB all bypassed

**Merge Behavior:**
- Receives input from exactly ONE path (whichever executed)
- All other inputs remain unset

---

## Execution Tracking

### Tracking Which Branch Executed

To know which branch was taken at a merge point, use **handle-specific inputs**:

```json
{
  "id": "merge",
  "type": "parser",
  "data": {
    "text": "{% if handle_true %}Adult path taken: {{ handle_true }}{% else %}Minor path taken: {{ handle_false }}{% endif %}"
  }
}
```

### Conditional Metadata

The conditional node yields metadata about its decision:

```python
yield self.yield_static({"selected": selected_handle})
```

Downstream nodes can access this via the conditional's `end` handle:

```json
{
  "edges": [
    {"source": "cond", "sourceHandle": "end", "target": "logger", "targetHandle": "metadata"}
  ]
}
```

### Debug Logging

Enable debug mode to track execution:

```python
graph = build(spec, message="test", debug=True)
```

Logs will show:
```
DEBUG Conditional node_123 produced handle=true; bypassing non-selected paths
DEBUG Node node_456 bypassed (all parents bypassed)
DEBUG Executing node node_789 (NodeParser)
```

---

## Best Practices

### 1. Design Convergence Points Carefully

**❌ Bad: Expecting all inputs**
```jinja2
{# Will fail if only one branch executed #}
{{ success_result }} and {{ error_result }}
```

**✅ Good: Handle partial inputs**
```jinja2
{% if success_result %}
  Success: {{ success_result }}
{% elif error_result %}
  Error: {{ error_result }}
{% else %}
  No result
{% endif %}
```

### 2. Use Different Target Handles for Each Branch

```json
{
  "edges": [
    {"source": "pathA", "target": "merge", "targetHandle": "result_a"},
    {"source": "pathB", "target": "merge", "targetHandle": "result_b"}
  ]
}
```

This makes it easy to determine which branch executed:
```jinja2
{% if result_a %}Branch A{% elif result_b %}Branch B{% endif %}
```

### 3. Avoid Cycles with Conditionals

**❌ Bad: Conditional in a cycle**
```
NodeA → Conditional → NodeB → NodeA (cycle!)
```

This can cause infinite loops or deadlocks. Use `NodeLoop` for iteration instead.

### 4. Document Branch Behavior

Add comments in your flow specification:
```json
{
  "id": "age_check",
  "type": "conditional",
  "data": {
    "condition": "{{ 'adult' if age >= 18 else 'minor' }}",
    "_comment": "Routes to adult_handler or minor_handler. Merge expects exactly one input."
  }
}
```

### 5. Test All Branches

Create test cases for each possible branch:
```python
# Test true branch
result_true = await run_agent(build(spec, message="age=25"))

# Test false branch  
result_false = await run_agent(build(spec, message="age=15"))
```

---

## Advanced Scenarios

### Scenario 1: Conditional Skip (Early Exit)

**Use case:** Skip expensive processing if condition not met

```
UserInput → Conditional → [false] → END
                |
              [true]
                ↓
           ExpensiveNode → END
```

**Implementation:**
```json
{
  "edges": [
    {"source": "cond", "sourceHandle": "false", "target": "end_node"},
    {"source": "cond", "sourceHandle": "true", "target": "expensive"}
  ]
}
```

### Scenario 2: Parallel Execution with Conditional Gate

**Use case:** Run multiple operations, then conditionally aggregate

```
         ┌→ ProcessA ─┐
UserInput┤            ├→ Conditional → [pass] → Aggregate
         └→ ProcessB ─┘              → [fail] → Error
```

**Key:** Conditional has multiple inputs to evaluate

```json
{
  "id": "gate",
  "type": "conditional",
  "data": {
    "condition": "{{ 'pass' if (result_a and result_b) else 'fail' }}"
  }
}
```

### Scenario 3: Retry Logic with Conditional

**Use case:** Retry operation if it fails (limited iterations)

```
Fetch → Conditional → [success] → Continue
           ↑ |                      
           | [retry]
           └───┘
```

**Implementation:** Use NodeLoop with conditional inside:
```json
{
  "nodes": [
    {"id": "loop", "type": "loop"},
    {"id": "attempt", "type": "fetch"},
    {"id": "check", "type": "conditional"}
  ]
}
```

### Scenario 4: Multi-Level Decision Tree

**Use case:** Complex branching logic (A→B→C or A→D)

```
      A
     /  \
    B    D
   /  \
  C    E
```

**Best Practice:** Use nested conditionals sparingly. Consider:
1. Flattening into switch/case with combined conditions
2. Using computed variables to simplify logic
3. Breaking into separate sub-flows (NodeInner)

---

## Execution Timeline Example

Given this flow:
```
UserInput → Conditional → [true] → PathA → Merge
                       → [false] → PathB → Merge
```

### When condition = true:

| Step | Node | Action | State |
|------|------|--------|-------|
| 1 | UserInput | Execute | executed |
| 2 | Conditional | Evaluate → "true" | executed |
| 3 | false edge | Mark bypass | - |
| 4 | PathB | Propagate bypass | bypassed |
| 5 | PathA | Execute | executed |
| 6 | Merge | Check dependencies → PathA done, PathB bypassed → Execute | executed |

### Dependency Check for Merge:

```python
def are_dependencies_satisfied(node_id: str) -> bool:
    for edge in graph.edges:
        if edge.target == node_id and not is_edge_bypassed(edge):
            # This edge is relevant
            if edge.source not in node_state:
                return False  # Source not yet processed
    return True
```

For Merge node:
- Edge from PathA: `is_edge_bypassed(edge_a)` = False, `edge_a.source` in node_state → ✅
- Edge from PathB: `is_edge_bypassed(edge_b)` = True → Skip this check
- Result: Dependencies satisfied → Execute

---

## Flow Convergence Anti-Patterns

### ❌ Anti-Pattern 1: Assuming All Branches Execute

```jinja2
{# This will fail if branch B bypassed #}
Final result: {{ result_a + result_b }}
```

**Fix:**
```jinja2
Final result: {{ (result_a or 0) + (result_b or 0) }}
```

### ❌ Anti-Pattern 2: Implicit Convergence Without Merge Node

```
PathA ──┐
        ├─→ LLMNode (expects input from A OR B)
PathB ──┘
```

Problem: LLMNode may execute before conditional completes if edges processed out of order.

**Fix:** Add explicit merge/gate node:
```
PathA ──┐
        ├─→ MergeParser → LLMNode
PathB ──┘
```

### ❌ Anti-Pattern 3: Using Same Target Handle for Multiple Branches

```json
{
  "edges": [
    {"source": "pathA", "target": "merge", "targetHandle": "input"},
    {"source": "pathB", "target": "merge", "targetHandle": "input"}
  ]
}
```

Problem: Can't distinguish which branch provided the data.

**Fix:** Use unique handles:
```json
{
  "edges": [
    {"source": "pathA", "target": "merge", "targetHandle": "input_a"},
    {"source": "pathB", "target": "merge", "targetHandle": "input_b"}
  ]
}
```

---

## Testing Conditional Flows

### Unit Test Template

```python
import pytest
from magic_agents.agt_flow import build, run_agent

@pytest.mark.asyncio
async def test_conditional_true_branch():
    spec = {
        "nodes": [...],
        "edges": [...]
    }
    
    graph = build(spec, message='{"age": 25}')
    
    result = []
    async for chunk in run_agent(graph):
        result.append(chunk)
    
    # Assert true branch executed
    assert "adult_path" in str(result)
    assert "minor_path" not in str(result)

@pytest.mark.asyncio
async def test_conditional_false_branch():
    spec = {
        "nodes": [...],
        "edges": [...]
    }
    
    graph = build(spec, message='{"age": 15}')
    
    result = []
    async for run_agent(graph):
        result.append(chunk)
    
    # Assert false branch executed
    assert "minor_path" in str(result)
    assert "adult_path" not in str(result)
```

### Integration Test: Verify Bypass Propagation

```python
def test_bypass_propagation():
    graph = build(spec, message="test")
    
    # Mock conditional to select "true"
    cond_node = graph.nodes["conditional_id"]
    cond_node.outputs = {"true": cond_node.prep({})}
    
    # Execute graph
    await execute_graph(graph)
    
    # Check node states
    assert node_state["true_path"] == "executed"
    assert node_state["false_path"] == "bypassed"
    assert "false_edge_id" in bypass_edges
```

---

## Performance Considerations

### Bypass Propagation Cost

- **Best Case:** O(E) where E = number of edges (all sequential)
- **Worst Case:** O(N×E) where N = nodes (deep nesting with many paths)

### Optimization: Early Bypass Detection

Current implementation checks bypass status multiple times. Optimize by:

```python
# Cache bypass status
@lru_cache(maxsize=1000)
def is_path_active(node_id: str) -> bool:
    """Check if any path to this node is active."""
    if node_state.get(node_id) == "executed":
        return True
    incoming = [e for e in graph.edges if e.target == node_id]
    return any(not is_edge_bypassed(e) for e in incoming)
```

### Memory: Track Only Active Paths

Instead of storing all bypass edges, track only active paths:

```python
active_paths: set[str] = set()  # IDs of edges still processing
```

---

## Summary

### Key Takeaways

1. **Only One Branch Executes** – Conditional nodes activate exactly one output handle
2. **Bypass Propagates** – Non-selected branches are automatically marked as bypassed
3. **All-or-Nothing Rule** – Merge nodes execute if **any** parent is active
4. **Partial Inputs** – Downstream nodes must handle missing data from bypassed branches
5. **Use Unique Handles** – Different target handles for each branch aid debugging

### Execution Flow Checklist

- [ ] Conditional node evaluates condition
- [ ] Selected handle emitted, others bypassed
- [ ] Bypass propagated to downstream nodes
- [ ] Active branch executes normally
- [ ] Bypassed branches skipped
- [ ] Merge point receives input from active branch only
- [ ] Merge node executes with partial inputs
- [ ] Flow continues downstream

### Design Principles

1. **Explicit is better than implicit** – Use unique handle names
2. **Handle missing data gracefully** – Always provide defaults
3. **Test all branches** – Don't assume conditions
4. **Avoid complex nesting** – Flatten when possible
5. **Document convergence points** – Make behavior clear

---

## References

- [NodeConditional Documentation](./NodeConditional.md)
- [Conditional Nodes Approaches](./ConditionalNodesApproaches.md)
- [Architecture Overview](../ARCHITECTURE.md)
- Source: `magic_agents/agt_flow.py` (lines 258-373)
