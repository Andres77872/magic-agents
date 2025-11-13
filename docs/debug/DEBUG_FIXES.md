# Debug System Fixes

## Issues Identified

### 1. Bypassed Nodes Missing Debug Information
**Problem:** When nodes were marked as bypassed (e.g., conditional branches not taken), they had incomplete or missing debug information.

**Impact:** Users couldn't properly analyze conditional flows or understand which nodes were bypassed and why.

**Root Cause:** The `mark_bypassed()` method only set the `was_bypassed` flag without capturing the node's current state (inputs, outputs, internal variables).

### 2. Debug Output Filtering
**Problem:** All nodes in the graph were being added to debug output, including nodes that were never reached during execution.

**Impact:** Debug output included irrelevant nodes with no information, making it harder to analyze actual execution flow.

### 3. Inconsistent Result Types (FIXED)
**Problem:** Debug messages were yielded as plain dictionaries (`{"type": "debug", "content": {...}}`), but regular content was yielded differently. This wasn't clearly documented.

**Impact:** Code that tried to access ChatCompletionModel attributes directly on results would fail.

**Solution:** Now ALL messages are yielded consistently as dictionaries with `type` and `content` fields:
- `{"type": "content", "content": ChatCompletionModel}` - Streaming content
- `{"type": "debug", "content": {...}}` - Per-node debug info  
- `{"type": "debug_summary", "content": {...}}` - Final summary

## Fixes Applied

### Fix 1: Enhanced Bypassed Node Information (`Node.py`)

**File:** `magic_agents/node_system/Node.py`

**Changes:**
```python
def mark_bypassed(self):
    """Mark this node as bypassed (e.g., in conditional flow)."""
    if not self.debug:
        return
        
    # Initialize debug info if not already done
    if self._debug_info is None:
        self._init_debug_info()
    
    # Capture current state before marking as bypassed
    self._debug_info.inputs = self._safe_copy_dict(self.inputs)
    self._debug_info.outputs = self._safe_copy_dict(self.outputs)
    self._debug_info.internal_variables = self._capture_internal_state()
    
    # Mark as bypassed
    self._debug_info.was_bypassed = True
    self._debug_info.was_executed = False
    
    logger.debug(f"Node ({self.node_id}): Marked as bypassed")
```

**Benefits:**
- Bypassed nodes now capture their complete state
- Includes inputs received before bypass
- Captures internal variables at time of bypass
- Properly sets both `was_bypassed` and `was_executed` flags

### Fix 2: Filter Debug Output to Relevant Nodes (`agt_flow.py`)

**File:** `magic_agents/agt_flow.py`

**Changes in both `execute_graph_loop()` and `execute_graph()`:**
```python
# Collect debug information from all nodes if debug mode is enabled
if debug_feedback:
    for node_id, node in nodes.items():
        if hasattr(node, 'get_debug_info'):
            node_debug_info = node.get_debug_info()
            # Only include nodes that were executed or bypassed (part of the execution)
            if node_debug_info and (node_debug_info.was_executed or node_debug_info.was_bypassed):
                debug_feedback.add_node_info(node_debug_info)
```

**Benefits:**
- Debug output only includes nodes that were part of the execution
- Excludes unreached nodes that have no relevant information
- Cleaner, more focused debug data
- Easier to analyze execution flow

### Fix 3: Documentation and Examples

**New Files Created:**

1. **`test/test_debug_mode.py`** - Comprehensive test suite demonstrating:
   - Basic debug mode usage
   - Conditional flow with bypassed nodes
   - Proper result handling (checking for debug vs content messages)
   - Example patterns for processing debug output

2. **`docs/debug/DEBUG_MODE_USAGE.md`** - Complete usage guide covering:
   - How to enable debug mode
   - Correct pattern for handling mixed message types
   - Common mistakes and solutions
   - Practical examples for analysis
   - Troubleshooting guide

3. **`docs/debug/DEBUG_FIXES.md`** - This document

## Migration Guide

### If You're Using Debug Mode

**Before (Incorrect):**
```python
async for result in run_agent(graph):
    # This crashes - assumes result is ChatCompletionModel directly!
    print(result.choices[0].delta.content, end='')
```

**After (Correct):**
```python
async for result in run_agent(graph):
    # Check message type first - all messages are dicts with type/content
    if result.get("type") == "content":
        # Extract ChatCompletionModel from wrapper
        chat_model = result["content"]
        if hasattr(chat_model, 'choices') and chat_model.choices:
            delta = chat_model.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                print(delta.content, end='')
    
    elif result.get("type") == "debug":
        # Per-node debug info
        node_info = result["content"]
        print(f"Node {node_info['node_id']} executed")
    
    elif result.get("type") == "debug_summary":
        # Final summary
        summary = result["content"]
        print(f"Total: {summary['executed_nodes']} nodes")
```

### New Capabilities

1. **Analyze Bypassed Nodes:**
```python
bypassed_nodes = [n for n in debug_info['nodes'] if n['was_bypassed']]
for node in bypassed_nodes:
    print(f"Bypassed: {node['node_id']}")
    print(f"  Had inputs: {list(node['inputs'].keys())}")
    print(f"  Internal state: {node['internal_variables']}")
```

2. **Cleaner Debug Output:**
```python
# Only nodes that were executed or bypassed appear in output
assert all(
    n['was_executed'] or n['was_bypassed'] 
    for n in debug_info['nodes']
)
```

## Testing

Run the new test suite to verify debug functionality:

```bash
pytest test/test_debug_mode.py -v
```

Or run standalone:
```bash
python test/test_debug_mode.py
```

## Performance Considerations

These fixes do not add significant performance overhead:
- Bypassed nodes: Only capture state when actually bypassed (rare operation)
- Filtering: O(n) scan of nodes at end of execution (already doing this)
- Documentation: No runtime impact

Debug mode itself still has overhead (memory for state capture), so continue to disable it in production.

## Backward Compatibility

**Breaking Changes:** None

**Enhanced Behavior:**
- Bypassed nodes now have complete information (previously incomplete)
- Debug output excludes unreached nodes (previously included with null data)
- Better documentation and examples

Existing code that handles debug messages correctly will continue to work. Code that incorrectly assumed all messages have the same structure will need to be updated (but it was already broken).

## Summary

| Issue | Status | Impact |
|-------|--------|--------|
| Bypassed nodes missing data | ✅ Fixed | High - Now capture complete state |
| Unreached nodes in output | ✅ Fixed | Medium - Cleaner debug data |
| Mixed message types unclear | ✅ Documented | High - Clear usage patterns |
| No example tests | ✅ Added | Medium - Easy to get started |

The debug system now provides complete, accurate information about graph execution, making it much easier to:
- Understand conditional flow execution
- Identify performance bottlenecks
- Debug complex graph behaviors
- Analyze which paths were taken or bypassed
