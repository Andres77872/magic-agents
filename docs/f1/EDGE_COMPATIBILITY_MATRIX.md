# Edge Compatibility Matrix

**Generated**: 2025-10-09  
**Purpose**: Comprehensive guide to edge connections, handle routing, and data flow between nodes in the Magic Agents system.

---

## Table of Contents

1. [Edge Anatomy](#edge-anatomy)
2. [Handle Routing Mechanics](#handle-routing-mechanics)
3. [Node-to-Node Edge Compatibility](#node-to-node-edge-compatibility)
4. [Handle Reference by Node](#handle-reference-by-node)
5. [Required vs Optional Edges](#required-vs-optional-edges)
6. [Common Edge Patterns](#common-edge-patterns)
7. [Edge Troubleshooting](#edge-troubleshooting)
8. [Best Practices](#best-practices)

---

## Edge Anatomy

### Basic Edge Structure

```json
{
  "source": "node-id-1",
  "sourceHandle": "output-handle-name",
  "target": "node-id-2",
  "targetHandle": "input-handle-name"
}
```

### Key Concepts

**sourceHandle ↔ content_type Matching**
- When a node yields data with `content_type='X'`, edges with `sourceHandle="X"` are triggered
- The executor routes events by matching `edge.sourceHandle == event.content_type`

**targetHandle as Input Key**
- The `targetHandle` becomes the key in the target node's `self.inputs` dictionary
- Target nodes retrieve data via `self.get_input('targetHandle_name')`

---

## Handle Routing Mechanics

### Three-Phase Edge Flow

```
Phase 1: SOURCE EMISSION
Source Node yields:
  yield_static(data, content_type='my_handle')
  
Phase 2: EDGE MATCHING
Edge with sourceHandle='my_handle' is triggered
  
Phase 3: TARGET RECEPTION
Target receives data via:
  self.inputs['targetHandle'] = data
```

### Critical Routing Rules

#### Rule 1: sourceHandle MUST match content_type

```python
# Node emits
yield self.yield_static(data, content_type='handle_user_message')

# Edge must use exact match
{"sourceHandle": "handle_user_message"}
```

#### Rule 2: NodeLoop Special Case

NodeLoop defines constants but uses different `content_type` values:

```python
# Implementation uses
yield self.yield_static(item, content_type='content')  # NOT 'handle_item'
yield self.yield_static(agg)  # defaults to 'end', NOT 'handle_end'
```

**Edge Configuration**:
```json
{
  "sourceHandle": "content",   // For items
  "sourceHandle": "default"    // For aggregation
}
```

---

## Node-to-Node Edge Compatibility

### Complete Edge Mapping Table

| Source Node | Source Handles | Target Node | Valid Target Handles | Required | Notes |
|-------------|----------------|-------------|---------------------|----------|-------|
| **NodeUserInput** | `handle_user_message` | NodeChat | `handle_user_message` | No | Primary chat input |
| | | NodeLLM | `handle_user_message` | No | Direct LLM input |
| | | NodeParser | Any custom | No | For templating |
| | | NodeConditional | `handle_input` | No | Branch on user input |
| | | NodeInner | `handle_user_message` | No | Pass to nested flow |
| | `handle_user_files` | NodeChat | `handle_user_files` | No | File attachments |
| | `handle_user_images` | NodeChat | `handle_user_images` | No | Image attachments |
| **NodeText** | `default` | NodeChat | `handle-system-context` | No | System prompt |
| | | NodeLLM | `handle_user_message` | No | Direct prompt |
| | | NodeParser | Any custom | No | Template input |
| | | NodeLoop | `handle_list` | No | Static list data |
| | | NodeConditional | `handle_input` | No | Condition input |
| **NodeClientLLM** | `default` | NodeLLM | `handle-client-provider` | **YES** | LLM client |
| **NodeChat** | `default` | NodeLLM | `handle-chat` | **YES*** | *When using Chat |
| **NodeLLM** | `content` | Any | Any | No | Streaming chunks |
| | `default` | NodeParser | Any custom | No | Post-process |
| | | NodeConditional | `handle_input` | No | Branch on result |
| | | NodeFetch | Any custom | No | Use in API |
| | | NodeLoop | `handle_loop` | No | Aggregate results |
| | | NodeSendMessage | `handle_send_extra` | No | Send with extras |
| | | NodeEND | Any | No | Terminal |
| **NodeParser** | `default` | NodeLLM | `handle_user_message` | No | Templated prompt |
| | | NodeFetch | Any custom | No | API parameters |
| | | NodeConditional | `handle_input` | No | Parsed input |
| | | Any | Any | No | Universal |
| **NodeFetch** | `default` | NodeParser | Any custom | No | Parse response |
| | | NodeLLM | `handle_user_message` | No | Feed to LLM |
| | | NodeConditional | `handle_input_N` | No | Branch on response |
| | | NodeLoop | `handle_list` | No | Iterate items |
| **NodeConditional** | Dynamic | Any | Any | No | Routes by condition |
| **NodeLoop** | `content` | NodeParser | Any custom | No | Process each item |
| | | NodeLLM | `handle_user_message` | No | LLM per item |
| | | NodeFetch | Any custom | No | API per item |
| | `default` | NodeParser | Any custom | No | Aggregated results |
| | | NodeLLM | `handle_user_message` | No | Analyze results |
| | | NodeEND | Any | No | Terminal |
| **NodeInner** | `handle_execution_content` | NodeParser | Any custom | No | Parse results |
| | | NodeLLM | `handle_user_message` | No | Feed to LLM |
| | | NodeSendMessage | `handle_send_extra` | No | Send output |
| | `handle_execution_extras` | NodeParser | Any custom | No | Parse extras |
| **NodeSendMessage** | `content` | Any | Any | No | Streaming chunks |
| | `default` | NodeEND | Any | No | Terminal |

---

## Handle Reference by Node

### Input/Output Handle Catalog

#### NodeUserInput
- **Outputs**: `handle_user_message`, `handle_user_files`, `handle_user_images`
- **Inputs**: None (source node)

#### NodeText
- **Outputs**: `default` (static text)
- **Inputs**: None (source node)

#### NodeClientLLM
- **Outputs**: `default` (MagicLLM instance)
- **Inputs**: None (source node)

#### NodeChat
- **Inputs**: `handle-system-context`, `handle_user_message`, `handle_messages`, `handle_user_files`, `handle_user_images`
- **Outputs**: `default` (ModelChat)

#### NodeLLM
- **Inputs**: `handle-client-provider` **(REQUIRED)**, `handle-chat`, `handle-system-context`, `handle_user_message`
- **Outputs**: `content` (streaming), `default` (final)
- **Flags**: `iterate=true` (for loops), `json_output=true` (parse JSON)

#### NodeParser
- **Inputs**: Any (all used as template variables)
- **Outputs**: `default` (parsed string)
- **Special**: Auto-parses JSON strings

#### NodeFetch
- **Inputs**: Any (for Jinja2 templating, requires at least one)
- **Outputs**: `default` (JSON response)
- **Template**: URL, data, json_data, headers support Jinja2

#### NodeConditional
- **Inputs**: `handle_input` **(REQUIRED)**, `handle_input_1`, `handle_input_2`, ...
- **Outputs**: Dynamic (based on condition), `default` (metadata)
- **Config**: `merge_strategy='flat'` or `'namespaced'`

#### NodeLoop
- **Inputs**: `handle_list` **(REQUIRED)**, `handle_loop`
- **Outputs**: `content` (items), `default` (aggregation)
- **Note**: Uses generic content_type values, not constant names

#### NodeInner
- **Inputs**: `handle_user_message` **(REQUIRED)**
- **Outputs**: `handle_execution_content`, `handle_execution_extras`
- **Config**: `magic_flow` (nested graph)

#### NodeSendMessage
- **Inputs**: `handle_send_extra`
- **Outputs**: `content` (streaming), `default` (final)
- **Config**: `message`, `json_extras`

#### NodeEND
- **Inputs**: Any (ignored)
- **Outputs**: `default` (empty ChatCompletionModel)

---

## Required vs Optional Edges

### Absolutely Required Edges

#### 1. NodeLLM ← NodeClientLLM (CRITICAL)
```json
{
  "source": "client-1",
  "sourceHandle": "default",
  "target": "llm-1",
  "targetHandle": "handle-client-provider"
}
```
**Failure without**: AttributeError or "missing client"

#### 2. NodeLLM ← Chat/UserMessage
At least one required:
```json
// Option A: Via Chat
{"sourceHandle": "default", "target": "llm-1", "targetHandle": "handle-chat"}

// Option B: Direct
{"sourceHandle": "handle_user_message", "target": "llm-1", "targetHandle": "handle_user_message"}
```
**Failure without**: ValueError "requires either a user message"

#### 3. NodeLoop ← List Input
```json
{"sourceHandle": "default", "target": "loop-1", "targetHandle": "handle_list"}
```
**Failure without**: ValueError "expects a list"

#### 4. NodeConditional ← Input
```json
{"sourceHandle": "...", "target": "conditional-1", "targetHandle": "handle_input"}
```
**Failure without**: ValueError "requires at least one input"

#### 5. NodeInner ← User Message
```json
{"sourceHandle": "...", "target": "inner-1", "targetHandle": "handle_user_message"}
```
**Failure without**: ValueError "requires input 'handle_user_message'"

---

## Common Edge Patterns

### Pattern 1: Basic Chat Flow
```json
[
  {"source": "user-1", "sourceHandle": "handle_user_message", "target": "chat-1", "targetHandle": "handle_user_message"},
  {"source": "client-1", "sourceHandle": "default", "target": "llm-1", "targetHandle": "handle-client-provider"},
  {"source": "chat-1", "sourceHandle": "default", "target": "llm-1", "targetHandle": "handle-chat"},
  {"source": "llm-1", "sourceHandle": "default", "target": "end-1", "targetHandle": "default"}
]
```

### Pattern 2: Loop Processing
```json
[
  {"source": "text-1", "sourceHandle": "default", "target": "loop-1", "targetHandle": "handle_list"},
  {"source": "loop-1", "sourceHandle": "content", "target": "llm-1", "targetHandle": "handle_user_message"},
  {"source": "client-1", "sourceHandle": "default", "target": "llm-1", "targetHandle": "handle-client-provider"},
  {"source": "llm-1", "sourceHandle": "default", "target": "loop-1", "targetHandle": "handle_loop"},
  {"source": "loop-1", "sourceHandle": "default", "target": "end-1", "targetHandle": "default"}
]
```
**Critical**: LLM must have `"iterate": true`

### Pattern 3: Conditional Branching
```json
[
  {"source": "user-1", "sourceHandle": "handle_user_message", "target": "conditional-1", "targetHandle": "handle_input"},
  {"source": "conditional-1", "sourceHandle": "adult", "target": "text-adult", "targetHandle": "default"},
  {"source": "conditional-1", "sourceHandle": "minor", "target": "text-minor", "targetHandle": "default"}
]
```
**Note**: Condition must render to "adult" or "minor"

### Pattern 4: Multi-Input Conditional
```json
[
  {"source": "fetch-user", "sourceHandle": "default", "target": "conditional-1", "targetHandle": "handle_input_1"},
  {"source": "fetch-account", "sourceHandle": "default", "target": "conditional-1", "targetHandle": "handle_input_2"}
]
```
**Config**: Use `"merge_strategy": "namespaced"`

### Pattern 5: API Integration
```json
[
  {"source": "user-1", "sourceHandle": "handle_user_message", "target": "parser-1", "targetHandle": "user_query"},
  {"source": "parser-1", "sourceHandle": "default", "target": "fetch-1", "targetHandle": "search_term"},
  {"source": "fetch-1", "sourceHandle": "default", "target": "llm-1", "targetHandle": "handle_user_message"}
]
```

### Pattern 6: Chained LLMs
```json
[
  {"source": "llm-1", "sourceHandle": "default", "target": "parser-1", "targetHandle": "first_result"},
  {"source": "parser-1", "sourceHandle": "default", "target": "llm-2", "targetHandle": "handle_user_message"},
  {"source": "client-2", "sourceHandle": "default", "target": "llm-2", "targetHandle": "handle-client-provider"}
]
```
**Note**: Each LLM needs its own client connection

---

## Edge Troubleshooting

### Error 1: "No data on edge"
**Cause**: sourceHandle doesn't match source content_type  
**Fix**: Check node implementation for actual content_type value

### Error 2: "Missing required input"
**Cause**: targetHandle not connected  
**Fix**: Add required edge (see Required Edges section)

### Error 3: "Template variable undefined"
**Cause**: NodeParser/Fetch template uses variable not provided  
**Fix**: Ensure edge targetHandle matches template variable name

### Error 4: "Loop not iterating"
**Cause**: handle_list not connected or invalid data  
**Fix**: Connect to `handle_list`, ensure source is array

### Error 5: "LLM same output per loop"
**Cause**: `iterate` flag not set  
**Fix**: Set `"iterate": true` in NodeLLM config

### Error 6: "Conditional not routing"
**Cause**: Condition doesn't match output handle  
**Fix**: Ensure condition renders to exact sourceHandle name

### Debugging Checklist
- [ ] sourceHandle matches actual content_type from source
- [ ] targetHandle matches expected input key in target
- [ ] Required edges present (ClientLLM → LLM, etc.)
- [ ] Data type compatibility (string/JSON/list)
- [ ] Special flags set (iterate, json_output)
- [ ] Enable debug mode for detailed logs

---

## Best Practices

### 1. Use Descriptive targetHandle Names
✅ `"targetHandle": "user_query"` not `"input1"`  
Makes templates readable: `{{ user_query }}`

### 2. Verify content_type in Implementation
Don't rely on constant names alone. Check actual `yield_static()` calls.

### 3. Document Template Variables
For NodeParser/Fetch, document which targetHandles map to template variables.

### 4. Use Namespaced Merge for Multi-Input
```json
{"data": {"merge_strategy": "namespaced"}}
```
Access: `{{ handle_input_1.field }}`

### 5. Validate Required Edges
Before execution, verify:
- All NodeLLM have client connections
- All loops have list inputs
- All conditionals have input connections

### 6. Handle Edge Cases
- Empty lists → NodeLoop
- Missing API responses → NodeFetch
- Invalid JSON → NodeLLM json_output
- Undefined template variables → NodeParser

### 7. Test Conditional Templates
Verify condition renders to valid handle names before deployment.

### 8. Use Consistent Naming
- Input handles: `handle-kebab-case`
- Output handles: `handle_snake_case`
- Custom handles: descriptive names

---

## Quick Reference

### Most Common Edges

```json
// User → Chat
{"source": "user", "sourceHandle": "handle_user_message", "target": "chat", "targetHandle": "handle_user_message"}

// Client → LLM (REQUIRED)
{"source": "client", "sourceHandle": "default", "target": "llm", "targetHandle": "handle-client-provider"}

// Chat → LLM
{"source": "chat", "sourceHandle": "default", "target": "llm", "targetHandle": "handle-chat"}

// LLM → Parser
{"source": "llm", "sourceHandle": "default", "target": "parser", "targetHandle": "llm_output"}

// Loop items
{"source": "loop", "sourceHandle": "content", "target": "processor", "targetHandle": "item"}

// Loop feedback
{"source": "processor", "sourceHandle": "default", "target": "loop", "targetHandle": "handle_loop"}

// Loop aggregation
{"source": "loop", "sourceHandle": "default", "target": "next", "targetHandle": "results"}

// Conditional branch
{"source": "conditional", "sourceHandle": "branch_name", "target": "handler", "targetHandle": "default"}

// Any → END
{"source": "any", "sourceHandle": "default", "target": "end", "targetHandle": "default"}
```

---

*Generated from magic-agents node system analysis*
