# Node Compatibility Matrix

## Overview

This document provides a comprehensive analysis of node compatibility within the Magic Agents node system. Each node type has specific input and output handles that determine which connections are valid.

## Node System Architecture: Handle Routing Explained

### The Event-Driven Model

magic-agents uses an **event-driven routing system** where:
1. Nodes emit **typed events** when processing
2. The executor **routes events** based on type matching
3. Target nodes **receive data** via named handles

### Three-Step Routing Process

#### Step 1: Node Emits Event

When a node completes processing, it yields an event with a specific type:

```python
# In NodeUserInput.py
yield self.yield_static(self._text, content_type='handle_user_message')

# This creates an event:
{
    'type': 'handle_user_message',  # ← This is the routing key
    'content': {'node': 'NodeUserInput', 'content': 'user text'}
}
```

#### Step 2: Executor Matches Edge

The executor finds edges where `sourceHandle` matches the event type:

```json
{
    "source": "user-1",
    "sourceHandle": "handle_user_message",  // ← Must match event type
    "target": "chat-1",
    "targetHandle": "handle_user_message"   // ← Key for target's inputs
}
```

The executor logic (simplified):
```python
if edge.sourceHandle == event['type']:
    target_node.inputs[edge.targetHandle] = event['content']
```

#### Step 3: Target Receives Input

The target node accesses the data using the `targetHandle` as the key:

```python
# In NodeChat.py
async def process(self, chat_log):
    user_msg = self.get_input('handle_user_message')  # ← Key from targetHandle
    self.chat.add_user_message(user_msg)
```

### Visual Flow Diagram

```
Source Node              Edge                    Target Node
┌──────────────┐        ┌────────────┐         ┌──────────────┐
│NodeUserInput │        │            │         │   NodeChat   │
│              │        │            │         │              │
│yield_static( │        │            │         │              │
│  content=    │        │            │         │              │
│  "hello",    │        │            │         │              │
│  content_type│        │            │         │              │
│  ='handle_   │        │            │         │              │
│  user_message│        │            │         │              │
│)             │        │            │         │              │
└──────┬───────┘        │            │         │              │
       │                │            │         │              │
       ▼                │            │         │              │
Event Emitted:          │            │         │              │
┌──────────────┐        │            │         │              │
│type:         │───────▶│sourceHandle│         │              │
│'handle_user_ │        │'handle_user│         │              │
│message'      │        │_message'   │         │              │
│              │        │            │         │              │
│content: {...}│        │targetHandle│────────▶│self.inputs[  │
└──────────────┘        │'handle_user│         │'handle_user_ │
                        │_message'   │         │message']     │
                        └────────────┘         │= content     │
                                               └──────────────┘
```

### Critical Routing Rules

#### Rule 1: sourceHandle MUST match content_type
```python
# Node code
yield self.yield_static(data, content_type='my_output')

# Edge MUST use
"sourceHandle": "my_output"  # ← Exact match required
```

#### Rule 2: targetHandle becomes input key
```python
# Edge uses
"targetHandle": "my_input"

# Target node accesses via
self.get_input('my_input')  # ← Same key
```

#### Rule 3: Multiple outputs need different types
```python
# NodeUserInput emits 3 separate events
yield self.yield_static(text, content_type='handle_user_message')
yield self.yield_static(files, content_type='handle_user_files')
yield self.yield_static(images, content_type='handle_user_images')

# Each routed independently by sourceHandle
```

#### Rule 4: 'end' type maps to 'default' handle
```python
# Most nodes use default content_type
yield self.yield_static(result)  # content_type defaults to 'end'

# Edge can use either
"sourceHandle": "end"      # Direct match
"sourceHandle": "default"  # Alias for compatibility
```

### Creating Custom Nodes

Template for custom node implementation:

```python
from magic_agents.node_system.Node import Node

class CustomProcessor(Node):
    # Define handle constants for clarity
    INPUT_DATA = 'handle_input_data'
    OUTPUT_SUCCESS = 'handle_success'
    OUTPUT_ERROR = 'handle_error'
    
    async def process(self, chat_log):
        # Get input using targetHandle key from edge
        data = self.get_input(self.INPUT_DATA, required=True)
        
        try:
            result = self.process_logic(data)
            # Emit success with typed output
            yield self.yield_static(
                result,
                content_type=self.OUTPUT_SUCCESS
            )
        except Exception as e:
            # Emit error with different type for alternate routing
            yield self.yield_static(
                {'error': str(e)},
                content_type=self.OUTPUT_ERROR
            )
```

### Common Routing Issues

**Issue**: "My edge isn't routing data"
```
Checklist:
□ Does sourceHandle match the content_type in yield_static()?
□ Is there a typo? (handle-user vs handle_user)
□ Did the source node actually emit that event type?
□ Check executor logs for routing decisions
```

**Issue**: "Target node says input is missing"
```
Checklist:
□ Does get_input() key match targetHandle in edge?
□ Did the edge actually connect (check graph structure)?
□ Is the source node executing before target?
□ Is the source node yielding content?
```

**Issue**: "NodeLoop routing doesn't work as documented"
```
Solution: NodeLoop uses content_type='content' for items and 
content_type='end' for aggregation. Use these in sourceHandle:
- Item iteration: "sourceHandle": "content"
- Aggregation: "sourceHandle": "default"
```

---

## Node Summary

| Node | Type | Purpose | Key Inputs | Key Outputs |
|------|------|---------|------------|-------------|
| `NodeUserInput` | Source | User input capture | None | `handle_user_message`, `handle_user_files`, `handle_user_images` |
| `NodeText` | Source | Static text | None | default (text) |
| `NodeClientLLM` | Provider | LLM client | None | default (MagicLLM) |
| `NodeChat` | Transformer | Chat preparation | `handle-system-context`, `handle_user_message`, `handle_messages` | default (ModelChat) |
| `NodeLLM` | Generator | LLM generation | `handle-client-provider`, `handle-chat`, `handle_user_message` | default (text/JSON) |
| `NodeParser` | Transformer | Template parsing | Any (dynamic) | default (parsed text) |
| `NodeFetch` | Action | HTTP requests | Any (for templating) | default (JSON) |
| `NodeConditional` | Router | Conditional branching | `handle_input` (+ optional extras) | Dynamic based on condition |
| `NodeLoop` | Iterator | List iteration | `handle_list`, `handle_loop` | `handle_item`, `handle_end` |
| `NodeInner` | Container | Nested flows | `handle_user_message` | `handle_execution_content` |
| `NodeSendMessage` | Output | Message with extras | `handle_send_extra` | default (ChatCompletionModel) |
| `NodeEND` | Terminal | Flow termination | Any | default (empty) |

## Compatibility Matrix

### Legend
- ✅ Highly Compatible - Direct, common usage
- 🟡 Compatible - Valid but less common
- 🟠 Conditionally Compatible - Specific scenarios only
- ❌ Not Compatible - Invalid connection

### Source → Target Compatibility

|  | UserInput | Text | ClientLLM | Chat | LLM | Parser | Fetch | Conditional | Loop | Inner | SendMessage | END |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **UserInput** | ❌ | ❌ | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ | 🟡 | ✅ | 🟡 | 🟡 |
| **Text** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ |
| **ClientLLM** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Chat** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **LLM** | ❌ | ❌ | ❌ | 🟠 | 🟠 | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ |
| **Parser** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Fetch** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ |
| **Conditional** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Loop** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 | 🟡 | ✅ | ✅ |
| **Inner** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 | ✅ | ✅ |
| **SendMessage** | ❌ | ❌ | ❌ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟠 | ✅ |

## Detailed Node Analysis

### 1. NodeUserInput
**Role**: Entry point for agent flows

**Outputs**:
- `handle_user_message` → User text input
- `handle_user_files` → File attachments  
- `handle_user_images` → Image attachments

**Best Connections**:
- → **NodeChat**: Natural chat flow (`handle_user_message` → `handle_user_message`)
- → **NodeParser**: Process user input with templates
- → **NodeConditional**: Branch based on user input
- → **NodeInner**: Pass to nested flow

### 2. NodeText
**Role**: Static text provider

**Outputs**: default (text string)

**Best Connections**:
- → **NodeChat**: System prompts (default → `handle-system-context`)
- → **NodeLLM**: Direct prompts (default → `handle_user_message`)
- → **NodeParser**: Template input
- → **NodeLoop**: Static list data (default → `handle_list`)

### 3. NodeClientLLM  
**Role**: LLM client provider (REQUIRED for NodeLLM)

**Outputs**: default (MagicLLM instance)

**Connections**: 
- → **NodeLLM** ONLY: (default → `handle-client-provider`)

### 4. NodeChat
**Role**: Chat message preparation

**Inputs**:
- `handle-system-context` → System prompt
- `handle_user_message` → User message
- `handle_messages` → Full message array
- `handle_user_images` → Images
- `handle_user_files` → Files

**Outputs**: default (ModelChat object)

**Connections**:
- → **NodeLLM** ONLY: (default → `handle-chat`)

### 5. NodeLLM
**Role**: LLM text generation

**Inputs** (REQUIRED):
- `handle-client-provider` → MagicLLM client from NodeClientLLM

**Inputs** (Optional):
- `handle-chat` → ModelChat from NodeChat
- `handle-system-context` → System prompt
- `handle_user_message` → User message

**Outputs**: default (generated text or JSON if `json_output=True`)

**Configuration**:
- `iterate=True` → Re-execute inside loops (required for NodeLoop)
- `json_output=True` → Parse JSON from output

**Best Connections**:
- → **NodeParser**: Post-process LLM output
- → **NodeConditional**: Branch on LLM decisions
- → **NodeFetch**: Use LLM output in API calls
- → **NodeSendMessage**: Send LLM response
- → **NodeEND**: Terminal

### 6. NodeParser
**Role**: Jinja2 template parsing

**Inputs**: Any (all used as template variables)

**Outputs**: default (parsed template string)

**Template Example**:
```jinja2
Hello {{ user_name }}, your request for {{ item }} is {{ status }}.
```

**Best Connections**: Can connect to almost any node

### 7. NodeFetch
**Role**: HTTP API requests

**Inputs**: Any (used in Jinja2 templates for URL/data/headers)

**Configuration**:
- `method` → GET, POST, PUT, DELETE, etc.
- `url` → API endpoint
- `data` or `json_data` → Request body (supports Jinja2)
- `headers` → Request headers

**Outputs**: default (JSON response)

**Best Connections**:
- → **NodeParser**: Process API response
- → **NodeLLM**: Feed API data to LLM
- → **NodeConditional**: Branch on response
- → **NodeLoop**: Iterate over response items

### 8. NodeConditional
**Role**: Conditional branching/routing

**Inputs**:
- `handle_input` (primary) → Condition context
- `handle_input_1`, `handle_input_2`, ... → Additional inputs

**Configuration**:
- `condition` (required) → Jinja2 template that renders to output handle name
- `merge_strategy` → 'flat' or 'namespaced'

**Outputs**: Dynamic handles based on condition evaluation

**Condition Examples**:
```jinja2
# IF pattern
{{ 'adult' if age >= 18 else 'minor' }}

# SWITCH pattern  
{{ status }}

# Complex multi-input (namespaced)
{{ 'approved' if user.age >= 18 and account.balance > 1000 else 'denied' }}
```

**Connections**: Can route to any downstream node type

### 9. NodeLoop
**Role**: List iteration with result aggregation

**Inputs**:
- `handle_list` (REQUIRED) → JSON string or list
- `handle_loop` (optional) → Per-iteration results to aggregate

**Outputs**:
- `content` (content_type, multiple) → Each list element during iteration
- `default` or `end` (content_type, once) → Aggregated results array after iteration

**Edge Configuration**:
- For iteration: `"sourceHandle": "content"` (not `handle_item`)
- For aggregation: `"sourceHandle": "default"` (not `handle_end`)

**Important Notes**:
- NodeLoop uses generic content_type values (`content` and `end`) for backward compatibility
- The OUTPUT_HANDLE_* constants in code are for reference only
- Connect downstream processing nodes (like NodeLLM with `iterate=True`) between iteration output and `handle_loop` input

**Best Connections**:
- `content` → **NodeLLM** (with `iterate=True`): Process each item
- `content` → **NodeParser**: Template each item
- `content` → **NodeFetch**: API call per item
- `default` → Any node: Process aggregated results

### 10. NodeInner
**Role**: Execute nested agent flow

**Inputs**: `handle_user_message` (REQUIRED)

**Outputs**:
- `handle_execution_content` → Aggregated inner flow content
- `handle_execution_extras` → Inner flow extras

**Configuration**: `magic_flow` → Inner AgentFlowModel definition

**Best Connections**:
- → **NodeParser**: Parse inner results
- → **NodeLLM**: Feed to LLM
- → **NodeConditional**: Branch on results
- → **NodeSendMessage**: Send inner output

### 11. NodeSendMessage
**Role**: Send message with extras/metadata

**Inputs**: `handle_send_extra` (optional) → Extra data

**Configuration**:
- `message` → Message text
- `json_extras` → JSON extra data

**Outputs**: default (ChatCompletionModel with extras)

**Best Connections**:
- → **NodeEND**: Most common (terminal)
- → **NodeParser**: Parse message
- → **NodeConditional**: Branch on message

### 12. NodeEND
**Role**: Flow termination marker

**Inputs**: Any (ignored)

**Outputs**: default (empty ChatCompletionModel)

**Connections**: None (terminal node)

## Common Connection Patterns

### Pattern 1: Simple Chat
```
NodeUserInput → NodeChat → NodeLLM → NodeEND
                            ↑
                  NodeClientLLM
```

### Pattern 2: System Prompt
```
NodeText → NodeChat → NodeLLM → NodeEND
             ↑           ↑
NodeUserInput    NodeClientLLM
```

### Pattern 3: Conditional Branch
```
                    → NodeLLM[branch1] → NodeEND
                   ↗
NodeUserInput → NodeConditional
                   ↘
                    → NodeLLM[branch2] → NodeEND
```

### Pattern 4: Loop Processing
```
NodeText → NodeLoop → NodeLLM → back to NodeLoop → NodeEND
                        ↑             (handle_loop)
                NodeClientLLM
```
*NodeLLM must have `iterate=True`*

### Pattern 5: API Integration
```
NodeUserInput → NodeParser → NodeFetch → NodeLLM → NodeEND
                                            ↑
                                    NodeClientLLM
```

### Pattern 6: Multi-Stage Pipeline
```
NodeUserInput → NodeLLM[1] → NodeParser → NodeFetch → NodeLLM[2] → NodeSendMessage → NodeEND
                   ↑                                      ↑
         NodeClientLLM[1]                      NodeClientLLM[2]
```

### Pattern 7: Inner Flow
```
NodeUserInput → NodeInner → NodeParser → NodeEND
                   ↓
          [Nested Flow Graph]
```

## Handle Naming Conventions

### Standard Input Handles
- `handle_user_message` → User text input
- `handle-system-context` → System prompt
- `handle-chat` → ModelChat object
- `handle-client-provider` → MagicLLM client
- `handle_list` → List for iteration
- `handle_loop` → Loop aggregation input
- `handle_input` → Primary conditional input
- `handle_send_extra` → SendMessage extra data

### Standard Output Handles
- `default` → Primary output (most nodes)
- `content` → Loop iteration output (NodeLoop items)
- `handle_execution_content` → Inner flow content
- `handle_user_message` → NodeUserInput text
- `handle_user_files` → NodeUserInput files
- `handle_user_images` → NodeUserInput images

**Note**: NodeLoop uses `content` for item iteration and `default`/`end` for aggregation (not `handle_item` or `handle_end`)

### Dynamic Handles (NodeConditional)
Handles are dynamically created based on condition template output. Example:
- Condition: `{{ 'adult' if age >= 18 else 'minor' }}`
- Output handles: `adult`, `minor`

## Edge Connection Rules

### Valid Edge Structure
```json
{
  "source": "node-id-1",
  "sourceHandle": "output-handle-name",
  "target": "node-id-2",
  "targetHandle": "input-handle-name"
}
```

### Critical Rules
1. **NodeClientLLM** must connect to **NodeLLM** via `handle-client-provider`
2. **NodeChat** output must connect to **NodeLLM** via `handle-chat`
3. **NodeLoop** requires `handle_list` input
4. **NodeLLM** inside loop must have `iterate=True` configuration
5. **NodeConditional** condition must render to exact output handle name
6. **NodeEND** accepts any input but has no outputs
7. **NodeInner** requires `magic_flow` configuration with nested graph

## Data Type Flow

### Text/String Flow
```
NodeText → NodeParser → NodeLLM → NodeSendMessage
```

### JSON Flow
```
NodeFetch (JSON) → NodeParser → NodeLLM
NodeLLM (json_output=True) → NodeConditional
```

### ModelChat Flow
```
NodeChat (ModelChat) → NodeLLM
```

### MagicLLM Client Flow
```
NodeClientLLM (MagicLLM) → NodeLLM
```

### List Flow
```
NodeLLM (JSON array) → NodeLoop → [process items] → NodeParser
```

## Advanced Patterns

### Multi-Input Conditional (Namespaced)
```json
{
  "id": "conditional-1",
  "type": "CONDITIONAL",
  "data": {
    "condition": "{{ 'approved' if input1.age >= 18 and input2.balance > 1000 else 'denied' }}",
    "merge_strategy": "namespaced"
  }
}
```

**Edges**:
- `NodeFetch[user].default` → `conditional-1.handle_input_1`
- `NodeFetch[account].default` → `conditional-1.handle_input_2`

### Nested Loops
```
Outer Loop → Inner Loop → Processing → Inner Loop (back) → Outer Loop (back)
```
*Use with caution: complex execution pattern*

### Chained LLM Calls
```
NodeLLM[1] → NodeParser → NodeLLM[2] → NodeParser → NodeLLM[3]
```
*Each NodeLLM needs its own NodeClientLLM connection*

## Troubleshooting

### Common Issues

**Issue**: NodeLLM not generating
- **Solution**: Ensure NodeClientLLM is connected to `handle-client-provider`

**Issue**: NodeLoop not iterating
- **Solution**: Check `handle_list` input is valid JSON array

**Issue**: NodeLLM generates same output for each loop item
- **Solution**: Set `iterate=True` in NodeLLM configuration

**Issue**: NodeConditional not routing
- **Solution**: Verify condition template renders to exact output handle name

**Issue**: NodeInner fails
- **Solution**: Ensure `magic_flow` is properly configured with valid nested graph

**Issue**: NodeFetch returns empty
- **Solution**: Check if any input handles have data (requires at least one)

## Best Practices

1. **Always connect NodeClientLLM** to NodeLLM nodes
2. **Use NodeParser** for complex string formatting and templating
3. **Set `iterate=True`** on NodeLLM inside loops
4. **Use NodeConditional** for branching logic instead of multiple parallel flows
5. **Aggregate loop results** by connecting back to `handle_loop` input
6. **Use NodeText** for system prompts and constants
7. **Chain NodeParser** for multi-stage data transformation
8. **Use NodeSendMessage** when you need to send metadata/extras
9. **Always end with NodeEND** for clean flow termination
10. **Test conditional templates** independently before deployment

---

*Generated from magic-agents node system analysis*
