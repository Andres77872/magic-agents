# Conditional Nodes Design Approaches

This document outlines **three different architectural approaches** for implementing conditional logic execution in Magic-Agents flows. Each approach offers different trade-offs in terms of simplicity, flexibility, and usability.

---

## Overview

Conditional nodes enable **branching logic** in agent flows, allowing the graph to take different execution paths based on runtime data. The two primary conditional patterns are:

1. **NodeIf** – Binary branching (true/false, yes/no, adult/minor)
2. **NodeSwitch** – Multi-way branching (success/error/timeout, status-based routing)

---

## Approach 1: Unified NodeConditional (Current Implementation)

### Description

A **single node type** (`NodeConditional`) that handles both if/else and switch/case patterns using **Jinja2 template evaluation**. The condition template renders to the **name of the output handle** that should be activated.

**Enhanced Features:**
- ✅ **Multiple input support** – merge data from multiple upstream nodes
- ✅ **Flexible merge strategies** – flat or namespaced merging
- ✅ **Data passthrough** – all merged input data flows to selected output
- ✅ **Dynamic outputs** – output handles determined by condition result

### Architecture

```
┌────────────────────────────────────────────────────────┐
│              NodeConditional                           │
├────────────────────────────────────────────────────────┤
│ Inputs:                                                │
│   - handle_input (primary, required)                  │
│   - handle_input_1, handle_input_2, ... (optional)    │
│                                                        │
│ Data:                                                  │
│   - condition (Jinja2 template → handle name)         │
│   - merge_strategy ('flat' | 'namespaced')            │
│                                                        │
│ Outputs:                                               │
│   - <dynamic> (determined by condition result)        │
└────────────────────────────────────────────────────────┘
```

### Node Specification

| Property | Value |
|----------|-------|
| **Type key** | `conditional` |
| **Input handles** | `handle_input` (required) – Primary JSON/string context<br>`handle_input_1`, `handle_input_2`, ... (optional) – Additional inputs |
| **Data fields** | `condition` – Jinja2 template that resolves to handle name<br>`merge_strategy` – `'flat'` or `'namespaced'` (default: `'flat'`) |
| **Output handles** | Dynamic – determined by condition result |

### Multiple Input Handling

#### Merge Strategy: Flat (Default)
All inputs merged into single flat dictionary. Later inputs override earlier ones for duplicate keys.

```json
{
  "id": "conditional",
  "type": "conditional",
  "data": {
    "condition": "{{ 'eligible' if age >= 18 and balance > 1000 else 'ineligible' }}",
    "merge_strategy": "flat"
  }
}
```
**Access:** Direct variable access: `{{ age }}`, `{{ balance }}`

#### Merge Strategy: Namespaced
Each input stored under its handle name to prevent key collisions.

```json
{
  "id": "conditional",
  "type": "conditional",
  "data": {
    "condition": "{{ 'valid' if user.age >= 18 and account.verified else 'invalid' }}",
    "merge_strategy": "namespaced"
  }
}
```
**Access:** Namespaced access: `{{ handle_input.age }}`, `{{ handle_input_1.verified }}`

### Examples

#### Example 1: Simple IF Pattern
```json
{
  "id": "age_check",
  "type": "conditional",
  "data": {
    "condition": "{{ 'adult' if age|int >= 18 else 'minor' }}"
  }
}
```

#### Example 2: SWITCH Pattern
```json
{
  "id": "status_router",
  "type": "conditional",
  "data": {
    "condition": "{{ status }}"
  }
}
```
If `status` is `"success"`, `"error"`, or `"timeout"`, the corresponding handle is activated.

#### Example 3: SWITCH with Default Fallback
```json
{
  "id": "payment_router",
  "type": "conditional",
  "data": {
    "condition": "{{ payment_method if payment_method in ['credit_card', 'paypal', 'crypto'] else 'unsupported' }}"
  }
}
```

#### Example 4: Complex Nested Logic
```json
{
  "id": "priority_check",
  "type": "conditional",
  "data": {
    "condition": "{{ 'urgent' if score >= 90 else ('high' if score >= 70 else ('medium' if score >= 40 else 'low')) }}"
  }
}
```

#### Example 5: Multiple Inputs with Flat Merge
```json
{
  "nodes": [
    {
      "id": "get_user",
      "type": "api_call",
      "data": {"endpoint": "/api/user"}
    },
    {
      "id": "get_account",
      "type": "api_call",
      "data": {"endpoint": "/api/account"}
    },
    {
      "id": "eligibility",
      "type": "conditional",
      "data": {
        "condition": "{{ 'eligible' if (age >= 18 and balance > 1000 and verified) else 'ineligible' }}",
        "merge_strategy": "flat"
      }
    }
  ],
  "edges": [
    {"from": "get_user", "to": "eligibility", "from_handle": "output", "to_handle": "handle_input"},
    {"from": "get_account", "to": "eligibility", "from_handle": "output", "to_handle": "handle_input_1"}
  ]
}
```

**Data Flow:**
1. `get_user` outputs: `{"age": 25, "name": "Alice"}`
2. `get_account` outputs: `{"balance": 5000, "verified": true}`
3. Merged context: `{"age": 25, "name": "Alice", "balance": 5000, "verified": true}`
4. Condition evaluates to: `"eligible"`
5. Downstream node receives all merged data

#### Example 6: Multiple Inputs with Namespaced Merge
```json
{
  "nodes": [
    {
      "id": "get_user_profile",
      "type": "api_call",
      "data": {"endpoint": "/api/profile/user"}
    },
    {
      "id": "get_company_profile",
      "type": "api_call",
      "data": {"endpoint": "/api/profile/company"}
    },
    {
      "id": "validator",
      "type": "conditional",
      "data": {
        "condition": "{{ 'valid' if (handle_input.verified and handle_input_1.verified) else 'invalid' }}",
        "merge_strategy": "namespaced"
      }
    }
  ],
  "edges": [
    {"from": "get_user_profile", "to": "validator", "from_handle": "output", "to_handle": "handle_input"},
    {"from": "get_company_profile", "to": "validator", "from_handle": "output", "to_handle": "handle_input_1"}
  ]
}
```

### Pros
- **Highly flexible** – can express any logic with Jinja2
- **Single implementation** – maintains DRY principle
- **Multiple inputs** – merge data from multiple sources
- **Extensible** – easy to add custom Jinja2 filters/functions
- **Compact** – fewer node types to maintain
- **Data passthrough** – complete context flows to downstream nodes

### Cons
- **Requires Jinja2 knowledge** – steeper learning curve for simple use cases
- **Template syntax** – can be verbose for simple conditions
- **Runtime errors** – template syntax errors only caught at runtime
- **Less discoverable** – output handles not declared in spec

### Implementation Status
✅ **Currently implemented** in `magic_agents.node_system.NodeConditional`  
📚 **Full guide available** in `docs/nodes/NodeConditionalGuide.md`  
💡 **Examples available** in `examples/conditional_examples.json`

---

## Approach 2: Separate NodeIf and NodeSwitch

### Description

Two **dedicated node types** optimized for their specific use cases:
- **NodeIf** – Binary branching with explicit `true`/`false` (or custom) output handles
- **NodeSwitch** – Multi-way branching with explicit case definitions

### Architecture

```
┌─────────────────────────────────────────┐
│              NodeIf                     │
├─────────────────────────────────────────┤
│ Input:  handle_input (value to test)   │
│ Data:   expression, operator, compare  │
│ Output: true_handle, false_handle      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│            NodeSwitch                   │
├─────────────────────────────────────────┤
│ Input:  handle_input (value to match)  │
│ Data:   cases (list), default_handle   │
│ Output: case_1, case_2, ..., default   │
└─────────────────────────────────────────┘
```

### NodeIf Specification

| Property | Value |
|----------|-------|
| **Type key** | `if` |
| **Input handles** | `handle_input` (required) |
| **Data fields** | `expression` (str) – Jinja2 expression evaluating to boolean<br>`true_handle` (str, default: `"true"`) – handle name for true path<br>`false_handle` (str, default: `"false"`) – handle name for false path |
| **Output handles** | `true_handle`, `false_handle` (or custom names) |

### NodeSwitch Specification

| Property | Value |
|----------|-------|
| **Type key** | `switch` |
| **Input handles** | `handle_input` (required) |
| **Data fields** | `value_expression` (str) – Jinja2 expression to extract match value<br>`cases` (list) – list of `{value, handle}` mappings<br>`default_handle` (str, default: `"default"`) – fallback handle |
| **Output handles** | One per case + default |

### Examples

**NodeIf Example:**
```json
{
  "id": "check_authenticated",
  "type": "if",
  "data": {
    "expression": "{{ user.authenticated }}",
    "true_handle": "authorized",
    "false_handle": "unauthorized"
  }
}
```

**NodeIf with Comparison:**
```json
{
  "id": "check_age",
  "type": "if",
  "data": {
    "expression": "{{ user.age >= 18 }}",
    "true_handle": "adult",
    "false_handle": "minor"
  }
}
```

**NodeSwitch Example:**
```json
{
  "id": "route_by_status",
  "type": "switch",
  "data": {
    "value_expression": "{{ status }}",
    "cases": [
      {"value": "success", "handle": "success_path"},
      {"value": "error", "handle": "error_path"},
      {"value": "timeout", "handle": "timeout_path"}
    ],
    "default_handle": "unknown_path"
  }
}
```

**NodeSwitch with Numeric Ranges:**
```json
{
  "id": "priority_router",
  "type": "switch",
  "data": {
    "value_expression": "{{ score }}",
    "cases": [
      {"value": ">90", "handle": "urgent"},
      {"value": "70-90", "handle": "high"},
      {"value": "<70", "handle": "normal"}
    ]
  }
}
```

### Pros
- **Explicit and clear** – purpose obvious from node type
- **Simpler configuration** – no need to learn Jinja2 for basic cases
- **Type safety** – clearer validation of inputs/outputs
- **Better discoverability** – handles are explicitly declared
- **IDE-friendly** – easier to provide autocomplete and validation

### Cons
- **More code to maintain** – two separate implementations
- **Less flexible** – complex logic requires NodeConditional anyway
- **Potential overlap** – users may be confused about which to use

### Implementation Pseudocode

**NodeIf:**
```python
class NodeIf(Node):
    INPUT_HANDLE_CTX = "handle_input"
    
    def __init__(self, expression: str, 
                 true_handle: str = "true", 
                 false_handle: str = "false", **kwargs):
        super().__init__(**kwargs)
        self.expression = expression
        self.true_handle = true_handle
        self.false_handle = false_handle
        self._template = jinja2.Environment().from_string(expression)
    
    async def process(self, chat_log):
        ctx = self.get_input(self.INPUT_HANDLE_CTX, required=True)
        if isinstance(ctx, str):
            ctx = json.loads(ctx)
        
        render_ctx = ctx if isinstance(ctx, dict) else {"value": ctx}
        result = self._template.render(**render_ctx)
        
        # Coerce to boolean
        is_true = str(result).strip().lower() in ('true', '1', 'yes')
        
        selected_handle = self.true_handle if is_true else self.false_handle
        self.outputs[selected_handle] = self.prep(ctx)
        
        yield {"type": selected_handle, "content": self.prep(ctx)}
        yield self.yield_static({"selected": selected_handle})
```

**NodeSwitch:**
```python
class NodeSwitch(Node):
    INPUT_HANDLE_CTX = "handle_input"
    
    def __init__(self, value_expression: str, 
                 cases: list, 
                 default_handle: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.value_expression = value_expression
        self.cases = cases  # [{"value": "x", "handle": "h1"}, ...]
        self.default_handle = default_handle
        self._template = jinja2.Environment().from_string(value_expression)
    
    async def process(self, chat_log):
        ctx = self.get_input(self.INPUT_HANDLE_CTX, required=True)
        if isinstance(ctx, str):
            ctx = json.loads(ctx)
        
        render_ctx = ctx if isinstance(ctx, dict) else {"value": ctx}
        value = str(self._template.render(**render_ctx)).strip()
        
        # Match against cases
        selected_handle = self.default_handle
        for case in self.cases:
            if self._match_case(value, case["value"]):
                selected_handle = case["handle"]
                break
        
        self.outputs[selected_handle] = self.prep(ctx)
        yield {"type": selected_handle, "content": self.prep(ctx)}
        yield self.yield_static({"selected": selected_handle, "matched_value": value})
    
    def _match_case(self, value: str, pattern: str) -> bool:
        # Support exact match, ranges, comparisons
        if pattern.startswith(">"):
            return float(value) > float(pattern[1:])
        elif pattern.startswith("<"):
            return float(value) < float(pattern[1:])
        elif "-" in pattern and pattern[0].isdigit():
            low, high = pattern.split("-")
            return float(low) <= float(value) <= float(high)
        else:
            return value == pattern
```

---

## Approach 3: Expression-Based Conditional Nodes

### Description

Simplified conditional nodes that use **Python expressions** (via `eval` with sandboxing) instead of Jinja2 templates. This approach is more familiar to developers and provides clearer semantics.

### Architecture

```
┌─────────────────────────────────────────┐
│              NodeIf                     │
├─────────────────────────────────────────┤
│ Input:  handle_input (context dict)    │
│ Data:   condition (Python expression)  │
│ Output: true, false (fixed names)      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│            NodeSwitch                   │
├─────────────────────────────────────────┤
│ Input:  handle_input (context dict)    │
│ Data:   expression, cases (dict)       │
│ Output: Dynamic (case keys + default)  │
└─────────────────────────────────────────┘
```

### NodeIf Specification

| Property | Value |
|----------|-------|
| **Type key** | `if` |
| **Input handles** | `handle_input` (required) |
| **Data fields** | `condition` (str) – Python boolean expression (e.g., `"age >= 18"`) |
| **Output handles** | `true`, `false` (fixed) |

### NodeSwitch Specification

| Property | Value |
|----------|-------|
| **Type key** | `switch` |
| **Input handles** | `handle_input` (required) |
| **Data fields** | `expression` (str) – Python expression to get value<br>`cases` (dict) – mapping of value → handle name |
| **Output handles** | Dynamic based on `cases` keys + `default` |

### Examples

**NodeIf Example:**
```json
{
  "id": "check_age",
  "type": "if",
  "data": {
    "condition": "age >= 18"
  }
}
```

**NodeSwitch Example:**
```json
{
  "id": "route_status",
  "type": "switch",
  "data": {
    "expression": "status",
    "cases": {
      "success": "success_path",
      "error": "error_path",
      "timeout": "timeout_path"
    }
  }
}
```

**Complex NodeIf:**
```json
{
  "id": "eligibility_check",
  "type": "if",
  "data": {
    "condition": "age >= 18 and credit_score > 600 and income > 30000"
  }
}
```

### Pros
- **Familiar syntax** – Python expressions are intuitive for developers
- **Simple configuration** – no template delimiters needed
- **Type-safe** – Python's type system aids validation
- **Easier debugging** – standard Python error messages
- **Better performance** – direct evaluation vs template rendering

### Cons
- **Security concerns** – `eval()` requires careful sandboxing
- **Limited context** – only input dict variables available
- **Less flexible** – can't use custom functions without extending sandbox
- **Potential injection** – must sanitize user-provided expressions

### Implementation with Sandboxed Eval

```python
import ast
import operator

class SafeEvaluator:
    """Safe evaluation of Python expressions with limited operations."""
    
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
        ast.Not: operator.not_,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
    }
    
    def eval(self, expr: str, context: dict) -> Any:
        """Safely evaluate expression with given context."""
        tree = ast.parse(expr, mode='eval')
        return self._eval_node(tree.body, context)
    
    def _eval_node(self, node, context):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return context.get(node.id)
        elif isinstance(node, ast.BinOp):
            op = self.ALLOWED_OPERATORS.get(type(node.op))
            if not op:
                raise ValueError(f"Operator {node.op} not allowed")
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            return op(left, right)
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context)
            result = True
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context)
                op_func = self.ALLOWED_OPERATORS.get(type(op))
                if not op_func:
                    raise ValueError(f"Operator {op} not allowed")
                result = result and op_func(left, right)
                left = right
            return result
        else:
            raise ValueError(f"Node type {type(node)} not allowed")

class NodeIf(Node):
    INPUT_HANDLE_CTX = "handle_input"
    OUTPUT_HANDLE_TRUE = "true"
    OUTPUT_HANDLE_FALSE = "false"
    
    def __init__(self, condition: str, **kwargs):
        super().__init__(**kwargs)
        self.condition = condition
        self.evaluator = SafeEvaluator()
    
    async def process(self, chat_log):
        ctx = self.get_input(self.INPUT_HANDLE_CTX, required=True)
        if isinstance(ctx, str):
            ctx = json.loads(ctx)
        
        try:
            result = self.evaluator.eval(self.condition, ctx)
            is_true = bool(result)
        except Exception as e:
            logger.error(f"NodeIf {self.node_id}: evaluation error: {e}")
            raise ValueError(f"Failed to evaluate condition: {self.condition}") from e
        
        selected_handle = self.OUTPUT_HANDLE_TRUE if is_true else self.OUTPUT_HANDLE_FALSE
        self.outputs[selected_handle] = self.prep(ctx)
        
        yield {"type": selected_handle, "content": self.prep(ctx)}
        yield self.yield_static({"condition": self.condition, "result": is_true})
```

---

## Comparison Matrix

| Feature | Approach 1 (Unified) | Approach 2 (Separate) | Approach 3 (Expression) |
|---------|---------------------|----------------------|------------------------|
| **Flexibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Simplicity** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Learning Curve** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Type Safety** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Maintainability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Security** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Recommendations

### Use Approach 1 (Current) When:
- Maximum flexibility is required
- Users are comfortable with Jinja2
- Complex conditional logic with filters/functions
- Need to minimize node type proliferation

### Use Approach 2 When:
- User audience is less technical
- Explicit is better than implicit
- Need strong IDE/tooling support
- Simple branching patterns are common

### Use Approach 3 When:
- Developer-focused audience (Python developers)
- Performance is critical
- Need familiar expression syntax
- Can ensure proper sandboxing

---

## Hybrid Recommendation

Implement **Approaches 1 + 2** together:
1. Keep `NodeConditional` for advanced users and complex logic
2. Add `NodeIf` and `NodeSwitch` as convenience wrappers for common cases
3. Have `NodeIf` and `NodeSwitch` internally delegate to `NodeConditional` with appropriate templates

This provides:
- **Backward compatibility** – existing flows continue working
- **Progressive disclosure** – beginners start with NodeIf/NodeSwitch
- **Single implementation** – reduces maintenance burden
- **Migration path** – users can graduate to NodeConditional as needed

Example delegation:
```python
class NodeIf(NodeConditional):
    def __init__(self, expression: str, true_handle="true", false_handle="false", **kwargs):
        # Convert to NodeConditional format
        condition = f"{{{{ '{true_handle}' if ({expression}) else '{false_handle}' }}}}"
        super().__init__(condition=condition, **kwargs)
```
