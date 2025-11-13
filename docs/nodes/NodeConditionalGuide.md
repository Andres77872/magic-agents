# NodeConditional Usage Guide (Approach 1)

Complete guide for using **NodeConditional** with multiple inputs, IF/SWITCH patterns, and data flow handling.

---

## Table of Contents

1. [Core Concept](#core-concept)
2. [Multiple Input Handling](#multiple-input-handling)
3. [Data Flow Patterns](#data-flow-patterns)
4. [IF Pattern Examples](#if-pattern-examples)
5. [SWITCH Pattern Examples](#switch-pattern-examples)

---

## Core Concept

`NodeConditional` evaluates a Jinja2 template against input context and activates the output handle matching the result.

```
┌──────────────────────────────────────────────────────────┐
│                    NodeConditional                       │
├──────────────────────────────────────────────────────────┤
│  Inputs:                                                 │
│    - handle_input (primary context)                     │
│    - handle_input_1, handle_input_2, ... (optional)     │
│                                                          │
│  Data:                                                   │
│    - condition (Jinja2 template → output handle name)   │
│    - merge_strategy ('flat' | 'namespaced')             │
│                                                          │
│  Outputs:                                                │
│    - <dynamic> (determined by condition result)         │
└──────────────────────────────────────────────────────────┘
```

**Key Features:**
- Multiple inputs with automatic merging
- Dynamic output handles
- Full Jinja2 expression support
- Data passthrough to selected output

---

## Multiple Input Handling

When `NodeConditional` has **2+ inputs**, it must combine them into a single context for evaluation. The `merge_strategy` field controls how this combination happens.

### Merge Strategy: Flat (Default)

**Purpose:** Combines all input dictionaries into a single flat structure.

**Behavior:**
- All keys from all inputs merged at the top level
- If multiple inputs have the same key, **later inputs override earlier**
- Order: `handle_input` → `handle_input_1` → `handle_input_2` → ...

```json
{
  "id": "conditional_1",
  "type": "conditional",
  "data": {
    "condition": "{{ 'adult' if age >= 18 else 'minor' }}",
    "merge_strategy": "flat"  // Optional - this is the default
  }
}
```

**Example:**
```python
# Input data
handle_input:   {"age": 25, "name": "Alice"}
handle_input_1: {"balance": 5000, "verified": true}

# After flat merge
{"age": 25, "name": "Alice", "balance": 5000, "verified": true}
```

**Access in template:** Direct variable access
- `{{ age }}` → 25
- `{{ name }}` → "Alice"
- `{{ balance }}` → 5000

**⚠️ Warning:** Key collisions result in data loss
```python
handle_input:   {"status": "pending"}
handle_input_1: {"status": "approved"}
# Result: {"status": "approved"}  ← First value lost!
```

---

### Merge Strategy: Namespaced

**Purpose:** Keeps each input isolated under its handle name to prevent key collisions.

**Behavior:**
- Each input stored as a nested object under its handle name
- No keys are lost or overridden
- Requires explicit namespace in condition templates

```json
{
  "id": "conditional_1",
  "type": "conditional",
  "data": {
    "condition": "{{ 'valid' if handle_input.age >= 18 and handle_input_1.verified else 'invalid' }}",
    "merge_strategy": "namespaced"
  }
}
```

**Example:**
```python
# Input data
handle_input:   {"age": 25, "status": "active"}
handle_input_1: {"balance": 5000, "status": "verified"}

# After namespaced merge
{
  "handle_input": {"age": 25, "status": "active"},
  "handle_input_1": {"balance": 5000, "status": "verified"}
}
```

**Access in template:** Namespaced access
- `{{ handle_input.age }}` → 25
- `{{ handle_input.status }}` → "active"
- `{{ handle_input_1.balance }}` → 5000
- `{{ handle_input_1.status }}` → "verified"

**✅ Advantage:** Both `status` values are preserved and accessible

---

### Choosing the Right Strategy

| Scenario | Recommended Strategy | Reason |
|----------|---------------------|--------|
| Single input | Either (no effect) | Only one input, no merging needed |
| Different keys | `flat` | Simpler syntax, no collision risk |
| Overlapping keys | `namespaced` | Prevents data loss from collisions |
| Need explicitness | `namespaced` | Clear source of each field |
| Simple conditions | `flat` | Less verbose templates |

### Visual Comparison

**Flat Merge:**
```
Input 1: {age: 25, name: "Alice"}       ┐
Input 2: {balance: 5000, verified: true}├─→ Merge Flat
Input 3: {premium: false}               ┘
                    ↓
         {age: 25, name: "Alice", balance: 5000, 
          verified: true, premium: false}
                    ↓
         Access: {{ age }}, {{ balance }}, {{ premium }}
```

**Namespaced Merge:**
```
Input 1: {age: 25, status: "active"}     ┐
Input 2: {balance: 5000, status: "ok"}   ├─→ Merge Namespaced
Input 3: {premium: false, status: "on"}  ┘
                    ↓
    {handle_input: {age: 25, status: "active"},
     handle_input_1: {balance: 5000, status: "ok"},
     handle_input_2: {premium: false, status: "on"}}
                    ↓
         Access: {{ handle_input.age }}, 
                 {{ handle_input.status }},  ← "active"
                 {{ handle_input_1.status }}, ← "ok"
                 {{ handle_input_2.status }}  ← "on"
         All three "status" values preserved!
```

---

## Data Flow Patterns

### Pattern 1: Single Input → Multiple Paths

```
NodeA ──▶ Conditional ──▶ NodeB (path1)
                      └──▶ NodeC (path2)
```

**Data Flow:**
- NodeA outputs: `{"age": 25, "name": "Alice"}`
- Conditional evaluates and selects: `"adult"`
- NodeB receives: `{"age": 25, "name": "Alice"}`

### Pattern 2: Multiple Inputs → Single Path

```
NodeA ──┐
NodeB ──┤──▶ Conditional ──▶ NodeD
NodeC ──┘
```

**Data Flow:**
- NodeA: `{"user_id": 123}`
- NodeB: `{"age": 25}`
- NodeC: `{"verified": true}`
- Merged: `{"user_id": 123, "age": 25, "verified": true}`
- NodeD receives all merged data

### Pattern 3: Multiple Inputs → Multiple Paths

```
NodeA ──┐
NodeB ──┤──▶ Conditional ──▶ NodeD (path1)
NodeC ──┘              └──▶ NodeE (path2)
```

All merged data flows to selected downstream node.

---

## IF Pattern Examples

### Example 1: Simple Age Check

```json
{
  "nodes": [
    {
      "id": "user_input",
      "type": "user_input",
      "data": {
        "prompt": "Enter your age:",
        "variable_name": "age"
      }
    },
    {
      "id": "age_check",
      "type": "conditional",
      "data": {
        "condition": "{{ 'adult' if age|int >= 18 else 'minor' }}"
      }
    },
    {
      "id": "adult_message",
      "type": "chat",
      "data": {
        "message": "You are an adult. Age: {{ age }}"
      }
    },
    {
      "id": "minor_message",
      "type": "chat",
      "data": {
        "message": "You are a minor. Age: {{ age }}"
      }
    }
  ],
  "edges": [
    {"from": "user_input", "to": "age_check", "from_handle": "output", "to_handle": "handle_input"},
    {"from": "age_check", "to": "adult_message", "from_handle": "adult", "to_handle": "handle_input"},
    {"from": "age_check", "to": "minor_message", "from_handle": "minor", "to_handle": "handle_input"}
  ]
}
```

### Example 2: Authentication Check with Custom Handles

```json
{
  "nodes": [
    {
      "id": "auth_check",
      "type": "conditional",
      "data": {
        "condition": "{{ 'authorized' if is_authenticated and has_token else 'unauthorized' }}"
      }
    },
    {
      "id": "secure_area",
      "type": "chat",
      "data": {"message": "Welcome! Token: {{ has_token }}"}
    },
    {
      "id": "login_prompt",
      "type": "chat",
      "data": {"message": "Please login first."}
    }
  ],
  "edges": [
    {"from": "auth_check", "to": "secure_area", "from_handle": "authorized", "to_handle": "handle_input"},
    {"from": "auth_check", "to": "login_prompt", "from_handle": "unauthorized", "to_handle": "handle_input"}
  ]
}
```

### Example 3: Multi-Input Eligibility Check

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
      "id": "eligibility_check",
      "type": "conditional",
      "data": {
        "condition": "{{ 'eligible' if (user.age >= 18 and account.balance > 1000 and account.verified) else 'ineligible' }}",
        "merge_strategy": "namespaced"
      }
    },
    {
      "id": "approve",
      "type": "chat",
      "data": {"message": "Approved for {{ user.name }}. Balance: ${{ account.balance }}"}
    },
    {
      "id": "deny",
      "type": "chat",
      "data": {"message": "Not eligible. Requirements not met."}
    }
  ],
  "edges": [
    {"from": "get_user", "to": "eligibility_check", "from_handle": "output", "to_handle": "handle_input"},
    {"from": "get_account", "to": "eligibility_check", "from_handle": "output", "to_handle": "handle_input_1"},
    {"from": "eligibility_check", "to": "approve", "from_handle": "eligible", "to_handle": "handle_input"},
    {"from": "eligibility_check", "to": "deny", "from_handle": "ineligible", "to_handle": "handle_input"}
  ]
}
```

### Example 4: Nested Conditionals (Priority Levels)

```json
{
  "nodes": [
    {
      "id": "priority_check",
      "type": "conditional",
      "data": {
        "condition": "{{ 'urgent' if score >= 90 else ('high' if score >= 70 else ('medium' if score >= 40 else 'low')) }}"
      }
    },
    {
      "id": "urgent_handler",
      "type": "chat",
      "data": {"message": "🔴 URGENT: Score {{ score }}"}
    },
    {
      "id": "high_handler",
      "type": "chat",
      "data": {"message": "🟠 HIGH: Score {{ score }}"}
    },
    {
      "id": "medium_handler",
      "type": "chat",
      "data": {"message": "🟡 MEDIUM: Score {{ score }}"}
    },
    {
      "id": "low_handler",
      "type": "chat",
      "data": {"message": "🟢 LOW: Score {{ score }}"}
    }
  ],
  "edges": [
    {"from": "priority_check", "to": "urgent_handler", "from_handle": "urgent", "to_handle": "handle_input"},
    {"from": "priority_check", "to": "high_handler", "from_handle": "high", "to_handle": "handle_input"},
    {"from": "priority_check", "to": "medium_handler", "from_handle": "medium", "to_handle": "handle_input"},
    {"from": "priority_check", "to": "low_handler", "from_handle": "low", "to_handle": "handle_input"}
  ]
}
```

---

## SWITCH Pattern Examples

### Example 1: Status Code Router

```json
{
  "nodes": [
    {
      "id": "api_call",
      "type": "api_call",
      "data": {"endpoint": "/api/process"}
    },
    {
      "id": "status_router",
      "type": "conditional",
      "data": {
        "condition": "{{ result.status }}"
      }
    },
    {
      "id": "success_handler",
      "type": "chat",
      "data": {"message": "✓ Success! Data: {{ result.data }}"}
    },
    {
      "id": "error_handler",
      "type": "chat",
      "data": {"message": "✗ Error: {{ result.error_message }}"}
    },
    {
      "id": "timeout_handler",
      "type": "chat",
      "data": {"message": "⏱ Timeout occurred"}
    }
  ],
  "edges": [
    {"from": "api_call", "to": "status_router", "from_handle": "output", "to_handle": "handle_input"},
    {"from": "status_router", "to": "success_handler", "from_handle": "success", "to_handle": "handle_input"},
    {"from": "status_router", "to": "error_handler", "from_handle": "error", "to_handle": "handle_input"},
    {"from": "status_router", "to": "timeout_handler", "from_handle": "timeout", "to_handle": "handle_input"}
  ]
}
```

### Example 2: Payment Method Router with Default

```json
{
  "nodes": [
    {
      "id": "payment_router",
      "type": "conditional",
      "data": {
        "condition": "{{ payment_method if payment_method in ['credit_card', 'paypal', 'crypto'] else 'unsupported' }}"
      }
    },
    {
      "id": "cc_processor",
      "type": "chat",
      "data": {"message": "Processing credit card..."}
    },
    {
      "id": "paypal_processor",
      "type": "chat",
      "data": {"message": "Redirecting to PayPal..."}
    },
    {
      "id": "crypto_processor",
      "type": "chat",
      "data": {"message": "Awaiting crypto transaction..."}
    },
    {
      "id": "unsupported_msg",
      "type": "chat",
      "data": {"message": "Method '{{ payment_method }}' not supported"}
    }
  ],
  "edges": [
    {"from": "payment_router", "to": "cc_processor", "from_handle": "credit_card", "to_handle": "handle_input"},
    {"from": "payment_router", "to": "paypal_processor", "from_handle": "paypal", "to_handle": "handle_input"},
    {"from": "payment_router", "to": "crypto_processor", "from_handle": "crypto", "to_handle": "handle_input"},
    {"from": "payment_router", "to": "unsupported_msg", "from_handle": "unsupported", "to_handle": "handle_input"}
  ]
}
```

### Example 3: Temperature Range Router

```json
{
  "nodes": [
    {
      "id": "temp_router",
      "type": "conditional",
      "data": {
        "condition": "{{ 'freezing' if temp < 0 else ('cold' if temp < 15 else ('moderate' if temp < 25 else ('hot' if temp < 35 else 'extreme'))) }}"
      }
    },
    {
      "id": "freezing_alert",
      "type": "chat",
      "data": {"message": "❄️ FREEZING: {{ temp }}°C"}
    },
    {
      "id": "cold_notice",
      "type": "chat",
      "data": {"message": "🧥 COLD: {{ temp }}°C"}
    },
    {
      "id": "moderate_notice",
      "type": "chat",
      "data": {"message": "☀️ MODERATE: {{ temp }}°C"}
    },
    {
      "id": "hot_warning",
      "type": "chat",
      "data": {"message": "🔥 HOT: {{ temp }}°C"}
    },
    {
      "id": "extreme_alert",
      "type": "chat",
      "data": {"message": "⚠️ EXTREME: {{ temp }}°C"}
    }
  ],
  "edges": [
    {"from": "temp_router", "to": "freezing_alert", "from_handle": "freezing", "to_handle": "handle_input"},
    {"from": "temp_router", "to": "cold_notice", "from_handle": "cold", "to_handle": "handle_input"},
    {"from": "temp_router", "to": "moderate_notice", "from_handle": "moderate", "to_handle": "handle_input"},
    {"from": "temp_router", "to": "hot_warning", "from_handle": "hot", "to_handle": "handle_input"},
    {"from": "temp_router", "to": "extreme_alert", "from_handle": "extreme", "to_handle": "handle_input"}
  ]
}
```

### Example 4: Multi-Input Order Router

```json
{
  "nodes": [
    {
      "id": "get_order",
      "type": "database_query",
      "data": {"query": "SELECT * FROM orders WHERE id = {{ order_id }}"}
    },
    {
      "id": "get_customer",
      "type": "database_query",
      "data": {"query": "SELECT tier FROM customers WHERE id = {{ customer_id }}"}
    },
    {
      "id": "get_inventory",
      "type": "database_query",
      "data": {"query": "SELECT stock FROM inventory WHERE product_id = {{ product_id }}"}
    },
    {
      "id": "order_router",
      "type": "conditional",
      "data": {
        "condition": "{{ 'expedite' if (customer_tier == 'premium' and order_total > 1000) else ('standard' if inventory_stock > 10 else ('backorder' if inventory_stock > 0 else 'out_of_stock')) }}",
        "merge_strategy": "flat"
      }
    },
    {
      "id": "expedite",
      "type": "chat",
      "data": {"message": "⚡ EXPEDITED: Premium order ${{ order_total }}"}
    },
    {
      "id": "standard",
      "type": "chat",
      "data": {"message": "📦 STANDARD: Stock: {{ inventory_stock }}"}
    },
    {
      "id": "backorder",
      "type": "chat",
      "data": {"message": "⏳ BACKORDER: Limited stock"}
    },
    {
      "id": "out_of_stock",
      "type": "chat",
      "data": {"message": "❌ OUT OF STOCK"}
    }
  ],
  "edges": [
    {"from": "get_order", "to": "order_router", "from_handle": "output", "to_handle": "handle_input"},
    {"from": "get_customer", "to": "order_router", "from_handle": "output", "to_handle": "handle_input_1"},
    {"from": "get_inventory", "to": "order_router", "from_handle": "output", "to_handle": "handle_input_2"},
    {"from": "order_router", "to": "expedite", "from_handle": "expedite", "to_handle": "handle_input"},
    {"from": "order_router", "to": "standard", "from_handle": "standard", "to_handle": "handle_input"},
    {"from": "order_router", "to": "backorder", "from_handle": "backorder", "to_handle": "handle_input"},
    {"from": "order_router", "to": "out_of_stock", "from_handle": "out_of_stock", "to_handle": "handle_input"}
  ]
}
```

**Data Flow:**
1. `get_order`: `{"order_total": 1500, "order_id": "ORD123"}`
2. `get_customer`: `{"customer_tier": "premium"}`
3. `get_inventory`: `{"inventory_stock": 5}`
4. Merged context: All three combined
5. Selected handle: `"expedite"`
6. Downstream node receives: All merged data

---

## Key Takeaways

### Multiple Inputs
- Use `handle_input`, `handle_input_1`, `handle_input_2`, etc.
- Choose merge strategy based on needs:
  - `flat`: Simple, direct access
  - `namespaced`: Safe, avoids collisions

### Data Passthrough
- **All merged input data** flows to selected output
- Downstream nodes receive complete context
- Use Jinja2 templates in downstream nodes to access data

### Output Handles
- Determined dynamically by condition result
- Must create edges for each possible output
- Unconnected outputs are valid (dead ends)

### Condition Patterns
- **IF**: `{{ 'true_case' if condition else 'false_case' }}`
- **SWITCH**: `{{ variable }}` or `{{ variable if variable in allowed else 'default' }}`
- **Complex**: Nested ternaries for multi-level decisions
