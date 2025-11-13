# NodeConditional Quick Start Guide

**TL;DR:** Use `NodeConditional` for both IF and SWITCH patterns with support for multiple inputs.

## The Golden Rule

> **The Jinja2 template in `condition` must evaluate to a STRING that matches an output handle name.**

```python
Template: "{{ 'adult' if age >= 18 else 'minor' }}"
Result:   "adult"  or  "minor"
          ↓              ↓
     Activates handle with that exact name
          ↓              ↓
     Must have edge: from_handle="adult"  or  from_handle="minor"
```

---

## Basic Syntax

```json
{
  "id": "my_conditional",
  "type": "conditional",
  "data": {
    "condition": "{{ jinja2_expression_that_returns_handle_name }}",
    "merge_strategy": "flat"  // Optional: "flat" (default) or "namespaced"
  }
}
```

**Note:** `merge_strategy` is only relevant when you have **multiple inputs** (handle_input_1, handle_input_2, etc.). For single input, it has no effect.

---

## IF Pattern (Binary Decision)

### Simple IF
```json
{
  "condition": "{{ 'yes' if age >= 18 else 'no' }}"
}
```

### Custom Handle Names
```json
{
  "condition": "{{ 'authorized' if is_authenticated else 'unauthorized' }}"
}
```

### Complex Condition
```json
{
  "condition": "{{ 'approved' if (age >= 18 and score > 600 and balance > 1000) else 'denied' }}"
}
```

---

## SWITCH Pattern (Multi-way Decision)

### Direct Value Match
```json
{
  "condition": "{{ status }}"
}
```
- If `status = "success"` → activates `success` handle
- If `status = "error"` → activates `error` handle

### With Default Fallback
```json
{
  "condition": "{{ payment_method if payment_method in ['card', 'paypal'] else 'unsupported' }}"
}
```

### Range-Based
```json
{
  "condition": "{{ 'low' if score < 40 else ('medium' if score < 70 else 'high') }}"
}
```

---

## Multiple Inputs

**When to use:** When your conditional decision depends on data from **2 or more upstream nodes**.

### Pattern 1: Flat Merge (Default - Simple)

**Use when:** Inputs have different keys, or you want simple direct access.

**Configuration:**
```json
{
  "condition": "{{ 'eligible' if age >= 18 and balance > 1000 else 'ineligible' }}",
  "merge_strategy": "flat"  // Can be omitted (this is default)
}
```

**Edges:**
```json
{
  "edges": [
    {"from": "get_user", "to": "conditional", "to_handle": "handle_input"},
    {"from": "get_account", "to": "conditional", "to_handle": "handle_input_1"},
    {"from": "get_settings", "to": "conditional", "to_handle": "handle_input_2"}
  ]
}
```

**Example data flow:**
```python
# get_user outputs:
{"age": 25, "name": "Alice"}

# get_account outputs:
{"balance": 5000, "verified": true}

# get_settings outputs:
{"premium": false}

# After FLAT merge → Single dictionary:
{"age": 25, "name": "Alice", "balance": 5000, "verified": true, "premium": false}
```

**How to access in condition:**
- Direct variable access: `{{ age }}`, `{{ balance }}`, `{{ premium }}`
- All keys are at the top level

**⚠️ Key collision behavior:**
- If multiple inputs have the same key, **later inputs win**
- Order: `handle_input` → `handle_input_1` → `handle_input_2` → ...
```python
# Example collision:
handle_input: {"status": "pending", "id": 1}
handle_input_1: {"status": "approved", "id": 2}

# Result: status = "approved" (from handle_input_1)
{"status": "approved", "id": 2}
```

### Pattern 2: Namespaced Merge (Safe - Explicit)

**Use when:** 
- Inputs have **overlapping keys** (e.g., both have "status" field)
- You want to be **explicit** about data sources
- Debugging/clarity is important

**Configuration:**
```json
{
  "condition": "{{ 'valid' if handle_input.age >= 18 and handle_input_1.balance > 1000 else 'invalid' }}",
  "merge_strategy": "namespaced"
}
```

**Example data flow:**
```python
# get_user outputs:
{"age": 25, "status": "active"}

# get_account outputs:
{"balance": 5000, "status": "verified"}

# After NAMESPACED merge → Nested dictionary:
{
  "handle_input": {"age": 25, "status": "active"},
  "handle_input_1": {"balance": 5000, "status": "verified"}
}
```

**How to access in condition:**
- Namespaced access: `{{ handle_input.age }}`, `{{ handle_input_1.balance }}`
- Different status values: `{{ handle_input.status }}` vs `{{ handle_input_1.status }}`
- Each input is isolated under its handle name

**✅ Key collision prevented:**
```python
# Both inputs have "status" - no conflict!
handle_input.status = "active"
handle_input_1.status = "verified"
# Both preserved and accessible
```

---

## Data Flow

### Single Input → Multiple Outputs

```
┌─────────┐
│  NodeA  │
└────┬────┘
     │
┌────▼────────┐
│ Conditional │
└─┬─────────┬─┘
  │         │
┌─▼──┐   ┌─▼──┐
│NodeB│   │NodeC│
└────┘   └────┘
```

**Data:** NodeA's output flows to selected downstream node (NodeB or NodeC)

### Multiple Inputs → Single Output

```
┌────┐
│NodeA│──┐
└────┘  │
        ├──▶┌────────────┐
┌────┐  │   │Conditional │──▶┌────┐
│NodeB│──┤   └────────────┘   │NodeD│
└────┘  │                      └────┘
        │
┌────┐  │
│NodeC│──┘
└────┘
```

**Data:** All inputs (A, B, C) merged → NodeD receives complete merged context

### Multiple Inputs → Multiple Outputs

```
┌────┐
│NodeA│──┐
└────┘  │
        ├──▶┌────────────┐──▶┌────┐
┌────┐  │   │Conditional │   │NodeD│
│NodeB│──┤   └─────┬──────┘   └────┘
└────┘  │         │
        │         └──▶┌────┐
┌────┐  │             │NodeE│
│NodeC│──┘             └────┘
└────┘
```

**Data:** All inputs merged → selected downstream node receives everything

---

## Common Patterns

### 1. Age Gate
```json
{
  "condition": "{{ 'adult' if age|int >= 18 else 'minor' }}"
}
```

### 2. Status Router
```json
{
  "condition": "{{ status }}"
}
```

### 3. Priority Classifier
```json
{
  "condition": "{{ 'urgent' if score >= 90 else ('high' if score >= 70 else 'low') }}"
}
```

### 4. Authentication Check
```json
{
  "condition": "{{ 'authorized' if (authenticated and has_token) else 'unauthorized' }}"
}
```

### 5. Multi-Source Validation
```json
{
  "condition": "{{ 'valid' if (user.verified and account.active and payment.valid) else 'invalid' }}",
  "merge_strategy": "namespaced"
}
```

---

## Choosing Between Flat vs Namespaced

### Decision Tree

```
Do you have multiple inputs?
├─ NO → merge_strategy doesn't matter (omit it)
└─ YES → Do the inputs have overlapping keys?
    ├─ NO → Use "flat" (default, simpler syntax)
    └─ YES → Use "namespaced" (prevents collisions)
```

### Quick Comparison

| Aspect | Flat | Namespaced |
|--------|------|------------|
| **Syntax** | `{{ age }}` | `{{ handle_input.age }}` |
| **When to use** | Different keys | Overlapping keys |
| **Pros** | ✅ Simpler, less typing | ✅ No collisions, explicit |
| **Cons** | ⚠️ Key collisions possible | ⚠️ More verbose |
| **Default** | ✅ Yes | ❌ No |

### Real-World Examples

**Use Flat when:**
```json
// User data + Account data + Settings (all different keys)
{
  "condition": "{{ 'eligible' if age >= 18 and balance > 1000 and premium else 'ineligible' }}",
  "merge_strategy": "flat"
}
```

**Use Namespaced when:**
```json
// User profile + Company profile (both have "name", "status", "verified")
{
  "condition": "{{ 'valid' if handle_input.verified and handle_input_1.verified else 'invalid' }}",
  "merge_strategy": "namespaced"
}
```

---

## Tips & Best Practices

### ✅ DO

- **Use meaningful handle names**: `adult`/`minor` instead of `true`/`false`
- **Add default fallbacks** for SWITCH: `{{ value if value in allowed else 'default' }}`
- **Use namespaced merge** when inputs have overlapping keys
- **Apply Jinja2 filters** for type safety: `{{ age|int }}`, `{{ price|float }}`
- **Omit merge_strategy** for single input (it's unnecessary)
- **Pass all data downstream**: merged context automatically flows to next node

### ❌ DON'T

- **Don't use empty conditions**: always return a handle name
- **Don't forget output edges**: create edges for all possible handles
- **Don't hardcode complex logic**: extract to variables when possible
- **Don't ignore merge collisions**: use namespaced merge if keys overlap
- **Don't specify merge_strategy** for single input nodes (no effect)

---

## Debugging

### Check What Gets Merged

Add a chat node after conditional to inspect data:
```json
{
  "id": "debug_output",
  "type": "chat",
  "data": {
    "message": "Received: {{ handle_input | tojson }}"
  }
}
```

### Common Errors

**Error:** `Undefined variable 'age'`
- **Fix:** Ensure input node provides `age` field, or use `{{ age|default(0) }}`

**Error:** `Empty handle rendered`
- **Fix:** Condition must return non-empty string: `{{ 'fallback' if not value else value }}`

**Error:** `No edge for handle 'X'`
- **Fix:** Create edge for all possible output handles

---

## Complete Example

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
        "condition": "{{ 'approved' if (age >= 18 and balance > 1000 and verified) else 'denied' }}",
        "merge_strategy": "flat"
      }
    },
    {
      "id": "approved_msg",
      "type": "chat",
      "data": {"message": "✓ Approved! Balance: ${{ balance }}"}
    },
    {
      "id": "denied_msg",
      "type": "chat",
      "data": {"message": "✗ Denied. Requirements not met."}
    }
  ],
  "edges": [
    {"from": "get_user", "to": "eligibility_check", "from_handle": "output", "to_handle": "handle_input"},
    {"from": "get_account", "to": "eligibility_check", "from_handle": "output", "to_handle": "handle_input_1"},
    {"from": "eligibility_check", "to": "approved_msg", "from_handle": "approved", "to_handle": "handle_input"},
    {"from": "eligibility_check", "to": "denied_msg", "from_handle": "denied", "to_handle": "handle_input"}
  ]
}
```

---

## Summary: Key Concepts

### 1. The Condition Template
- **Must return a string** that matches an output handle name
- Example: `"{{ 'success' if code == 200 else 'error' }}"` returns `"success"` or `"error"`
- That string becomes the active output handle

### 2. Output Handles
- **Dynamic** - determined by template result
- **Require edges** - must create edge for each possible handle
- **Case-sensitive** - `"Success"` ≠ `"success"`

### 3. Merge Strategy (for multiple inputs only)
- **`flat`** (default): Simple merge, direct access, collisions possible
  - Use when: Different keys, simple scenarios
  - Access: `{{ age }}`, `{{ balance }}`
- **`namespaced`**: Safe merge, explicit access, no collisions
  - Use when: Overlapping keys, need clarity
  - Access: `{{ handle_input.age }}`, `{{ handle_input_1.balance }}`

### 4. Data Flow
- **All merged input data** flows to the selected output
- Downstream nodes receive complete context
- Can use Jinja2 templates to access data

### 5. Common Mistakes
- ❌ Empty condition results
- ❌ Missing edges for possible handles
- ❌ Key collisions with flat merge
- ❌ Forgetting to use namespace syntax with namespaced merge

---

## Reference

- **Full Guide:** `docs/nodes/NodeConditionalGuide.md`
- **Examples:** `examples/conditional_examples.json`
- **Implementation:** `magic_agents/node_system/NodeConditional.py`
- **Approaches:** `docs/nodes/ConditionalNodesApproaches.md`
