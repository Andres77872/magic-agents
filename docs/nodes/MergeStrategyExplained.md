# NodeConditional: merge_strategy Explained

Complete explanation of how `merge_strategy` works in `NodeConditional` when handling multiple inputs.

---

## What is merge_strategy?

When `NodeConditional` receives data from **multiple upstream nodes** (via `handle_input`, `handle_input_1`, `handle_input_2`, etc.), it must combine all the data into a **single context dictionary** before evaluating the Jinja2 condition template.

The `merge_strategy` field determines **HOW** these multiple inputs are combined.

---

## The Two Strategies

### Strategy 1: `"flat"` (Default)

**What it does:**
- Takes all input dictionaries and merges them into **one flat dictionary**
- All keys from all inputs are combined at the **top level**

**Example:**
```python
# Input from handle_input
{"age": 25, "name": "Alice"}

# Input from handle_input_1
{"balance": 5000, "verified": true}

# Input from handle_input_2
{"premium": false, "account_type": "standard"}

# After flat merge → Single flat dictionary
{
  "age": 25,
  "name": "Alice",
  "balance": 5000,
  "verified": true,
  "premium": false,
  "account_type": "standard"
}
```

**How to access in condition:**
```jinja2
{{ age }}
{{ balance }}
{{ premium }}
```
Simple, direct access to all variables.

**⚠️ KEY COLLISION WARNING:**
If multiple inputs have the **same key name**, the **later input wins** and **overwrites** earlier values.

```python
# Input from handle_input
{"status": "pending", "id": 123}

# Input from handle_input_1
{"status": "approved", "amount": 500}

# After flat merge
{"status": "approved", "id": 123, "amount": 500}
#  ↑ "pending" is LOST! Overwritten by handle_input_1
```

---

### Strategy 2: `"namespaced"`

**What it does:**
- Keeps each input **isolated** under its handle name
- Creates a **nested dictionary** structure
- **No key collisions** possible

**Example:**
```python
# Input from handle_input
{"age": 25, "status": "active"}

# Input from handle_input_1
{"balance": 5000, "status": "verified"}

# Input from handle_input_2
{"premium": false, "status": "enabled"}

# After namespaced merge → Nested dictionary
{
  "handle_input": {
    "age": 25,
    "status": "active"
  },
  "handle_input_1": {
    "balance": 5000,
    "status": "verified"
  },
  "handle_input_2": {
    "premium": false,
    "status": "enabled"
  }
}
```

**How to access in condition:**
```jinja2
{{ handle_input.age }}
{{ handle_input.status }}      ← "active"
{{ handle_input_1.balance }}
{{ handle_input_1.status }}    ← "verified"
{{ handle_input_2.status }}    ← "enabled"
```
Explicit, namespaced access. Must use `handle_name.field`.

**✅ NO COLLISIONS:**
All three `"status"` values are preserved and accessible independently.

---

## When to Use Each Strategy

### Use `"flat"` When:

1. **Different keys across inputs**
   ```python
   Input 1: {"age": 25}
   Input 2: {"balance": 5000}
   Input 3: {"premium": false}
   # No overlapping keys → flat is safe and simpler
   ```

2. **Simple, straightforward conditions**
   ```jinja2
   {{ 'eligible' if age >= 18 and balance > 1000 else 'ineligible' }}
   ```

3. **You want minimal typing**
   ```jinja2
   # Flat: simple
   {{ age }}
   
   # Namespaced: more verbose
   {{ handle_input.age }}
   ```

### Use `"namespaced"` When:

1. **Overlapping keys across inputs**
   ```python
   Input 1: {"status": "pending", "verified": true}
   Input 2: {"status": "approved", "verified": false}
   # Both have "status" and "verified" → use namespaced!
   ```

2. **Need to distinguish data sources**
   ```jinja2
   {{ 'valid' if handle_input.verified and handle_input_1.verified else 'invalid' }}
   # Clear: checking TWO different "verified" fields
   ```

3. **Debugging/clarity is important**
   ```jinja2
   # Explicit about where each value comes from
   {{ handle_input.name }}      ← from user profile
   {{ handle_input_1.name }}    ← from company profile
   ```

---

## Complete Examples

### Example 1: Flat Merge (Different Keys)

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
      "id": "check_eligibility",
      "type": "conditional",
      "data": {
        "condition": "{{ 'eligible' if age >= 18 and balance > 1000 and verified else 'ineligible' }}",
        "merge_strategy": "flat"
      }
    }
  ],
  "edges": [
    {"from": "get_user", "to": "check_eligibility", "from_handle": "output", "to_handle": "handle_input"},
    {"from": "get_account", "to": "check_eligibility", "from_handle": "output", "to_handle": "handle_input_1"}
  ]
}
```

**Data Flow:**
```python
# get_user outputs
{"age": 25, "name": "Alice"}

# get_account outputs
{"balance": 5000, "verified": true}

# Merged context (flat)
{"age": 25, "name": "Alice", "balance": 5000, "verified": true}

# Condition evaluates
"eligible"  # age=25 >= 18, balance=5000 > 1000, verified=true
```

---

### Example 2: Namespaced Merge (Overlapping Keys)

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
      "id": "validate_profiles",
      "type": "conditional",
      "data": {
        "condition": "{{ 'valid' if (handle_input.verified and handle_input_1.verified and handle_input.status == 'active') else 'invalid' }}",
        "merge_strategy": "namespaced"
      }
    }
  ],
  "edges": [
    {"from": "get_user_profile", "to": "validate_profiles", "from_handle": "output", "to_handle": "handle_input"},
    {"from": "get_company_profile", "to": "validate_profiles", "from_handle": "output", "to_handle": "handle_input_1"}
  ]
}
```

**Data Flow:**
```python
# get_user_profile outputs
{"name": "Alice", "status": "active", "verified": true}

# get_company_profile outputs
{"name": "TechCorp", "status": "approved", "verified": true}

# Merged context (namespaced)
{
  "handle_input": {"name": "Alice", "status": "active", "verified": true},
  "handle_input_1": {"name": "TechCorp", "status": "approved", "verified": true}
}

# Condition evaluates
"valid"  # Both verified=true AND user status="active"
```

**Why namespaced here?**
Both profiles have `name`, `status`, and `verified` fields. With flat merge, we'd lose half the data!

---

## Common Mistakes

### ❌ Mistake 1: Using flat with overlapping keys

```json
{
  "condition": "{{ 'valid' if status == 'active' }}",
  "merge_strategy": "flat"
}
```

**Problem:**
```python
Input 1: {"status": "active"}
Input 2: {"status": "pending"}
# Result: {"status": "pending"}  ← Input 1's value LOST!
```

**Fix:** Use namespaced
```json
{
  "condition": "{{ 'valid' if handle_input.status == 'active' and handle_input_1.status == 'pending' }}",
  "merge_strategy": "namespaced"
}
```

---

### ❌ Mistake 2: Forgetting namespace syntax with namespaced merge

```json
{
  "condition": "{{ 'valid' if age >= 18 }}",
  "merge_strategy": "namespaced"
}
```

**Problem:** With namespaced merge, `age` doesn't exist at top level!

**Error:** `UndefinedError: 'age' is undefined`

**Fix:** Use namespace
```json
{
  "condition": "{{ 'valid' if handle_input.age >= 18 }}",
  "merge_strategy": "namespaced"
}
```

---

### ❌ Mistake 3: Specifying merge_strategy for single input

```json
{
  "id": "check",
  "type": "conditional",
  "data": {
    "condition": "{{ 'adult' if age >= 18 else 'minor' }}",
    "merge_strategy": "flat"  // ← Unnecessary, only one input!
  }
}
```

**Not wrong, but unnecessary.** With only one input, merge_strategy has no effect. Just omit it:

```json
{
  "id": "check",
  "type": "conditional",
  "data": {
    "condition": "{{ 'adult' if age >= 18 else 'minor' }}"
  }
}
```

---

## Quick Decision Guide

```
┌─────────────────────────────────────────┐
│ How many inputs does your conditional  │
│ node receive?                           │
└─────────────┬───────────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
  ONE              TWO OR MORE
    │                   │
    ↓                   ↓
Don't specify    ┌──────────────────┐
merge_strategy   │ Do inputs have   │
(not needed)     │ overlapping keys?│
                 └─────┬────────────┘
                       │
              ┌────────┴────────┐
              │                 │
             NO                YES
              │                 │
              ↓                 ↓
      merge_strategy: "flat"   merge_strategy: "namespaced"
      (or omit - default)      Access: {{ handle_input.key }}
      Access: {{ key }}
```

---

## Technical Details

### Merge Order (Flat Strategy)

Inputs are merged in handle order:
1. `handle_input` (base)
2. `handle_input_1` (merges into base, overwrites on collision)
3. `handle_input_2` (merges, overwrites on collision)
4. ... and so on

**Code equivalent:**
```python
merged = {}
merged.update(handle_input)      # Base
merged.update(handle_input_1)    # Overwrites duplicates
merged.update(handle_input_2)    # Overwrites duplicates
# Result: latest values win
```

### Namespace Names (Namespaced Strategy)

The exact handle names are used as keys:
- `handle_input` → `{"handle_input": {...}}`
- `handle_input_1` → `{"handle_input_1": {...}}`
- `handle_input_2` → `{"handle_input_2": {...}}`

**Important:** Use the **exact handle name** in your templates.

---

## Summary Table

| Feature | Flat | Namespaced |
|---------|------|------------|
| **Structure** | Single flat dict | Nested dict by handle |
| **Access** | `{{ key }}` | `{{ handle_name.key }}` |
| **Collisions** | Later overwrites earlier | No collisions |
| **Verbosity** | Low | Higher |
| **Use case** | Different keys | Overlapping keys |
| **Default** | ✅ Yes | No |
| **Clarity** | Less explicit | More explicit |

---

## Related Documentation

- **Quick Start:** `NodeConditionalQuickStart.md`
- **Full Guide:** `NodeConditionalGuide.md`
- **Examples:** `examples/conditional_examples.json`
- **Implementation:** `magic_agents/node_system/NodeConditional.py`
