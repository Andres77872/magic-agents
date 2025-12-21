# NodeConditional

Implements **branching logic** (`if/else` and `switch`-style) inside a Magic-Agents flow.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeConditional` |
| **Type key** | `conditional` |
| **Input handles** | At least one input is required. Conventionally `handle_input` is used as the primary input; additional inputs are allowed. |
| **Data fields (JSON spec)** | See [Data Fields](#data-fields) section below |
| **Dynamic output handles** | *Any* handle names referenced by `condition` (e.g. `true`, `false`, `success`, `timeout`, …). |

## Data Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `condition` | string | *required* | Jinja2 template that resolves to the name of the outgoing handle to continue with. |
| `merge_strategy` | `"flat"` \| `"namespaced"` | `"flat"` | How to merge multiple inputs into context. |
| `output_handles` | string[] | `null` | Declared output handle names for build-time validation. |
| `default_handle` | string | `null` | Fallback handle if condition evaluates to empty string. |
| `handles` | object | `null` | Custom handle name mappings. |

### Example with All Options

```json
{
  "id": "smart_router",
  "type": "conditional",
  "data": {
    "condition": "{{ status | default('') }}",
    "merge_strategy": "flat",
    "output_handles": ["success", "error", "pending"],
    "default_handle": "error",
    "handles": {
      "input": "my_custom_input"
    }
  }
}
```

## Configurable Handles

Handle names can be customized via the `handles` field in `data`:

```json
{
  "id": "router",
  "type": "conditional",
  "data": {
    "condition": "{{ status }}",
    "handles": {
      "input": "my_custom_input_handle"
    }
  }
}
```

| Handle Key | Aliases | Default Value | Description |
|------------|---------|---------------|-------------|
| `input` | `context` | `handle_input` | Primary input handle |

## Runtime Behaviour

1. **Gather context** – merge all received inputs into a single render context.
   - JSON strings are decoded when possible.
   - The primary input (`handle_input`) is also exposed as `value` (so templates can branch on `value` directly).
2. **Render** the `condition` template using *jinja2* with the merged context in scope.
   ```python
   env = jinja2.Environment()
   selected = env.from_string(condition).render(**ctx)
   selected = str(selected).strip()
   ```
3. **Emit chosen path** – the node yields an event whose `type` is the selected handle:
   ```python
   yield self.yield_static(ctx, content_type=selected)
   ```
4. **Mark bypass (executor)** – `execute_graph()` bypasses every *other* outgoing edge from this node and recursively propagates bypass to descendants **only when all parents are bypassed**.
   *Algorithm sketch (from `magic_agents/agt_flow.py`)*
   ```text
   dfs(node):
       if node already bypassed: return
       if all parents bypassed:
           node.bypass = True
           for child in node.children:
               dfs(child)
   ```
5. **If no edge matches** the rendered handle, execution yields a `debug` error and all outgoing edges are bypassed.

## Flow-Execution Notes

- Downstream nodes must listen on the handle named by the rendered output.
- A node whose *all* parents are bypassed should **auto-bypass** without executing.
- Mixed parents (some bypassed, some not) ⇒ wait until at least one non-bypassed parent completes.

## Example 1 – If/Else
```json
{
  "id": "check_age",
  "type": "conditional",
  "data": {
    "condition": "{{ 'adult' if user.age >= 18 else 'minor' }}"
  }
}
```
Outputs: `adult`, `minor` handles.

## Example 2 – Switch/Case
```json
{
  "id": "router",
  "type": "conditional",
  "data": {
    "condition": "{{ status }}"
  }
}
```
If `status` resolves to `error`, execution continues via `error` handle; all other handles are bypassed.

## Implementation Status

- ✅ Extends `Node` base; overrides `process`.
- ✅ Accepts `condition` and `merge_strategy` in constructor / `data`.
- ✅ *jinja2* dependency included in `requirements.txt`.
- ✅ Bypass routing + propagation is implemented in `magic_agents/agt_flow.py` (`execute_graph`).

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `condition` | The Jinja2 condition template |
| `merge_strategy` | How inputs are merged (`flat` or `namespaced`) |
| `selected_handle` | The handle name selected by condition evaluation |
| `context_data` | The merged context used for evaluation |
| `output_handles` | Declared output handles (if specified) |
| `default_handle` | Fallback handle (if specified) |
| `merge_collisions` | Key collisions detected during flat merge |

## Error Handling

The node yields debug errors for:
- **ConfigurationError**: Missing or invalid `condition` template, invalid `merge_strategy`
- **InputError**: No inputs received on any handle
- **TemplateError**: Undefined variable in template
- **TemplateSyntaxError**: Invalid Jinja2 syntax in condition
- **TemplateEvaluationError**: Failed to evaluate condition template
- **EmptyHandleError**: Condition evaluated to empty string with no `default_handle`
- **GraphRoutingError**: Selected handle has no matching edge (runtime)

When an error occurs, the conditional emits a `__bypass_all__` signal to bypass all downstream nodes.

## Build-Time Validation

When `output_handles` is specified, the graph validator checks:
1. All declared handles have at least one outgoing edge
2. If `default_handle` is specified, it has a matching edge

Warnings are issued for:
- Edges with handles not in `output_handles` (potential typo)
- Conditionals without `output_handles` (runtime-only validation)

## Fan-Out Support

Multiple edges can share the same `sourceHandle`. All targets will execute when that handle is selected:

```json
{
  "edges": [
    {"source": "cond", "target": "processor_1", "sourceHandle": "parallel"},
    {"source": "cond", "target": "processor_2", "sourceHandle": "parallel"},
    {"source": "cond", "target": "processor_3", "sourceHandle": "parallel"}
  ]
}
```

When `parallel` is selected, all three processors execute concurrently.

## Merge Strategy Details

### Flat Merge (Default)

All inputs are merged into a single flat dictionary. Later inputs override earlier ones for colliding keys.

```python
# Input 1: {"status": "ok", "score": 95}
# Input 2: {"status": "error", "message": "API failed"}
# Result: {"status": "error", "score": 95, "message": "API failed"}
```

**Warning**: Key collisions are tracked in debug output. Consider using `namespaced` if inputs have overlapping keys.

### Namespaced Merge

Each input is stored under its handle name, preventing collisions:

```python
# Input on handle_input_0: {"status": "ok"}
# Input on handle_input_1: {"status": "error"}
# Result: {
#   "handle_input_0": {"status": "ok"},
#   "handle_input_1": {"status": "error"}
# }
```

Access namespaced values in condition:
```jinja2
{{ 'route_a' if handle_input_0.status == 'ok' else 'route_b' }}
```


## Best Practices

### 1. Always Declare `output_handles`
For production flows, explicitly list expected output handles for build-time validation:
```json
{
  "condition": "{{ status }}",
  "output_handles": ["success", "error", "timeout"]
}
```

### 2. Use `default_handle` for Robustness
Protect against unexpected values with a fallback:
```json
{
  "condition": "{{ category | default('') }}",
  "default_handle": "other"
}
```

### 3. Choose the Right Merge Strategy
| Scenario | Strategy | Reason |
|----------|----------|--------|
| Single input or unique keys | `flat` | Simpler syntax |
| Overlapping keys | `namespaced` | Prevents data loss |
| Need source clarity | `namespaced` | Explicit data origin |

### 4. Use Jinja2 Filters
Leverage filters for safer templates:
```jinja2
{{ status | default('unknown') }}
{{ score | int >= 18 }}
{{ items | length > 0 }}
```

### 5. Handle Convergence Properly
When branches merge back, use `| default(none)` in downstream templates:
```jinja2
{% set result_a = input_a | default(none) %}
{% if result_a is not none %}{{ result_a }}{% endif %}
```
