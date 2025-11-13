# NodeConditional

Implements **branching logic** (`if/else` and `switch`-style) inside a Magic-Agents flow.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeConditional` |
| **Type key** | `conditional` |
| **Input handles** | `handle_input` (required) – JSON / string used as *context* for evaluating the condition. |
| **Data fields (JSON spec)** | `condition` – *jinja2* template that resolves to the **name of the outgoing handle** to continue with.<br>Examples:<br>• `{{ "true" if user.age > 18 else "false" }}` → classic *if/else*.<br>• `{{ status }}` where `status` may be `success`, `error`, etc. → *switch/case*. |
| **Dynamic output handles** | *Any* handle names referenced by `condition` (e.g. `true`, `false`, `success`, `timeout`, …). |

## Runtime Behaviour

1. **Gather context** – read `handle_input` and `json.loads` if necessary to obtain `ctx` (dict / primitive).
2. **Render** the `condition` template using *jinja2* with `ctx` in scope.
   ```python
   env = jinja2.Environment()
   selected = env.from_string(condition).render(**ctx)
   selected = str(selected).strip()
   ```
3. **Validate** – if `selected` is not one of the node's declared outputs raise an execution error.
4. **Emit chosen path** – yield
   ```python
   yield self.yield_static(ctx, content_type=selected)
   ```
5. **Mark bypass** – For every *other* outgoing edge from this node, set `bypass=True` on that edge **and** recursively propagate to all descendant nodes *unless* a descendant still has at least one non-bypassed parent.
   *Algorithm sketch*
   ```text
   dfs(node):
       if node already bypassed: return
       if all parents bypassed:
           node.bypass = True
           for child in node.children:
               dfs(child)
   ```

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
- ✅ Unit tests provided in `test/test_conditional_flows.py` for branching + bypass propagation.
- ✅ Engine executor honours `bypass` flag in `agt_flow.py` (execute_graph function).
