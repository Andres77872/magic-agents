# NodeLoop

Iterates over a **list of items** and aggregates per-item results.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeLoop` |
| **Type key** | `loop` |
| **Input handles** | `handle_list` (required), `handle_loop` (optional) |
| **Output handles** | `handle_item` (per element), `handle_end` (final aggregation) |

## Example

```json
{
  "id": "iterate_items",
  "type": "loop"
}
```

The node expects an upstream edge to provide `handle_list`, e.g. from `NodeFetch` or `NodeParser`.

## Runtime Logic (simplified)

```python
raw = self.get_input('handle_list', required=True)
items = json.loads(raw) if isinstance(raw, str) else raw
for item in items:
    yield self.yield_static(item)          # handle_item
agg = self.inputs.get('handle_loop', [])
yield self.yield_static(agg)               # handle_end
```

- Per-iteration children can return results on `handle_loop`; these are **collected** and emitted at the end.
- `iterate=true` on downstream `NodeLLM` enables per-item generation.
