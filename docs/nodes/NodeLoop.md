# NodeLoop

Iterates over a **list of items** and aggregates per-item results.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeLoop` |
| **Type key** | `loop` |
| **Input handles** | `handle_list` (required), `handle_loop` (optional) |
| **Output handles** | `handle_item` (per element), `handle_end` (final aggregation) |

## Configurable Handles

Handle names can be customized via the `handles` field in `data`:

```json
{
  "id": "iterate_items",
  "type": "loop",
  "data": {
    "handles": {
      "input_list": "my_list_input",
      "input_loop": "my_loop_feedback",
      "output_item": "my_item_output",
      "output_end": "my_aggregation"
    }
  }
}
```

| Handle Key | Aliases | Default Value | Description |
|------------|---------|---------------|-------------|
| `input_list` | `list` | `handle_list` | Input list to iterate |
| `input_loop` | `loop` | `handle_loop` | Per-iteration feedback |
| `output_item` | `item` | `handle_item` | Current item output |
| `output_end` | `end` | `handle_end` | Aggregated results |

## Example

```json
{
  "id": "iterate_items",
  "type": "loop"
}
```

The node expects an upstream edge to provide `handle_list`, e.g. from `NodeFetch` or `NodeParser`.

## Runtime Logic (simplified)

When a graph contains a `loop` node, `execute_graph()` delegates to a specialized executor (`execute_graph_loop`) which:

- Reads `handle_list` as a JSON string or Python list
- For each item:
  - Publishes the current item on the loop node as `handle_item`
  - Runs the sub-graph connected to `handle_item`
  - Collects feedback into `handle_loop`
- After all items:
  - Publishes the aggregated list as `handle_end`
  - Runs the end-graph connected to `handle_end`

- Per-iteration children can return results on `handle_loop`; these are **collected** and emitted at the end.
- `iterate=true` on downstream `NodeLLM` enables per-item generation.

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `iterate` | Always `True` for loop nodes |
| `input_handle_list` | Configured list input handle |
| `input_handle_loop` | Configured loop feedback handle |
| `output_handle_item` | Configured item output handle |
| `output_handle_end` | Configured end output handle |

## Error Handling

The node yields debug errors for:
- **InputError**: Required `handle_list` input not provided
- **JSONParseError**: Input string is not valid JSON
- **ValidationError**: Input is not a list type
