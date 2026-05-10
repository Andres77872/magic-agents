# `constant`

## Purpose

Yield a typed primitive value (`int`, `bool`, `str`, `float`) into the graph.

## Runtime class

- `NodeConstant`
- model: `ConstantNodeModel` with `value_type` and `value` fields

## Model fields

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `value_type` | `string` | Optional | `"str"` |
| `value` | any | Optional | `null` |

The value is coerced to the declared `value_type` at model construction time via `coerce_primitive_by_type`.

## Default output

- `handle_constant_output`

## Important behavior

- source node — no input handles
- resolves its typed value at model construction, before graph execution
- useful for injecting configuration constants, flags, or fixed parameters into the graph
- handle can be overridden via the `handles` dict (`output` or `value` key)

## Example

```json
{
  "id": "threshold",
  "type": "constant",
  "data": { "value_type": "float", "value": "0.85" }
}
```
