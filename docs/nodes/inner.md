# `inner`

## Purpose

Execute a nested graph from inside a parent graph.

## Runtime class

- `NodeInner`
- model: `InnerNodeModel`

## Default input

- `handle_user_message`
- optional `handle_client_extras`

## Default outputs

- `handle_content_stream` — emitted during execution (real-time streaming chunks)
- `handle_execution_content` — emitted after execution completes (aggregated content)
- `handle_execution_extras` — emitted when inner graph produces extras (final extras)

## Important behavior

- accepts embedded graph config via `magic_flow`, `flow`, `graph`, or `subgraph`
- builds the child graph recursively during parent build
- forwards child streaming chunks in real time
- merges client extras with parent state exposure
- exposes full parent state as `parent_state` by default, or mapped keys via `parent_state_mapping`

## Gotchas

- malformed `magic_flow` does not crash build; execution emits configuration errors and bypasses downstream
- child `flow_state` is intentionally isolated

## Example

```json
{
  "id": "inner-step",
  "type": "inner",
  "data": {
    "magic_flow": {
      "type": "chat",
      "nodes": [{"id": "u", "type": "user_input"}, {"id": "e", "type": "end"}],
      "edges": []
    }
  }
}
```
