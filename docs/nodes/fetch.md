# `fetch`

## Purpose

Perform an HTTP request, or expose an HTTP request as a callable tool.

## Runtime class

- `NodeFetch`
- model: `FetchNodeModel`

## Default output

- `handle_fetch_output`

## Important behavior

- supports `url`/`endpoint`, `params`/`query`, `data`/`body`, `json_data`/`json_body`
- resolves `{{env.NAME}}` placeholders before execution
- templates URL, headers, params, and body values
- in `tool_mode`, does **not** execute immediately; it yields a `FetchToolCallable`

## Tool mode fields

- `tool_mode`
- `tool_name`
- `tool_parameters`

## Gotchas

- non-GET requests with no body produce an error payload
- if no inputs are set in normal mode, the node yields `{}` and returns

## Example

```json
{
  "id": "fetch-user",
  "type": "fetch",
  "data": {
    "url": "https://api.example.com/users/{{ handle_fetch_input }}",
    "method": "GET"
  }
}
```
