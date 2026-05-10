# `end`

## Purpose

Terminal node used to finalize a path.

## Runtime class

- `NodeEND`
- no dedicated runtime model in the current node map

## Default output

- `handle_end_output`

## Important behavior

- yields an empty `ChatCompletionModel`
- build automatically appends an edge from each `end` node to the internal `void` sink

## Example

```json
{
  "id": "finish",
  "type": "end"
}
```
