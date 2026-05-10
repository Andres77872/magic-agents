# `text`

## Purpose

Emit a static text payload into the graph.

## Runtime class

- `NodeText`
- model: `TextNodeModel`

## Default output

- `handle_text_output`

## Important behavior

- resolves content from `text` or `content`
- yields exactly one static output

## Example

```json
{
  "id": "welcome",
  "type": "text",
  "data": {"text": "Welcome!"}
}
```
