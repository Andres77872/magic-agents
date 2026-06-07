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
- ignores runtime inputs; use `parser` if the output must depend on upstream data

## Example

```json
{
  "id": "welcome",
  "type": "text",
  "data": {"text": "Welcome!"}
}
```
