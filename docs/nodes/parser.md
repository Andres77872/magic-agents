# `parser`

## Purpose

Render a Jinja2 template against the node inputs.

## Runtime class

- `NodeParser`
- model: `ParserNodeModel`

## Default output

- `handle_parser_output` — canonical routed output for downstream nodes

## Input handles

Parser nodes accept arbitrary input handles via template references (e.g., `{{ handle_parser_input_0 }}`). The frontend may declare `inputs` arrays for UX purposes, but these are stripped before sending to backend.

## Important behavior

- parses string inputs as JSON when possible before rendering
- resolves template from `text`, `content`, or `template`
- works with any routed input handles, not just parser-specific names

## Example

```json
{
  "id": "format",
  "type": "parser",
  "data": {
    "template": "Hello {{ handle_user_message }}"
  }
}
```
