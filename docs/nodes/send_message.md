# `send_message`

## Purpose

Emit a user-facing `ChatCompletionModel` payload, optionally with extras.

## Runtime class

- `NodeSendMessage`
- model: `SendMessageNodeModel`

## Default input

- `handle_send_extra`

## Outputs

- `content` — immediate user-facing stream event (system streaming)
- `handle_message_output` — canonical routed output for downstream nodes

## Important behavior

- emits twice: first on `content` for immediate user stream, then on `handle_message_output` for graph routing
- `handle_send_extra` accepts JSON string or dict; parsed automatically in process()
- uses `message` (or alias `content`) as the user-facing text
- when `message` is empty, uses `json_extras` (or alias `extras`) as a compatibility fallback for older graphs
- is one of the few nodes that intentionally emits on the system streaming event type `content`

## Example

```json
{
  "id": "send",
  "type": "send_message",
  "data": {"message": "Finished"}
}
```
