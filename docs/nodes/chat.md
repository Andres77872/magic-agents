# `chat`

## Purpose

Build or reuse a `ModelChat` transcript for downstream LLM generation.

## Runtime class

- `NodeChat`
- model: `ChatNodeModel`

## Default inputs

- `handle-system-context`
- `handle_user_message`
- `handle_messages`
- `handle_user_files`
- `handle_user_images`

## Default output

- `handle_chat_output`

## Important behavior

- receives backend/model-provided chat history through fields such as `history_messages`
- `NodeChat` does not call `load_chat`; that parameter is not part of the `NodeChat` runtime contract
- can directly accept a full messages list on `handle_messages`
- supports image payload variants, with validation for mixed shapes

## Example

```json
{
  "id": "chat",
  "type": "chat"
}
```
