# `user_input`

## Purpose

Entry node that injects the initial user message and optional files, images, and extras.

## Runtime class

- `NodeUserInput`
- model: `UserInputNodeModel`

## Default outputs

- `handle_user_message`
- `handle_user_files`
- `handle_user_images`
- `handle_client_extras`

## Important behavior

- uses configured `session_id` when present and reuses backend-provided `chat_log.id_chat`
- when `session_required` is true and no session exists, leaves creation to the backend instead of generating a frontend/runtime UUID
- mirrors configured `session_id` into `chat_log.id_thread` when no thread ID is already present
- resolves text from `text`, `content`, or `message`
- passes `extras` through only when present

## Common fields

- `template`
- `text` / `content` / `message`
- `files`
- `images`
- `extras`
- `session_id`
- `session_required`
- `handles`

## Example

```json
{
  "id": "user-input",
  "type": "user_input",
  "data": {
    "text": "Hello"
  }
}
```
