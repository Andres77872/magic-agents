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

- creates `chat_log.id_chat` and `chat_log.id_thread` when missing
- resolves text from `text`, `content`, or `message`
- passes `extras` through only when present

## Common fields

- `text` / `content` / `message`
- `files`
- `images`
- `extras`
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
