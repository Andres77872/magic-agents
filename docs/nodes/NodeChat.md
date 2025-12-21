# NodeChat

Maintains **conversation context** and memory via `ModelChat`.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeChat` |
| **Type key** | `chat` |
| **Input handles** | `handle-system-context`, `handle_user_message`, `handle_messages`, `handle_user_files`, `handle_user_images` |
| **Output handle (recommended edge.sourceHandle)** | `handle-chat` |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `message` | `str` | Initial message. Overwritten by `build()` when the graph is built. |
| `memory` | `dict` | Optional settings used when creating/loading the chat. Common keys: `stm`, `ltm`, `max_input_tokens`. |

## Configurable Handles

Handle names can be customized via the `handles` field in `data`:

```json
{
  "data": {
    "handles": {
      "system_context": "my_system_handle",
      "user_message": "my_message_handle",
      "messages": "my_messages_handle",
      "user_files": "my_files_handle",
      "user_images": "my_images_handle"
    }
  }
}
```

| Handle Key | Aliases | Default Value | Description |
|------------|---------|---------------|-------------|
| `system_context` | `system` | `handle-system-context` | System prompt input |
| `user_message` | `message` | `handle_user_message` | User message input |
| `messages` | - | `handle_messages` | Direct message list |
| `user_files` | `files` | `handle_user_files` | Attached files |
| `user_images` | `images` | `handle_user_images` | Attached images |

## Example

```json
{
  "id": "chat_memory",
  "type": "chat",
  "data": {
    "memory": { "stm": 8, "ltm": 32 }
  }
}
```

## Runtime Highlights

```python
if msgs := self.get_input('handle_messages'):
    self.chat.messages = msgs
else:
    if sys := self.get_input('handle-system-context'):
        self.chat.set_system(sys)
    if user := self.get_input('handle_user_message'):
        self.chat.add_user_message(user)

yield self.yield_static(self.chat)
```

- Combines new inputs with stored messages.
- Returns a `ModelChat` ready for downstream `NodeLLM` nodes (wire it with an edge using `sourceHandle: "handle-chat"` and `targetHandle: "handle-chat"`).

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `messages_count` | Number of messages in chat |
| `has_system_message` | Whether system message is set |
| `memory` | Memory configuration dict |

## Error Handling

The node yields debug errors for:
- **ValidationError**: When `UserImage` and `UserFile` formats are mixed (must be all single strings or all pairs)
