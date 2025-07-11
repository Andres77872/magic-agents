# NodeChat

Maintains **conversation context** and memory via `ModelChat`.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeChat` |
| **Type key** | `chat` |
| **Input handles** | `handle-system-context`, `handle_user_message`, `handle_messages`, `handle_user_files`, `handle_user_images` |
| **Output handle** | `handle_chat` |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `memory` | `dict` | `{ "stm": 8, "ltm": 32 }` – short-term/long-term memory message limits.
| `max_input_tokens` | `int` | For LLM context window management. |

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
- Returns a `ModelChat` ready for downstream LLM nodes.
