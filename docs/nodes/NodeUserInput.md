# NodeUserInput

Seeds the agent flow with the **user’s initial message**, files, and images. It also assigns unique `id_chat` and `id_thread` identifiers to the session.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeUserInput` |
| **Type key** | `user_input` |
| **Output handles** | `handle_user_message`, `handle_user_files`, `handle_user_images` |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The user’s message. Overwritten by `build()` when the graph is built. |
| `files` | `list[str]` | Optional file list attached to the message. |
| `images` | `list[str]` | Optional images (URLs / base64). |

## Example

```json
{
  "id": "user_input",
  "type": "user_input"
}
```

## Runtime Logic

```python
async def process(self, chat_log):
    if not chat_log.id_chat:
        chat_log.id_chat = uuid.uuid4().hex
    if not chat_log.id_thread:
        chat_log.id_thread = uuid.uuid4().hex

    yield self.yield_static(self._text, content_type=self.HANDLER_USER_MESSAGE)
    yield self.yield_static(self.files, content_type=self.HANDLER_USER_FILES)
    yield self.yield_static(self.images, content_type=self.HANDLER_USER_IMAGES)
```

- Ensures IDs exist.
- Emits three separate events so downstream nodes can choose what to consume.
