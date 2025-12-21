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

## Configurable Handles

Output handle names can be customized via the `handles` field in `data`:

```json
{
  "id": "user_input",
  "type": "user_input",
  "data": {
    "handles": {
      "user_message": "my_message_output",
      "user_files": "my_files_output",
      "user_images": "my_images_output"
    }
  }
}
```

| Handle Key | Aliases | Default Value | Description |
|------------|---------|---------------|-------------|
| `user_message` | `message` | `handle_user_message` | User text output |
| `user_files` | `files` | `handle_user_files` | Files output |
| `user_images` | `images` | `handle_user_images` | Images output |

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

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `text` | User's text message |
| `files` | List of attached files |
| `images` | List of attached images |
