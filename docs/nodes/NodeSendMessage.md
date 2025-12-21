# NodeSendMessage

Emits **extra JSON payloads** (e.g.
references, UI-specific data) alongside normal chat content.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeSendMessage` |
| **Type key** | `send_message` |
| **Input handle** | `handle_send_extra` |
| **Output event** | `content` (`ChatCompletionModel` with optional `extras`) |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `message` | `str` | (Currently unused by implementation.) |
| `json_extras` | `str` | Text payload emitted in `ChatCompletionModel.choices[0].delta.content`. |

## Configurable Handles

Handle names can be customized via the `handles` field in `data`:

```json
{
  "id": "send_references",
  "type": "send_message",
  "data": {
    "json_extras": "Input is empty",
    "handles": {
      "send_extra": "my_extra_input"
    }
  }
}
```

| Handle Key | Aliases | Default Value | Description |
|------------|---------|---------------|-------------|
| `send_extra` | `extra` | `handle_send_extra` | Extra data input |

## Example

```json
{
  "id": "send_references",
  "type": "send_message",
  "data": {
    "json_extras": "Input is empty"
  }
}
```

## Runtime Logic (simplified)

```python
extra = self.get_input('handle_send_extra')
yield self.yield_static(ChatCompletionModel(..., extras=extra))
```

- Emits a `content` event (so it is forwarded to the caller/stream).
- If `handle_send_extra` is a JSON string, it is parsed; if it is a raw string, it is wrapped as `{"text": "<value>"}`.

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `message` | The configured message (currently unused) |
| `json_extras` | The extra content to emit |
