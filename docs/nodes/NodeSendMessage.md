# NodeSendMessage

Emits **extra JSON payloads** (e.g.
references, UI-specific data) alongside normal chat content.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeSendMessage` |
| **Type key** | `send_message` |
| **Input handle** | `handle_send_extra` |
| **Output handle** | `handle_generated_end` (same as `NodeLLM` default) |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `json_extras` | `str` | Jinja2 template rendered to JSON-able dict. Will become `ChatCompletionModel.extras`. |

## Example

```json
{
  "id": "send_references",
  "type": "send_message",
  "data": {
    "json_extras": "{{ handle_send_extra }}"
  }
}
```

## Runtime Logic (simplified)

```python
extra = self.get_input('handle_send_extra')
yield self.yield_static(ChatCompletionModel(..., extras=extra))
```

- Does **not** affect the textual response; only attaches metadata.
- Ideal for citations, images, or structured UI data.
