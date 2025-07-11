# NodeText

Emits a **static text string** into the flow.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeText` |
| **Type key** | `text` |
| **Input handles** | _none_ |
| **Output handle** | `handle_void` (default) |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The literal string to emit. |

## Example

```json
{
  "id": "welcome_text",
  "type": "text",
  "data": {
    "text": "Welcome! I'm processing your request..."
  }
}
```

## Runtime Logic

```python
async def process(self, chat_log):
    yield self.yield_static(self._text)
```

- No inputs; always produces exactly one content chunk.
- Commonly used for placeholder messages, system prompts, or fixed responses.
