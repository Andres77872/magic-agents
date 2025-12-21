# NodeText

Emits a **static text string** into the flow.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeText` |
| **Type key** | `text` |
| **Input handles** | _none_ |
| **Output** | Default output (`end`, aliased by `edge.sourceHandle`) containing the configured `text` |

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

- No inputs; always produces exactly one output event (`end`).
- Commonly used to provide static data to downstream nodes (e.g., as an input into `NodeParser` or `NodeLLM`).

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `text` | The static text (truncated to 500 chars) |
| `text_length` | Total length of the text |
