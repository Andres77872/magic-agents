# NodeEND

Marks the **termination** of a flow. It yields an (empty) `ChatCompletionModel` as an `end` event for bookkeeping/routing.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeEND` |
| **Type key** | `end` (user-defined) / `void` (internal auto-generated) |
| **Input handles** | any |
| **Output** | Default output (`end`, aliased by `edge.sourceHandle`) containing an (empty) `ChatCompletionModel` |

## Type Key Clarification

| Type | Usage | Description |
|------|-------|-------------|
| `end` | User-defined | Explicit terminal node in your flow definition |
| `void` | Internal | Automatically added by `build()` to collect unconnected outputs |

## Example

```json
{
  "id": "finish",
  "type": "end"
}
```

## Runtime Logic (simplified)

```python
async def process(self, chat_log):
    yield self.yield_static(ChatCompletionModel(id='', model='', choices=[ChoiceModel()]))
```

- `execute_graph` forwards only `content` events to the caller; `NodeEND` emits an `end` event (used for routing/bookkeeping).
- Most user-visible output comes from nodes that emit `content` events (e.g. `NodeLLM`, `NodeSendMessage`).
- A hidden *void* node is automatically added by `build()` to swallow unconnected outputs.

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `is_terminal_node` | Always `True` for END nodes |
