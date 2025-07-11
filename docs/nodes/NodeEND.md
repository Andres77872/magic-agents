# NodeEND

Marks the **termination** of a flow. It does not generate new content except a blank `ChatCompletionModel` chunk to ensure streams close cleanly.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeEND` |
| **Type key** | `end` (user-visible) / `void` (internal) |
| **Input handles** | any |
| **Output handle** | `handle_generated_end` |

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

- Emits an empty chunk so streaming clients detect completion.
- A hidden *void* node is automatically added by `build()` to swallow unconnected outputs.
