# NodeLLM

Generates **LLM completions** (streamed or batch) via `MagicLLM`.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeLLM` |
| **Type key** | `llm` |
| **Important input handles** | `handle-client-provider` (LLM client), `handle-chat`, `handle-system-context`, `handle_user_message` |
| **Output handle** | `handle_generated_content` (default) |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `stream` | `bool` | If `true`, yields chunks as the LLM streams. |
| `json_output` | `bool` | If `true`, parses the final generated text into JSON. |
| `iterate` | `bool` | Re-run on each Loop iteration (when placed inside a `NodeLoop`). |
| `temperature` / `top_p` | `float` | Standard sampling params. |
| `extra_data` | `dict` | Arbitrary provider-specific params. |

## Example

```json
{
  "id": "generate_response",
  "type": "llm",
  "data": {
    "stream": true,
    "temperature": 0.7,
    "max_tokens": 512
  }
}
```

## Runtime Logic (simplified)

```python
async def process(self, chat_log):
    client = self.get_input('handle-client-provider', required=True)
    chat = build_ModelChat_from_inputs(...)
    if stream:
        async for chunk in client.llm.async_stream_generate(chat, **params):
            yield self.yield_static(chunk)
    else:
        intention = await client.llm.async_generate(chat, **params)
        yield self.yield_static(intention)
    if self.json_output:
        self.generated = parse_json(self.generated)
    yield self.yield_static(self.generated)
```

- Combines system + user prompts and optional memory (chat).
- Supports JSON extraction heuristics for code-block or brace-matching.
- When `iterate=true`, response is **not cached** so each loop item causes a fresh call.
