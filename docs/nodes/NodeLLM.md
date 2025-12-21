# NodeLLM

Generates **LLM completions** (streamed or batch) via `MagicLLM`.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeLLM` |
| **Type key** | `llm` |
| **Important input handles** | `handle-client-provider` (LLM client), `handle-chat`, `handle-system-context`, `handle_user_message` |
| **Output events** | `content` (streamed `ChatCompletionModel` chunks), plus a final **default output** (`end`, aliased by `edge.sourceHandle`, commonly `handle_generated_end`) |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `stream` | `bool` | If `true`, yields chunks as the LLM streams. |
| `json_output` | `bool` | If `true`, parses the final generated text into JSON. |
| `iterate` | `bool` | Re-run on each Loop iteration (when placed inside a `NodeLoop`). |
| `temperature` / `top_p` | `float` | Standard sampling params. |
| `max_tokens` | `int` | Token limit (passed to the provider via `extra_data`). |
| `extra_data` | `dict` | Arbitrary provider-specific params. |

## Field Aliases

| Primary Field | Alias | Notes |
|---------------|-------|-------|
| `json_output` | `json_mode` | Both enable JSON parsing |
| `max_tokens` | `max_output_tokens` | Token limit |

## Configurable Handles

Handle names can be customized via the `handles` field in `data`:

```json
{
  "data": {
    "handles": {
      "client_provider": "my_custom_client_handle",
      "chat": "my_chat_handle",
      "system_context": "my_system_handle",
      "user_message": "my_message_handle"
    }
  }
}
```

| Handle Key | Aliases | Default Value | Description |
|------------|---------|---------------|-------------|
| `client_provider` | `client` | `handle-client-provider` | LLM client input |
| `chat` | - | `handle-chat` | Chat context input |
| `system_context` | `system` | `handle-system-context` | System prompt input |
| `user_message` | `message` | `handle_user_message` | User message input |

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
            yield self.yield_static(chunk, content_type='content')
    else:
        intention = await client.llm.async_generate(chat, **params)
        yield self.yield_static(intention, content_type='content')
    if self.json_output:
        self.generated = parse_json(self.generated)
    yield self.yield_static(self.generated)
```

- Combines system + user prompts and optional memory (chat).
- Supports JSON extraction heuristics for code-block or brace-matching.
- When `iterate=true`, response is **not cached** so each loop item causes a fresh call.

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `stream` | Whether streaming is enabled |
| `json_output` | Whether JSON parsing is enabled |
| `iterate` | Whether re-execution in loops is enabled |
| `generated` | Generated text (truncated to 500 chars) |
| `extra_data` | Additional LLM parameters |

## Error Handling

The node yields debug errors for:
- **InputError**: Missing required `handle_user_message` when no chat context
- **JSONParseError**: Failed to parse LLM output as JSON when `json_output=true`
- **JSONExtractionError**: No JSON content found in generated output
