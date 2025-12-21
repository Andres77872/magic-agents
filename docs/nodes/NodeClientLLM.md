# NodeClientLLM

Provides a configured **`MagicLLM` client** instance to downstream `NodeLLM` nodes.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeClientLLM` |
| **Type key** | `client` |
| **Input handles** | _none_ |
| **Output handle (recommended edge.sourceHandle)** | `handle-client-provider` |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `engine` | `str` | Provider identifier, e.g. `openai`, `anthropic`. |
| `model` | `str` | Model name, e.g. `gpt-4o-mini`. |
| `api_info` | `dict \| str \| null` | API config (dict or JSON string). Example: `{ "api_key": "sk-...", "base_url": "https://api.openai.com/v1" }`. |
| `extra_data` | `dict` | Additional provider-specific params. |

## Field Aliases

| Primary Field | Alias | Notes |
|---------------|-------|-------|
| `engine` | `provider` | LLM provider name |
| `model` | `model_name` | Model identifier |
| `api_info` | `config`, `credentials` | API configuration |

## Supported Engines

| Engine | Description |
|--------|-------------|
| `openai` | OpenAI API |
| `anthropic` | Anthropic API |
| `google` | Google AI API |
| `azure` | Azure OpenAI |
| `amazon` | Amazon Bedrock |
| `cohere` | Cohere API |
| `cloudflare` | Cloudflare Workers AI |

## Example

```json
{
  "id": "llm_client",
  "type": "client",
  "data": {
    "engine": "openai",
    "model": "gpt-4o-mini",
    "api_info": {
      "api_key": "sk-...",
      "base_url": "https://api.openai.com/v1"
    }
  }
}
```

## Runtime Logic (simplified)

```python
api_info = json.loads(data.api_info) if isinstance(data.api_info, str) else data.api_info
args = {
    'engine': data.engine,
    'model': data.model,
    **api_info,
    **data.extra_data
}
if 'api_key' in args:
    args['private_key'] = args['api_key']
self.client = MagicLLM(**args)

yield self.yield_static(self.client)
```

- Converts `api_key` ➜ `private_key` for `MagicLLM` compatibility.
- Downstream `NodeLLM` consumes the client via `handle-client-provider`.

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `engine` | Configured LLM provider |
| `model` | Configured model name |
| `client_initialized` | Whether client was created successfully |
| `init_error` | Error message if initialization failed |
| `init_error_type` | Exception type if initialization failed |

## Error Handling

The node yields debug errors for:
- **ConfigurationError**: Failed to initialize MagicLLM client (invalid credentials, missing params, etc.)
