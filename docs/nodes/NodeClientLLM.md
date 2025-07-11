# NodeClientLLM

Provides a configured **`MagicLLM` client** instance to downstream `NodeLLM` nodes.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeClientLLM` |
| **Type key** | `client` |
| **Input handles** | _none_ |
| **Output handle** | `handle-client-provider` |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `engine` | `str` | Provider identifier, e.g. `openai`, `anthropic`. |
| `model` | `str` | Model name, e.g. `gpt-4o-mini`. |
| `api_info` | `dict` | `{ "api_key": "sk-...", "base_url": "https://api.openai.com/v1" }`. |
| `extra_data` | `dict` | Additional provider-specific params. |

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
