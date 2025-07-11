# NodeFetch

Performs an **HTTP request** (GET/POST/etc.) and yields the parsed JSON/text response.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeFetch` |
| **Type key** | `fetch` |
| **Input handle** | `handle_fetch_input` |
| **Output handles** | `handle_response_json`, `handle_response_text` (impl-specific) |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str` | Endpoint to call (Jinja2-templated). |
| `method` | `str` | `GET`, `POST`, etc. |
| `headers` | `dict` | HTTP headers (templated). |
| `json_data` / `body` | `dict` / `str` | Request payload (templated). |
| `timeout` | `float` | Seconds before abort. |

## Example

```json
{
  "id": "search_api",
  "type": "fetch",
  "data": {
    "url": "https://api.example.com/search",
    "method": "POST",
    "headers": {
      "Authorization": "Bearer {{ token }}",
      "Content-Type": "application/json"
    },
    "json_data": {
      "q": "{{ handle_fetch_input }}"
    }
  }
}
```

## Runtime Logic (simplified)

```python
async def process(self, chat_log):
    url = jinja_render(self.url, **self.inputs)
    headers = jinja_render_dict(self.headers, **self.inputs)
    body = jinja_render_json(self.json_data, **self.inputs)
    resp = await httpx_async_request(method, url, headers=headers, json=body)
    yield self.yield_static(resp.json())
```

- Template rendering allows dynamic URLs/bodies.
- Response is auto-parsed as JSON when possible.
