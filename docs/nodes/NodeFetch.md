# NodeFetch

Performs an **HTTP request** (GET/POST/etc.) and yields the parsed JSON/text response.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeFetch` |
| **Type key** | `fetch` |
| **Input handles** | Any (all `inputs` are available for templating) |
| **Output** | Default output (`end`, aliased by `edge.sourceHandle`) containing the parsed JSON response |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str` | Endpoint to call (Jinja2-templated). |
| `method` | `str` | `GET`, `POST`, etc. |
| `headers` | `dict \| str` | HTTP headers (dict or JSON string). |
| `data` | `dict \| str` | Request payload sent as `data` (templated). |
| `json_data` | `dict \| str` | Request payload sent as `json` (templated). |
| `params` | `dict \| str` | URL query parameters. |
| `body` | `dict \| str` | Request body (alternative to `data`). |

## Field Aliases

| Primary Field | Alias | Notes |
|---------------|-------|-------|
| `url` | `endpoint` | API endpoint URL |
| `params` | `query` | URL query parameters |
| `json_data` | `json_body` | JSON request body |

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
      "q": "{{ query }}"
    }
  }
}
```

## Runtime Logic (simplified)

```python
async def process(self, chat_log):
    # URL is rendered with Jinja2 using current inputs
    rendered_url = Template(self.url).render(self.inputs)
    # `data` / `json_data` are also rendered via Jinja2 before sending
    response_json = await aiohttp_request(rendered_url, ...)
    yield self.yield_static(response_json)
```

- The current implementation always calls `response.json()` (so endpoints should return JSON).
- If **all inputs are empty**, the node skips the request and yields `{}`.

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `url` | Configured URL template |
| `method` | HTTP method |
| `headers` | Request headers |
| `body` | Request body (if set) |
| `json_data` | JSON body (if set) |

## Error Handling

The node yields debug errors for:
- **TemplateError**: URL templating failed with Jinja2
- **HTTPError**: HTTP request failed with status code
- **NetworkError**: Network connection failed
- **UnexpectedError**: Other unexpected errors
