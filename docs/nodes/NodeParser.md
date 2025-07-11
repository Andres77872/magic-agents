# NodeParser

Renders a **Jinja2 template** using the aggregated `inputs` dictionary and outputs the rendered text.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeParser` |
| **Type key** | `parser` |
| **Input handle** | `handle_parser_input` (default) |
| **Output handle** | `handle_parser_output` |

## Data Fields (JSON spec)

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Jinja2 template string to render. |

## Example

```json
{
  "id": "format_results",
  "type": "parser",
  "data": {
    "text": "Found {{ handle_parser_input.results | length }} results for query '{{ handle_parser_input.query }}'"
  }
}
```

## Runtime Logic (simplified)

```python
async def process(self, chat_log):
    rendered = jinja_env.from_string(self.text).render(**self.inputs)
    yield self.yield_static(rendered)
```

- Can access any upstream handle via `self.inputs`.
- Supports all Jinja2 features: loops, conditionals, filters, macros.
- Commonly used to build dynamic prompts for `NodeLLM`, HTTP payloads for `NodeFetch`, etc.
