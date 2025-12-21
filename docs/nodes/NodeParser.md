# NodeParser

Renders a **Jinja2 template** using the aggregated `inputs` dictionary and outputs the rendered text.

| Property | Value |
|----------|-------|
| **Python class** | `magic_agents.node_system.NodeParser` |
| **Type key** | `parser` |
| **Input handles** | Any (all `inputs` are available to the template). Values that look like JSON strings are decoded first. |
| **Output** | Default output (`end`, aliased by `edge.sourceHandle`) containing the rendered text |

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
    "text": "Found {{ results | length }} results for query '{{ query }}'"
  }
}
```

## Runtime Logic (simplified)

```python
async def process(self, chat_log):
    # Uses a shared Jinja2 Environment with extra filters (see `magic_agents.util.template_parser`)
    rendered = template_parse(template=self.text, params=self.inputs)
    yield self.yield_static(rendered)
```

- Can access any upstream handle via `self.inputs`.
- Supports all Jinja2 features: loops, conditionals, filters, macros.
- Commonly used to build dynamic prompts for `NodeLLM`, HTTP payloads for `NodeFetch`, etc.

## Debug Information

When `debug=True`, the following internal state is captured:

| Variable | Description |
|----------|-------------|
| `template` | The Jinja2 template (truncated to 500 chars) |
| `template_length` | Total length of the template |

## Error Handling

Template parsing uses `magic_agents.util.template_parser` which may raise exceptions on invalid Jinja2 syntax. Ensure your templates are valid before deploying.
