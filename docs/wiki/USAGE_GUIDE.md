# Usage guide

This guide is about authoring graphs that match the current runtime.

## Authoring rules that save you pain

1. Always include exactly one `user_input` node.
2. Treat handles as explicit API contracts between nodes.
3. Use `output_handles` and `default_handle` on conditionals.
4. Use `iterate: true` on `llm` nodes that must re-run inside a loop.
5. Use `{{env.NAME}}` only for build-time secrets/config values.
6. Do not rely on `master`; the current runtime ignores it.

## Common flow shapes

### Linear

`user_input -> parser -> client -> llm -> end`

### Search-assisted

`user_input -> parser -> llm(json_output) -> conditional -> fetch -> parser -> llm(stream) -> send_message/end`

### Loop processing

`text(list json) -> loop -> llm(iterate=true) -> loop -> end`

### Nested graph

`user_input -> inner -> parser/llm/end`

## Pattern: tool-capable LLM

You can feed an `llm` with tools from:

- `fetch` with `tool_mode: true`
- `python_exec`
- `mcp`
- task subagents loaded through MagicLLM when enabled

The build step can auto-fill missing tool input handles on edges from those nodes into the LLM.

## Pattern: conditional with safe fallback

Good:

```json
{
  "type": "conditional",
  "data": {
    "condition": "{{ 'adult' if age >= 18 else 'minor' }}",
    "output_handles": ["adult", "minor"],
    "default_handle": "minor"
  }
}
```

Why: build-time validation can now tell you when an edge is missing.

## Pattern: debug-first authoring

For new graphs, start with:

```json
{
  "debug": true,
  "debug_config": {
    "preset": "verbose",
    "redact_sensitive": true
  }
}
```

Then tighten it later.

## Anti-patterns

- depending on the legacy `master` field
- relying on undocumented handle defaults across non-trivial graphs
- assuming all cycles are rejected up front
- expecting `loop` to behave like a generic while-loop
- mixing stale README examples with current routing semantics without checking the node docs

## Next reads

- [GRAPH_FORMAT.md](GRAPH_FORMAT.md)
- [HANDLES_AND_ROUTING.md](HANDLES_AND_ROUTING.md)
- [../nodes/loop.md](../nodes/loop.md)
- [../nodes/conditional.md](../nodes/conditional.md)
