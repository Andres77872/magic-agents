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

### How graph execution actually routes data

Graph JSON is a set of `nodes` plus `edges`. Each edge says: take `sourceHandle` from the source node's outputs and place it into `targetHandle` on the target node's inputs. Nodes execute when the reactive runtime can satisfy their required inputs; downstream nodes receive data only through the handles connected by edges.

Practical rules:

- Use node docs as the source of truth for valid handles. A wrong handle is not a cosmetic issue — it changes whether data arrives.
- Normal nodes (`parser`, `llm`, `fetch` outside `tool_mode`, `python_exec` with `data.code`, etc.) pass values to downstream graph nodes through ordinary handles.
- Tool-provider nodes connected to an `llm` use `handle-tool-definition-N` target handles so `NodeLLM` can collect tool schemas/functions before calling magic-llm.
- `python_exec` with `data.code` is node-mode, not tool-provider mode. In that mode the build step preserves graph-routing handles and does not auto-assign LLM tool handles.
- Runtime-overridable LLM settings can be fed through dedicated input handles such as `handle-llm-temperature`, `handle-llm-max_tokens`, and `handle-llm-json_output`.

### Composition nodes

Use composition nodes deliberately; they are not interchangeable shortcuts:

- `loop` splits a list-like input into iterations. Put `iterate: true` on LLM nodes that must re-run per item, and route feedback back to the loop so it can aggregate results.
- `conditional` chooses an output handle based on its `condition`. Define `output_handles` and `default_handle` so non-selected branches can be bypassed predictably.
- `inner` embeds a subgraph (`magic_flow`/`flow`/`graph`/`subgraph`) and runs it as a child graph. Parent hooks are cloned into the child graph, while child debug events are forwarded back through yielded events rather than by propagating `debug_callback` directly.

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
