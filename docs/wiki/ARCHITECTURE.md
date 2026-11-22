# Architecture

This page documents the current build-side architecture from the codebase.

## Main modules

| Path | Role |
| --- | --- |
| `magic_agents/agt_flow.py` | Graph normalization, validation, node factory, recursive inner-flow build |
| `magic_agents/node_system/` | Node implementations and graph sorting helpers |
| `magic_agents/models/factory/` | Pydantic-style graph/node/edge models |
| `magic_agents/execution/` | Reactive runtime, dispatchers, loop execution, conditional bypass |
| `magic_agents/hooks/` | Flow hook protocols, runtime config, execution-scoped registry |
| `magic_agents/debug/` | Debug config, observers, emitters, execution summaries |
| `magic_agents/mcp/` | MCP session management, discovery, namespacing, dispatch |
| `magic_agents/util/` | Graph validation, handle registry, env resolution, template helpers |

## Build pipeline

```mermaid
flowchart TD
  A[raw graph dict] --> B[normalize content wrapper]
  B --> C[require_valid_coordination]
  C --> D[private deep copy]
  D --> E[resolve env placeholders in MCP connection fields]
  E --> F[drop JSON 'hooks' key with a warning]
  F --> G[validate_skills_topology]
  G --> H[validate_graph]
  H --> I[filter invalid/self-loop edges]
  I --> J[auto-assign tool handles + edge ids]
  J --> K[sort nodes and edges]
  K --> L[inject internal void node; missing targetHandle -> handle-void]
  L --> M[instantiate node classes]
  M --> N[build inner graphs]
  N --> O[ConditionalEdgeValidator + validate_edge_handles]
  O --> P[instantiate AgentFlowModel]
  P --> Q[contract validation chain unless mode is off]
  Q --> R[attach validation report]
```

## Build details worth knowing

### 1. Input normalization

`build()` accepts either:

- a flat graph dict with `nodes` and `edges`
- a nested wrapper with `content.nodes` and `content.edges`

### 2. Environment placeholder resolution

`{{env.VAR_NAME}}` is resolved only in connection settings: `client` credentials
(`api_info`, `extra_data`), `fetch` requests (URL, headers, params, body) and `mcp`
server settings (`command`, `args`, `env`, `cwd`, `url`, `headers`). Anywhere else
(text, prompts, templates, constants) it is kept as literal text, so a graph author
cannot read the server's environment through a node's output.

Other Jinja-like placeholders such as `{{ handle_user_message }}` are preserved for runtime templating.

### 3. Structural validation

`validate_graph()` currently enforces:

- exactly one `user_input` node
- no duplicate edges with the same `(source, target, sourceHandle, targetHandle)`
- existing source/target node references
- self-loop detection
- Fetch request templates (`FetchTemplateValidationError`)
- Hook `function_template` code (`HookValidationError`)
- edge `hooks.hook_node_id` references an existing `hook` node, with at most one enabled edge per Hook
- lifecycle Hook bindings (`validate_lifecycle_hooks`)

Those are only the first-pass structural checks. After node/edge instantiation, the build path also runs conditional validation plus the broader contract-validation chain documented in [VALIDATION.md](VALIDATION.md).

### 4. Tool-edge backfilling

Before node creation, `_assign_tool_handles()` inspects edges from tool-capable nodes (`fetch` in `tool_mode`, tool-mode `python_exec`, `inner` in `tool_mode`, `mcp`, and `node_tool`) into `llm` nodes.

It auto-fills:

- `sourceHandle` based on the provider node
- `targetHandle` as `handle-tool-definition-N` when missing

Exception: `python_exec` runs as a normal graph node when `tool_mode` is `false`, or when it has `data.code` and `tool_mode` is not `true`; `_assign_tool_handles()` then skips automatic tool-handle assignment and existing handles are preserved so the result routes through graph edges. Without `code` it is a model-written-code tool; with `code` and `tool_mode: true` it is a fixed tool.

### 5. Sorting and positioning

`sort_nodes()` uses `perform_topological_sort()` plus automatic positions for nodes with missing or default positions.

Important nuance: `perform_topological_sort()` falls back to graph insertion order when `networkx.topological_sort()` is unfeasible. That is why cycle behavior needs careful documentation; see [VALIDATION.md](VALIDATION.md).

### 6. Internal `void` sink injection

Build always appends an internal `void` node and rewrites edges targeting `handle-void` to that node. An edge that still has no `targetHandle` after tool-handle backfilling defaults to `handle-void`, so it is routed to the sink and delivers nothing.

`end` nodes also get an automatic edge to the internal sink.

### 7. Node factory

`create_node()` maps JSON node `type` to:

- a runtime node class
- an optional model class for validation/coercion

This is also where aliases such as `json_mode -> json_output` or `provider -> engine` get normalized through node models.

The runtime factory currently maps **21** node types, including `codex`, `memory`, `node_tool` and `skills`.

### 8. Recursive inner graphs

`inner` nodes carry embedded `magic_flow` JSON. During build, each valid nested graph is built recursively and attached as `node.inner_graph`.

## Runtime-relevant architecture pieces

Even though execution is covered in detail elsewhere, these components are part of the architecture boundary:

- `GraphEventDispatcher` — dependency maps, input trackers, bypass propagation
- `NodeInputTracker` — readiness/bypass bookkeeping per node, keyed by edge ID for correct fan-in tracking
- `ConditionalRouting` protocol — runtime contract for conditional-like nodes
- `GraphDebugFeedback` — whole-graph execution summary
- `RuntimeConfig` / `HookRegistry` — graph/node lifecycle hook registration and invocation
- `ObserverRegistry` — per-execution debug observer resolution (`NullObserver`, `DefaultObserver`, `CompositeObserver`)
- `CallbackEmitter` — module-level callback bridge for selected persisted execution/debug output consumers; see [../hooks/CALLBACK_EMITTER.md](../hooks/CALLBACK_EMITTER.md)

See [EXECUTION_MODEL.md](EXECUTION_MODEL.md).
