# Graph format

This guide documents the JSON structure the current code actually consumes.

## Top-level graph shape

```json
{
  "type": "chat",
  "timeout": 60,
  "debug": true,
  "debug_config": {
    "preset": "verbose",
    "redact_sensitive": true
  },
  "contract_config": {
    "mode": "warn",
    "strict_runtime": false
  },
  "nodes": [],
  "edges": [],
  "master": "user-input"
}
```

## Top-level keys

| Key | Required | Current behavior |
| --- | --- | --- |
| `type` | no | Graph label. Defaults to `chat`. |
| `timeout` | no | Graph-level input-wait timeout in seconds. Defaults to `60`. |
| `debug` | no | Enables debug capture and debug event emission. |
| `debug_config` | no | Resolved into `DebugConfig.from_dict(...)`. |
| `contract_config` | no | Validation behavior. Supports `mode: off|shadow|warn|strict` plus `strict_runtime` (currently deferred). Defaults to `warn`. |
| `nodes` | yes | Node definitions. |
| `edges` | yes | Edge definitions. |
| `coordination` | no | Optional coordination policy (`CoordinationPolicy`). Validated at build by `require_valid_coordination()`; see [../nodes/llm.md](../nodes/llm.md) (`messaging`). |
| `persistence_enabled` | no | `true`/`false` overrides `RuntimeConfig.persistence_enabled` for this graph; `null` (default) defers to the runtime config. |
| `master` | no | Legacy field. Present in older examples, but not read by the current runtime. |

## Programmatic-only graph fields

`AgentFlowModel` also supports a `hooks` field for graph-level `FlowHooks` registration.

That field is meaningful when building graphs from Python objects, not as normal JSON transport. The wiki keeps it documented here because it affects the actual runtime model surface.

## Node shape

```json
{
  "id": "item-processor",
  "type": "llm",
  "position": {"x": 600, "y": 100},
  "data": {
    "stream": true,
    "iterate": true,
    "handles": {
      "output_generated": "my_generated_handle"
    }
  }
}
```

### Node fields

| Field | Required | Notes |
| --- | --- | --- |
| `id` | yes | Unique node ID. |
| `type` | yes | One of the built-in node type literals. |
| `position` | no | Optional UI metadata. Build can auto-assign positions. |
| `data` | no | Node-specific config. |

## Edge shape

```json
{
  "id": "e4-loop-item-to-llm",
  "source": "loop-node",
  "target": "item-processor",
  "sourceHandle": "handle_item",
  "targetHandle": "handle_user_message",
  "hooks": {
    "hook_node_id": "edge-hook-1",
    "timeout_override": 5,
    "enabled": true
  }
}
```

| Field | Required | Notes |
| --- | --- | --- |
| `id` | no | Optional identifier. `EdgeNodeModel` generates a unique `uuid.uuid4().hex` value if omitted. |
| `source` | yes | Source node ID. |
| `target` | yes | Target node ID. |
| `sourceHandle` | no | Output handle from the source node. |
| `targetHandle` | no | Input handle on the target node. Tool edges into an `llm` are auto-filled with `handle-tool-definition-N`; any other missing value defaults to `handle-void`, which routes the edge to the internal sink so it delivers nothing. |
| `hooks` | no | Optional edge-level hook config. If enabled and `hook_node_id` resolves to a `hook` node, the dispatcher invokes it when the edge is traversed. `hooks.hook_type` is deprecated and ignored. |
| `sourcePort` / `targetPort` | no | Optional port identity alongside the handle names (`EdgeNodeModel`). |

## Node type literals

Current accepted node types from the runtime factory / model layer:

```text
user_input, end, parser, client, llm, fetch, send_message,
chat, text, constant, void, loop, inner, conditional, python_exec, mcp,
hook, codex, memory, node_tool, skills
```

That is **21** implemented node types.

`node_tool` is backend-only schema authoring for client-executed tools. It emits `handle-tool-definition` and must be connected to an `llm` tool input. Frontend/UI support is a separate repository rollout requirement before public graph authoring.

## Concrete examples

### 1. Conditional branch from a browsing graph

```json
{
  "id": "conditional-queries-check",
  "type": "conditional",
  "data": {
    "condition": "{{ 'has_queries' if queries is defined and queries is iterable and queries|list|length > 0 else 'no_queries' }}",
    "merge_strategy": "flat",
    "output_handles": ["has_queries", "no_queries"],
    "default_handle": "no_queries"
  }
}
```

### 2. Loop + iterate example

```json
{
  "id": "item-processor",
  "type": "llm",
  "data": {
    "stream": true,
    "iterate": true,
    "max_tokens": 100,
    "temperature": 0.7
  }
}
```

### 3. Client secrets via build-time env resolution

```json
{
  "id": "llm-client",
  "type": "client",
  "data": {
    "engine": "openai",
    "model": "gpt-4o-mini",
    "api_info": {
      "api_key": "{{env.OPENAI_API_KEY}}",
      "base_url": "https://api.openai.com/v1"
    }
  }
}
```

## Important format nuances

### Handle overrides live inside `data.handles`

Many nodes let you override default input/output handle names.

### Node models accept aliases

Examples:

- `user_input.content` and `user_input.message` alias `text`
- `text.content` aliases `text`
- `parser.content` and `parser.template` alias `text`
- `client.provider` aliases `engine`
- `client.model_name` aliases `model`
- `fetch.endpoint` aliases `url`
- `fetch.query` aliases `params`
- `llm.json_mode` aliases `json_output`
- `llm.max_output_tokens` aliases `max_tokens`
- `send_message.content` aliases `message`
- `send_message.extras` aliases `json_extras`
- `inner.flow`, `inner.graph`, and `inner.subgraph` alias `magic_flow`

### Hook-related graph fields are partial by design

- graph-level `hooks` lives on `AgentFlowModel` and is primarily a programmatic API
- edge-level `hooks` is part of `EdgeNodeModel` and can be represented in graph data
- `contract_config.mode = "strict"` exists on the model surface, but parts of strict enforcement are still deferred in validation/runtime follow-up work

### Parser input names are conventional, not fixed

Many examples use `handle_parser_input_0`, `handle_parser_input_1`, and so on. The runtime itself just consumes whatever handles are routed into the parser node.

## Example flow shapes worth reading

Complete graphs used by the test suite (they build against the current code):

- [../../tests/fixtures/graphs/browsing.json](../../tests/fixtures/graphs/browsing.json): conditional routing and parallel fetches
- [../../tests/fixtures/graphs/deep_research.json](../../tests/fixtures/graphs/deep_research.json): fetch tools in an agent loop
- [../../tests/fixtures/graphs/image.json](../../tests/fixtures/graphs/image.json): Text → Chat system context
