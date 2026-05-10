# Graph JSON Contract

## Purpose

This document defines the authoritative JSON contract for agent flow graphs in `magic-agents`. The backend validates all graphs against this contract before execution. Frontend (`magic-ui`) must serialize graphs matching this structure.

**Policy**: Clean-break validation. Legacy graphs with invalid structure fail with clear errors. No backward-compatibility shim.

---

## Graph-Level Structure

A valid graph JSON contains:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `nodes` | `array[NodeJSON]` | **Required** | Array of node definitions |
| `edges` | `array[EdgeJSON]` | **Required** | Array of edge definitions |

Optional fields:
- `type`: `"graph"` (optional metadata)
- `debug`: `boolean` (optional, runtime debug mode)
- `debug_config`: `object` (optional, debug configuration)
- `timeout`: `number` (optional, graph-level timeout in seconds, default `60.0`)
- `contract_config`: `object` (optional, validation mode: `"off"`, `"shadow"`, `"warn"`, `"strict"`; default `"warn"`)
- `hooks`: `FlowHooks` (optional, programmatic graph-level hook protocol — not serializable via JSON; injected at build time; see [hooks/README.md](hooks/README.md))

### Example

```json
{
  "nodes": [
    { "id": "input", "type": "user_input", "data": {} },
    { "id": "output", "type": "end", "data": {} }
  ],
  "edges": [
    { "source": "input", "target": "output", "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input" }
  ]
}
```

---

## Node JSON Structure

Each node in `nodes` array must contain:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `string` | **Required** | Unique non-empty identifier |
| `type` | `string` | **Required** | Canonical type key (17 types) |
| `data` | `object` | **Required** | Node-specific configuration (may be empty `{}`) |
| `position` | `object` | Optional | Canvas position `{x, y}` (default `{x:0, y:0}`) |

### Validation Rules

1. **`id` must be unique** — duplicate IDs rejected with error
2. **`id` must be non-empty string** — empty IDs rejected
3. **`type` must be canonical** — unknown types rejected with available types list
4. **`data` fields must match node model** — unknown fields rejected (see `extra='forbid'`)
5. **UI-only fields are rejected** — `measured`, `inputs`, `selected`, `dragging` are not permitted

---

## Canonical Node Types (17 Types)

| Type Key | Node Class | Model Class | Description |
|----------|------------|-------------|-------------|
| `user_input` | `NodeUserInput` | `UserInputNodeModel` | User message/file/image input |
| `end` | `NodeEND` | `EndNodeModel` | Terminal output node |
| `parser` | `NodeParser` | `ParserNodeModel` | Jinja2 template renderer |
| `client` | `NodeClientLLM` | `ClientNodeModel` | LLM client configuration |
| `llm` | `NodeLLM` | `LlmNodeModel` | LLM generation node |
| `fetch` | `NodeFetch` | `FetchNodeModel` | HTTP request node |
| `send_message` | `NodeSendMessage` | `SendMessageNodeModel` | User-facing message output |
| `chat` | `NodeChat` | `ChatNodeModel` | Chat memory node |
| `text` | `NodeText` | `TextNodeModel` | Static text node |
| `constant` | `NodeConstant` | `ConstantNodeModel` | Typed primitive value source |
| `void` | `NodeEND` | `None` | Silent terminal (no output) |
| `loop` | `NodeLoop` | `LoopNodeModel` | Iteration control node |
| `inner` | `NodeInner` | `InnerNodeModel` | Subgraph execution node |
| `conditional` | `NodeConditional` | `ConditionalNodeModel` | Branch routing node |
| `python_exec` | `NodePythonExec` | `PythonExecNodeModel` | Python execution node |
| `mcp` | `NodeMcp` | `McpNodeModel` | MCP tool integration node |
| `hook` | `NodeHook` | `HookNodeModel` | Python function template for hooks |

---

## Node Field Contracts

### Common Fields (BaseNodeModel)

All node types inherit base fields:

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `position` | `object{x,y}` | Optional | `{x:0, y:0}` |
| `extra_data` | `object` | Optional | `{}` |

### user_input Fields

| Field | Type | Required | Default | Aliases |
|-------|------|----------|---------|---------|
| `template` | `string` | Optional | `null` | - |
| `text` | `string` | Optional | `null` | `content`, `message` |
| `files` | `array` | Optional | `null` | - |
| `images` | `array` | Optional | `null` | - |
| `extras` | `object` | Optional | `null` | - |

### end Fields

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `end` | `string` | Optional | `null` |

### parser Fields

| Field | Type | Required | Default | Aliases |
|-------|------|----------|---------|---------|
| `text` | `string` | Optional | `""` | `content`, `template` |

**Note**: `inputs` array is UI-only (frontend strips before POST).

### client Fields

| Field | Type | Required | Default | Aliases |
|-------|------|----------|---------|---------|
| `engine` | `string` | Optional | `null` | `provider` |
| `api_info` | `object|string` | Optional | `null` | `config`, `credentials` |
| `model` | `string` | Optional | `null` | `model_name` |

### llm Fields

| Field | Type | Required | Default | Aliases |
|-------|------|----------|---------|---------|
| `top_p` | `number` | Optional | `null` | - |
| `stream` | `boolean` | Optional | `false` | - |
| `json_output` | `boolean` | Optional | `false` | `json_mode` |
| `temperature` | `number` | Optional | `null` | - |
| `max_tokens` | `integer` | Optional | `null` | `max_output_tokens` |
| `iterate` | `boolean` | Optional | `false` | - |

### fetch Fields

| Field | Type | Required | Default | Aliases |
|-------|------|----------|---------|---------|
| `url` | `string` | Optional | `null` | `endpoint` |
| `method` | `string` | Optional | `"GET"` | - |
| `headers` | `object|string` | Optional | `null` | - |
| `params` | `object|string` | Optional | `null` | `query` |
| `body` | `object|string` | Optional | `null` | `data` |
| `json_data` | `object|string` | Optional | `null` | `json_body` |
| `tool_mode` | `boolean` | Optional | `false` | - |
| `tool_name` | `string` | Optional | `null` | - |
| `tool_parameters` | `object` | Optional | `null` | - |

### send_message Fields

| Field | Type | Required | Default | Aliases |
|-------|------|----------|---------|---------|
| `message` | `string` | Optional | `""` | `content` |
| `json_extras` | `string` | Optional | `""` | `extras` |

### text Fields

| Field | Type | Required | Default | Aliases |
|-------|------|----------|---------|---------|
| `text` | `string` | Optional | `""` | `content` |

### loop Fields

No additional fields beyond base.

### inner Fields

| Field | Type | Required | Default | Aliases |
|-------|------|----------|---------|---------|
| `magic_flow` | `object` | Optional | `null` | `flow`, `graph`, `subgraph` |
| `parent_state_mapping` | `object` | Optional | `null` | - |

### conditional Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `condition` | `string` | **Required** | - | Jinja2 template for routing |
| `merge_strategy` | `string` | Optional | `"flat"` | `"flat"` or `"namespaced"` |
| `handles` | `object` | Optional | `null` | Custom handle mappings |
| `output_handles` | `array` | Optional | `null` | Declared output handles |
| `default_handle` | `string` | Optional | `null` | Fallback handle |

**Note**: `condition` must be valid Jinja2 syntax.

### python_exec Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `safety_mode` | `string` | Optional | `"subprocess"` | Execution mode |
| `timeout` | `number` | Optional | `30.0` | Max execution seconds |
| `max_output_chars` | `integer` | Optional | `8000` | Max output length |

### mcp Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `servers` | `array` | **Required** | - | MCP server configs (min 1) |
| `init_timeout` | `number` | Optional | `10.0` | Server init timeout (1-120s) |
| `tool_timeout` | `number` | Optional | `30.0` | Tool call timeout (1-300s) |
| `discovery_timeout` | `number` | Optional | `30.0` | Tool discovery timeout (5-120s) |

Each `servers` entry requires:
- `transport`: `"stdio"` or `"http"` (required)
- For stdio: `command` (string, required)
- For HTTP: `url` (string, required)

### constant Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `value_type` | `string` | Optional | `"str"` | Primitive type: `"int"`, `"bool"`, `"str"`, `"float"` |
| `value` | `any` | Optional | `null` | Primitive value (coerced to `value_type` via model validator) |

### hook Fields

See [nodes/hook.md](nodes/hook.md) and [hooks/README.md](hooks/README.md) for runtime hook behavior.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `function_template` | `string` | Optional | `""` | Python function template (def or async def) for hook execution |
| `timeout_override` | `integer` | Optional | `null` | Per-hook timeout in seconds (global default 30s) |
| `hook_type` | `string` | Optional | `"custom"` | Lifecycle marker: `"pre"`, `"post"`, `"error"`, `"custom"` |

### chat Fields

Uses `ChatNodeModel` for chat configuration, including model-provided history fields such as `history_messages`.

### void Fields

No Pydantic model. Behaves as silent `end` node.

---

## Edge JSON Structure

Each edge in `edges` array must contain:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | `string` | Optional | Edge identifier (auto-generated if missing) |
| `source` | `string` | **Required** | Source node `id` |
| `target` | `string` | **Required** | Target node `id` |
| `sourceHandle` | `string` | Optional | Output handle on source node |
| `targetHandle` | `string` | Optional | Input handle on target node |
| `type` | `string` | Optional | Edge visual type (default `"default"`) |
| `hooks` | `object` | Optional | Edge-level hook config: `{hook_node_id, hook_type, timeout_override, enabled}`; see [hooks/README.md](hooks/README.md) |

### Validation Rules

1. **`source` must reference existing node**
2. **`target` must reference existing node**
3. **Handles must be valid** — unknown handles rejected

---

## Handle Naming Convention

### Canonical Output Handles

| Node Type | Output Handles |
|-----------|----------------|
| `user_input` | `handle_user_message`, `handle_user_files`, `handle_user_images`, `handle_client_extras` |
| `text` | `handle_text_output` |
| `constant` | `handle_constant_output` |
| `parser` | `handle_parser_output` |
| `fetch` | `handle_fetch_output` (or tool handle in tool mode) |
| `client` | `handle-client-provider` |
| `llm` | `handle_streaming_content`, `handle_generated_content`, `handle-tool-calls` |
| `chat` | `handle_chat_output` |
| `send_message` | `handle_message_output`, `content` (system streaming) |
| `loop` | `handle_item`, `handle_end` |
| `conditional` | Dynamic (from `output_handles` or condition result) |
| `inner` | `handle_content_stream`, `handle_execution_content`, `handle_execution_extras` |
| `end` | `handle_end_output` |
| `hook` | `handle-user-output`, `handle-debug-output`, `handle-feedback-output` |

### Canonical Input Handles

| Node Type | Input Handles |
|-----------|---------------|
| `constant` | None (source node) |
| `llm` | `handle-client-provider`, `handle-chat`, `handle-system-context`, `handle_user_message`, tool handles |
| `end` | `handle_flow_input` |
| `parser` | Arbitrary (template references) |
| `hook` | `handle-hook-context` (receives `HookContext` at runtime) |
| `send_message` | `handle_send_extra`, `handle_send_content` |

---

## Legacy Handle Migration

**Clean-break policy** — legacy handles are rejected at validation.

| Legacy Handle | Previously Emitted By | New Handle |
|---------------|----------------------|------------|
| `handle_generated_end` | `llm` | `handle_generated_content` |
| `handle_generated_end` | `parser` | `handle_parser_output` |
| `handle_generated_end` | `send_message` | `handle_message_output` |
| `handle_generated_end` (target) | `end` input | `handle_flow_input` |

Users must reconnect edges manually. No automatic migration.

---

## Validation Behavior

### Backend Validation (`extra='forbid'`)

All Pydantic node models reject unknown fields. When validation fails:

| Error Type | Trigger | Message Format |
|------------|---------|----------------|
| `ExtraInputsNotPermitted` | Unknown field in `data` | `"Extra inputs are not permitted [field_name]"` |
| `UnsupportedNodeType` | Unknown `type` key | `"Unsupported node type: {type}. Available types: [...]"` |
| `MissingRequired` | Missing required field | `"Field required [field_name]"` |
| `InvalidHandle` | Unknown handle | `"Invalid handle: '{handle}' is not valid for {node_type}"` |

### Contract Validation (`contract_config`)

Graphs may include a `contract_config` object at the top level with a `mode` field:

| Mode | Behavior |
|------|----------|
| `"off"` | Disable all contract validation (rollback path) |
| `"shadow"` | Compute diagnostics, attach to report, no surfacing |
| `"warn"` | Surface diagnostics as warnings, execution proceeds (default) |
| `"strict"` | Block execution on contract errors (partially implemented — some enforcement is deferred to follow-up) |

**Note**: `strict` mode and `strict_runtime` are declared in the model surface but full enforcement across all validation dimensions is deferred. Current behaviour blocks only on `GraphValidationError` (structural graph errors); additional strict checks (`targetHandle`, fan-in cardinality, etc.) emit warnings rather than errors regardless of mode. Expect this to harden in subsequent releases.

A `GraphContractReport` is attached to the built model and accessible via `model.contract_report`.

### UI-Only Fields Rejected

These fields are frontend-only and rejected by backend:

- `measured` — ReactFlow node sizing
- `inputs` — Parser input handle list (frontend strips before POST)
- `selected` — ReactFlow selection state
- `dragging` — ReactFlow drag state
- `positionAbsolute` — ReactFlow computed position

Frontend MUST strip these before sending to backend.

---

## Example Valid Graph

```json
{
  "nodes": [
    { "id": "input", "type": "user_input", "position": {"x": 0, "y": 0}, "data": {} },
    { "id": "client", "type": "client", "position": {"x": 200, "y": 100}, "data": {"engine": "openai", "model": "gpt-4"} },
    { "id": "llm", "type": "llm", "position": {"x": 400, "y": 0}, "data": {"temperature": 0.7, "stream": true} },
    { "id": "output", "type": "end", "position": {"x": 600, "y": 0}, "data": {} }
  ],
  "edges": [
    { "source": "input", "target": "llm", "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message" },
    { "source": "client", "target": "llm", "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider" },
    { "source": "llm", "target": "output", "sourceHandle": "handle_generated_content", "targetHandle": "handle_flow_input" }
  ]
}
```

---

## Type Synchronization (Frontend)

Frontend TypeScript interfaces in `magic-ui/src/App/Flow/Types/nodeModels.ts` mirror this contract.

**Manual sync required** — no autogeneration pipeline. When backend Pydantic models change, frontend interfaces must be updated within the same release cycle.

---

## See Also

- [HANDLES_AND_ROUTING.md](wiki/HANDLES_AND_ROUTING.md) — Handle routing protocol
- [VALIDATION.md](wiki/VALIDATION.md) — Build-time validation details
- `spec.md` in `.dev/sdd/changes/node-json-canvas-refactor/` — Full specification
