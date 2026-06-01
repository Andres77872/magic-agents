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
| `type` | `string` | **Required** | Canonical type key (20 types) |
| `data` | `object` | **Required** | Node-specific configuration (may be empty `{}`) |
| `position` | `object` | Optional | Canvas position `{x, y}` (default `{x:0, y:0}`) |

### Validation Rules

1. **`id` must be unique** — duplicate IDs rejected with error
2. **`id` must be non-empty string** — empty IDs rejected
3. **`type` must be canonical** — unknown types rejected with available types list
4. **`data` fields must match node model** — unknown fields rejected (see `extra='forbid'`)
5. **UI-only fields are rejected** — `measured`, `inputs`, `selected`, `dragging` are not permitted

---

## Canonical Node Types (20 Types)

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
| `node_tool` | `NodeTool` | `ToolNodeModel` | Schema-only OpenAI-compatible function tool for client execution |
| `memory` | `NodeMemory` | `MemoryNodeModel` | Vector insights node — extraction, embedding, search, and injection of memory context via similarity |
| `hook` | `NodeHook` | `HookNodeModel` | Python function template for hooks |
| `codex` | `NodeCodex` | `CodexNodeModel` | Knowledge hub: trigger-based content injection into user messages |

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

### node_tool Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `tool` | `object` | **Required** | - | Raw OpenAI-compatible function tool object |

Strict validation applies before provider execution:

- `tool.type` must be `"function"`
- `tool.function` must be an object
- `tool.function.name` must match `^[a-zA-Z0-9_-]{1,64}$`
- `tool.function.description` must be present and non-empty
- `tool.function.parameters` must be an object JSON Schema with `type: "object"`

`node_tool` emits the validated raw object on `handle-tool-definition`. The backend never executes this function server-side; clients execute provider tool calls emitted by `NodeLLM`.

Schema-only NodeTool calls on `llm.handle-tool-calls` use this envelope:

```json
{
  "execution": "client",
  "source": "schema_only",
  "tool_calls": []
}
```

Do not mix `node_tool` schemas with callable server tools on the same LLM node in v1. Frontend/UI rollout is external to this backend repository and must add compatible node data, edge validation, save/load, and client execution handling before public UI use.

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

### memory Fields

Uses `MemoryNodeModel` for vector memory configuration. NodeMemory extracts memories from user messages via LLM extraction, stores them with embeddings, and injects relevant past memories as prompt context.

**Data structure**:
```json
{
  "instructions": "Extract key concerns from this message",
  "memory_entries": [
    {"content": "User prefers email communication", "trigger": "preference"}
  ],
  "top_k": 10
}
```

When `top_k` is absent, the default value of `5` is used (Pydantic `Field(default=5)`).

**`MemoryEntry` sub-model fields**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `string` | Optional | Auto-generated 32-char UUID4 hex | **STRICTLY auto-generated. DO NOT provide in JSON — raises `ValidationError`.** Auto-generated as `uuid.uuid4().hex` (32 hex chars). |
| `source_id` | `string` or `null` | Optional | `null` | Read-back field populated by vector DB search. Contains the scope-aware composite document ID (UUID format) from the vector storage. Distinct from auto-generated `id`. **Not provided in input JSON** — populated by VDB search result reconstruction. `null` when metadata lacks the `doc_id` key (backward-compatible with pre-existing entries). |
| `content` | `string` | **Required** | — | Memory content text (min 1 character). |
| `trigger` | `string` | Optional | `""` | Memory tag/category (singular string, unlike CodexEntry's `triggers` list). |
| `created_at` | `string (ISO 8601)` | Optional | Auto-generated UTC timestamp | Memory creation timestamp. Auto-generated via `datetime.now(UTC)`. |

**`MemoryEntry` validation rules**:
- `id` MUST NOT be provided manually. If set in JSON, the model raises `ValidationError` with message `"MemoryEntry.id is auto-generated. Do not provide a value for 'id'. Remove the 'id' field from the memory entry JSON."`
- `source_id` is intentionally unvalidated — it is a read-back field populated by vector DB search. Unlike `id`, there is no `reject_manual_id` validator on `source_id`. Users CAN set `source_id` in input JSON, but it will be overwritten by VDB search results at runtime.
- `content` MUST be non-empty (min 1 character). Empty content raises `ValidationError`.
- Extra unknown fields in a `MemoryEntry` are rejected (inherited `extra='forbid'`).
- `id` and `created_at` are always auto-generated at construction time, never read from JSON input.

**`MemoryNodeModel` fields**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `instructions` | `string` | Optional | `null` | LLM extraction prompt template. When `null`, only vector search is performed (no LLM extraction). When set, NodeMemory uses this as the system prompt for the extraction LLM call. |
| `memory_entries` | `array` | Optional | `[]` | Seed memory entries (array of `MemoryEntry` objects). This is a seed/export snapshot only — the vector DB is the source of truth for runtime search. |
| `top_k` | `integer` | Optional | `5` | Number of top memory matches to retrieve from vector search. Valid range: 1–50. When absent, Pydantic fills the default (5). |

**Handle contract**:

| Direction | Handle | Purpose | Mandatory |
|-----------|--------|---------|-----------|
| Input | `handle_memory_input` | User message input (from upstream Codex or UserInput node) | Yes |
| Input | `handle-client-provider` | MagicLLM client delivery (from NodeClientLLM) | No — degrades to search-only mode if missing |
| Output | `handle_memory_output` | Memory-enriched message output (NOT `handle_user_message` — avoids competing-edge race with Codex) | N/A |

**Output handle design**: NodeMemory emits `handle_memory_output` (not `handle_user_message`) to avoid the competing-edge anti-pattern when both NodeCodex and NodeMemory exist in the same graph. If Codex also emits `handle_user_message`, having both emit the same handle causes a non-deterministic race on the downstream Chat node. Graph authors can override the output handle to `handle_user_message` via `handles={"output": "handle_user_message"}` when no Codex coexistence is needed.

**Scope isolation**: Memories are scoped by the compound key `(chat_session_id, node_id)`:
- Every `search()` call enforces `filter_scope={"session_id": chat_log.id_chat, "node_id": self.node_id}`
- Two NodeMemory nodes with different `node_id` values produce independent search results within the same chat session
- The same NodeMemory node across different chat sessions produces independent search results

**`<memory_content>` is prompt text — NOT a security boundary**:
- The runtime does NOT parse, validate, or escape XML. Memory content and user messages pass through unmodified.
- Content containing `</memory_content>`, `<script>`, or any tag-like text is inserted verbatim between the hard-coded delimiter tags.
- The wrapper is a prompt engineering hint for the LLM, not a structural or security boundary — matches the `<codex_content>` policy established by NodeCodex exactly.

**Graph wiring examples**:

1. **Memory alone (no Codex)** — Override output to `handle_user_message` for zero-config downstream:
```json
{
  "nodes": [
    {"id": "input", "type": "user_input", "data": {}},
    {"id": "memory-1", "type": "memory", "data": {"memory_entries": []}},
    {"id": "chat", "type": "chat"},
    {"id": "end", "type": "end"}
  ],
  "edges": [
    {"source": "input", "target": "memory-1", "sourceHandle": "handle_user_message", "targetHandle": "handle_memory_input"},
    {"source": "memory-1", "target": "chat", "sourceHandle": "handle_memory_output", "targetHandle": "handle_user_message"},
    {"source": "chat", "target": "end"}
  ]
}
```

2. **Memory + Codex coexistence** — Distinct handles prevent competing-edge race:
```json
{
  "nodes": [
    {"id": "input", "type": "user_input", "data": {}},
    {"id": "codex", "type": "codex", "data": {"codex_entries": []}},
    {"id": "client", "type": "client", "data": {"engine": "openai"}},
    {"id": "memory-1", "type": "memory", "data": {"instructions": "Extract key points"}},
    {"id": "chat", "type": "chat"},
    {"id": "end", "type": "end"}
  ],
  "edges": [
    {"source": "input", "target": "codex", "sourceHandle": "handle_user_message", "targetHandle": "handle_codex_input"},
    {"source": "codex", "target": "memory-1", "sourceHandle": "handle_user_message", "targetHandle": "handle_memory_input"},
    {"source": "client", "target": "memory-1", "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
    {"source": "memory-1", "target": "chat", "sourceHandle": "handle_memory_output", "targetHandle": "handle_user_message"},
    {"source": "chat", "target": "end"}
  ]
}
```

**Scope-aware composite IDs**: NodeMemory uses a **scope-aware composite** SHA256 hash as the vector DB document ID for upserts:

```
doc_id = str(uuid.UUID(sha256(content + "|" + session_id + "|" + node_id)[:32]))
```

This ensures:
- **Same-scope dedup**: Same content within the same session AND same node produces the same document ID — no duplicates.
- **Cross-scope isolation**: Different sessions or different nodes produce different document IDs — preventing cross-scope metadata overwrite.
- **Qdrant compatibility**: 32-hex-char truncation + UUID formatting produces valid Qdrant point IDs.
- The `MemoryEntry.source_id` field (returned from search) holds this document ID for traceability.

**Constructor runtime deps**: Runtime dependencies (`vector_db`, `embedding_client`) are injected through the `run_agent(deps=...)` public API — not serialized in graph JSON. See [`docs/nodes/memory.md`](nodes/memory.md#dependency-injection-via-run_agentdeps) for usage examples.

**`_capture_internal_state` fields** (debugging):
| Field | Type | Description |
|-------|------|-------------|
| `memory_entry_count` | `int` | Total entries in the runtime memory list |
| `extracted_count` | `int` | Memories extracted in the last `process()` run |
| `injected_count` | `int` | Memories injected in the last `process()` run |
| `top_k` | `int` | Configured number of top matches to retrieve from vector search. Mirrors the `top_k` field from `MemoryNodeModel` (default 5, range 1–50). |
| `vector_db_available` | `bool` | Always `true` — vector DB is auto-created as ephemeral fallback if none is injected. Retained for backward compatibility. |
| `vector_db_backend` | `str` | Backend type: `"ephemeral"` (InMemoryVectorDB auto-created or manually passed), `"qdrant"` (QdrantVectorDB), or `"external"` (any other VectorDB-compatible implementation). Diagnostic field — signals configuration, not health. |

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

### codex Fields

Uses `CodexNodeModel` for knowledge configuration. Each codex entry defines a trigger keyword and associated content that is prepended to user messages when triggers match.

**Data structure**:
```json
{
  "codex_entries": [
    {
      "id": "550e8400e29b41d4a716446655440000",
      "triggers": ["help", "ayuda"],
      "content": "Help resources: /docs/faq"
    }
  ]
}
```

**`CodexEntry` fields**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `string` | Optional | Auto-generated UUID4 hex | Entry ID. Auto-generated as 32-char UUID4 hex if omitted. If provided, must be a valid UUID4 string — accepts both hyphenated (`550e8400-e29b-41d4-a716-446655440000`) and non-hyphenated (`550e8400e29b41d4a716446655440000`) formats. Invalid UUIDs are rejected with `ValidationError`. |
| `triggers` | `array[string]` | **Required** | — | Trigger keywords (min 1). Matching is case-insensitive word-boundary (`\b`) using Python `re.search()`. |
| `content` | `string` | **Required** | — | Content to prepend when triggers match (min 1 char). Wrapped in `<codex_content>...</codex_content>` — this is prompt annotation text, not parsed XML. |

**Trigger matching details**:
- Case-insensitive: trigger `"HELP"` matches input `"help"`, `"Help"`, `"HELP"`, etc.
- Word-boundary (`\b`): prevents false positives — trigger `"ai"` does NOT match `"EMAIL"`, `"trains"`, or `"plain"`.
- Each entry matches at most once (first matching trigger per entry). Multiple entries can all match.
- **Known limitation**: Triggers containing non-word characters (`c++`, `node.js`, `foo-bar`) may behave unexpectedly with `\b` word-boundary semantics. Graph authors should avoid non-word characters in triggers, or test coverage against Python `re` `\b` behavior.

**`<codex_content>` is prompt text — NOT a security boundary**:
- The runtime does NOT parse, validate, or escape XML. Content and user messages pass through unmodified.
- Codex content containing `</codex_content>`, `<script>`, or any tag-like text is inserted verbatim between the hard-coded delimiter tags.
- The wrapper is a prompt engineering hint for the LLM, not a structural or security boundary.
- Graph authors who need content sanitization or transformation should place a `NodeParser` downstream.

**⚠️ WARNING — Competing edge anti-pattern**:
Codex-enabled graphs MUST NOT retain a direct `UserInput.handle_user_message → Chat.handle_user_message` edge alongside the Codex edges. Two competing writes to `Chat.inputs['handle_user_message']` (one from UserInput, one from Codex) causes race behavior where the Codex-transformed message may be silently overwritten.

The Codex output edge MUST replace the direct edge. Correct wiring:

```json
{
  "edges": [
    {"source": "input", "target": "codex",
     "sourceHandle": "handle_user_message", "targetHandle": "handle_codex_input"},
    {"source": "codex", "target": "chat",
     "sourceHandle": "handle_user_message", "targetHandle": "handle_user_message"}
  ]
}
```

Automatic detection or validation of competing edges is **NOT implemented** — graph author responsibility.

**`_capture_internal_state` fields** (debugging):
| Field | Type | Description |
|-------|------|-------------|
| `codex_entry_count` | `int` | Number of configured codex entries |
| `trigger_count` | `int` | Total number of triggers across all entries |
| `input_handle` | `string` | Active input handle name |
| `output_handle` | `string` | Active output handle name |

**NodeCodex is NOT auto-inserted by `build()`** — graph authors MUST explicitly declare the `codex` node in their agent JSON.

### chat Fields

Uses `ChatNodeModel` for chat configuration, including session management, message history assembly via the 5-slot merge pipeline, and STM windowing controls.

#### Field Table

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `session_id` | `string` or `null` | No | `null` | — | Unique thread/conversation ID for persistence. Maps to backend session storage for message continuity across turns. |
| `session_required` | `boolean` | No | `false` | — | If `true`, enforce session presence with auto-create fallback. When `false`, the node operates without session persistence. |
| `history_messages` | `array` or `null` | No | `null` | — | **Backend-authoritative** message history. Populated by backend at build time — NOT read from JSON config. Injected as **Slot 1** of the 5-slot merge pipeline. Each element is a message dict with `role`, `content`, and optional metadata. |
| `custom_messages` | `array` or `null` | No | `null` | — | Pre-user context messages injected as **Slot 3** of the 5-slot merge pipeline. Useful for injecting static instructions, few-shot examples, or system-style context before the current user message. Each element is a message dict. |
| `messages_append_mode` | `boolean` | No | `false` | — | Message append mode. `false` = legacy **REPLACE** mode (runtime input messages replace `history_messages`). `true` = **APPEND** mode (runtime messages appended to `history_messages` in Slot 2). |
| `message` | `string` or `null` | No | `null` | — | Legacy inline message field. Kept for backward compatibility. NOT the primary user-message input path — user messages enter via the `handle_user_message` runtime handle (Slot 5). |
| `memory` | `object` or `null` | No | `null` | — | Legacy configuration dict. **DEPRECATED** — only `memory.max_input_tokens` is consumed. When `max_input_tokens` is `null` AND `memory.max_input_tokens` is set, the legacy value is used as fallback with a runtime deprecation warning. The legacy `stm` key is deprecated and ignored: when present, NodeChat logs a deprecation warning at init. Use `max_messages` instead. Other keys (`ltm`, etc.) are ignored. |
| `handles` | `object` or `null` | No | `null` | — | Handle name overrides for custom edge wiring. Maps handle types to custom handle names, e.g. `{"input": "handle_custom_input", "output": "handle_user_message"}`. |
| `max_messages` | `int` or `null` | No | `null` | `>= 1` | Maximum non-system conversation messages to keep after windowing. **System messages are always preserved and NOT counted** against this limit. The last user message (current turn) is also always preserved. Applied after the 5-slot merge, before the message list is yielded to downstream. Older messages are dropped (newest kept). When `null`, no message-count limit is enforced. |
| `max_input_tokens` | `int` or `null` | No | `null` | `>= 1` | Maximum estimated tokens for the assembled chat context. Applied AFTER `max_messages` truncation when `truncation_strategy='tail'`, or independently when `truncation_strategy='token_budget'`. Falls back to legacy `memory.max_input_tokens` if that dict key is present (with deprecation warning). Token estimation uses a hardcoded GPT-5 tokenizer (not configurable via `model`). When `null`, no token-budget limit is enforced. |
| `truncation_strategy` | `string` | No | `"tail"` | `"tail"` or `"token_budget"` | Windowing algorithm. `"tail"` = keep the newest N non-system messages (oldest dropped first). `"token_budget"` = skip message-count truncation entirely and rely solely on token-budget enforcement via `max_input_tokens`. Any other string value is rejected by Pydantic validation (`Literal['tail', 'token_budget']`). |
| `model` | `string` or `null` | No | `null` | — | Model name for usage/logging tracking only. Does **NOT** affect STM windowing token estimation, which uses a hardcoded GPT-5 tokenizer. Optional — purely informational. |

#### Windowing Semantics (STM)

Windowing is recomputed fresh every `process()` execution over the **accumulated** message list after all 5 merge slots complete:

1. **System message preservation**: All `role='system'` messages are always preserved and excluded from the `max_messages` count. They survive truncation unchanged.
2. **Last user preservation**: The most recent `role='user'` message (current turn) is always preserved and never dropped, regardless of truncation aggressiveness.
3. **Tool-call atomicity**: Tool-call chains (an `assistant` message with `tool_calls` + consecutive `tool` role results) are treated as atomic units. A chain is either fully kept or fully dropped — never split. Windowing cuts only at chain boundaries.
4. **Windowing timing**: Applied after the 5-slot merge (post-Slot 5, pre-yield) on every `process()` turn. This means windowing operates over the full accumulated message history, not just the current turn.
5. **Two-layer defense**:
   - **Layer 1 (PRIMARY)**: `NodeChat.process()` applies `apply_windowing()` with `max_messages` + `max_input_tokens` post-merge.
   - **Layer 2 (SAFETY NET)**: `ModelChat.__init__(max_input_tokens=...)` retains existing provider-level token truncation. Layer 1 is the primary control; Layer 2 is a fallback.
6. **Hardcoded GPT-5 token estimation**: Token estimation for STM windowing uses a hardcoded GPT-5 tokenizer (`tiktoken.encoding_for_model("gpt-5")`). The `model` field on the node does NOT affect token estimation; if the model is used, it is for usage/logging tracking only. If GPT-5 is unknown to tiktoken, the encoding silently falls back to `cl100k_base`.

#### NodeLLM No-CHAT Relation

When a graph connects `user_input → llm` **without** a `chat` node (the no-CHAT fallback path), `NodeLLM` applies its own windowing on `history_messages` using the same `apply_windowing()` utility. In this path, when no `max_messages` is explicitly configured, a runtime default of **30** (`NodeLLM.DEFAULT_MAX_MESSAGES`) is applied to protect against unbounded context. This default does NOT apply when a `chat` node is present — NodeChat owns windowing in that case.

#### Legacy Compatibility Notes

- **`max_input_tokens` first-class field wins**: When both `max_input_tokens` and `memory.max_input_tokens` are set, the first-class field takes precedence and NO deprecation warning is logged.
- **`memory` field NOT removed**: The `memory` dict remains for backward compatibility but is formally DEPRECATED. Only the `max_input_tokens` key is consumed; other keys pass through unmodified. Removal is deferred to the next major version.
- **Additive defaults**: All new STM fields (`max_messages`, `max_input_tokens`, `truncation_strategy`, `model`) are `null`/`"tail"` by default. Existing graphs without these fields work identically — no windowing is applied unless explicitly configured.
- **Zero values are rejected**: Pydantic `ge=1` validation rejects `max_messages=0` and `max_input_tokens=0` at model construction time with `ValidationError`.

#### Example JSON

```json
{
  "id": "chat-1",
  "type": "chat",
  "data": {
    "session_id": "thread-abc-123",
    "session_required": true,
    "history_messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there!"}
    ],
    "custom_messages": [
      {"role": "system", "content": "You are a helpful assistant."}
    ],
    "messages_append_mode": true,
    "max_messages": 50,
    "max_input_tokens": 128000,
    "truncation_strategy": "tail",
    "model": "gpt-4o"
  }
}
```

**Minimal configuration** (no windowing, all defaults):
```json
{
  "id": "chat-1",
  "type": "chat",
  "data": {}
}
```

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
| `codex` | `handle_user_message` |
| `memory` | `handle_memory_output` |

### Canonical Input Handles

| Node Type | Input Handles |
|-----------|---------------|
| `constant` | None (source node) |
| `llm` | `handle-client-provider`, `handle-chat`, `handle-system-context`, `handle_user_message`, tool handles; runtime-overridable generation handles: `handle-llm-temperature`, `handle-llm-top_p`, `handle-llm-max_tokens`, `handle-llm-stream`, `handle-llm-iterate`, `handle-llm-json_output` |
| `end` | `handle_flow_input` |
| `parser` | Arbitrary (template references) |
| `hook` | `handle-hook-context` (receives `HookContext` at runtime) |
| `codex` | `handle_codex_input` |
| `memory` | `handle_memory_input`, `handle-client-provider` |

### Port Cardinality

| Node Type | Handle | Cardinality | Exclusive |
|-----------|--------|-------------|-----------|
| `memory` | `handle_memory_input` | `one` | `True` |

Notes:
- `handle_memory_input` accepts exactly one incoming edge (cardinality `one`, exclusive `True`).
- `handle-client-provider` does NOT have a cardinality entry — the default behavior allows the edge without additional cardinality enforcement.

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

---

## Known Limitations

### 1. OpenAI-Only Embedding

Only OpenAI-compatible engines support `async_embedding()`. Non-OpenAI engines cause `async_embedding()` to return `None`, which makes NodeMemory skip all vector operations (embedding, upsert, and search) while still supporting LLM extraction (Phase 2) when `instructions` is configured.

**Behavior**: When embedding is unavailable, NodeMemory logs a diagnostic warning mentioning the OpenAI-only limitation and yields the original user message unchanged. The graph continues executing — no crash.

**Workaround**: Use an OpenAI-compatible engine for any graph that requires memory enrichment via vector similarity. Non-OpenAI graphs work correctly but output messages without memory context.

### 2. No Codex Trigger Integration

NodeMemory and NodeCodex operate independently. There is no automatic mechanism to convert Codex triggers into memory tags. A `codex_entries` trigger like `"help"` does NOT automatically create a memory entry with `trigger="help"`.

**Workaround**: Graph authors can duplicate trigger values manually in both the `codex_entries` config (for Codex content injection) and the `instructions` prompt text (for memory extraction). This is explicit but requires manual synchronization.

**Follow-up**: Codex-trigger-as-memory-tag integration is deferred to a separate SDD. Any enhancement to NodeCodex compatibility requires a Codex SDD amendment.

### 3. `<memory_content>` Is NOT a Security Boundary

The `<memory_content>` wrapper is pure prompt annotation text. NodeMemory performs no XML parsing, escaping, or sanitization. Content containing `</memory_content>`, `<script>`, or any tag-like sequences passes through unmodified between the delimiters.

This matches the `<codex_content>` policy exactly — see the [codex Fields](#codex-fields) section for the same guarantee.

**Implication**: Downstream nodes (LLM, Chat, SendMessage) receive raw text. Graph authors who need content sanitization or transformation should place a `NodeParser` or other transform node downstream of the Memory node.

### 4. No Hard Validation for Missing Runtime Dependencies

NodeMemory does not enforce hard validation in `validate_graph()` for missing vector DB or MagicLLM client.

- **No vector DB**: An ephemeral `InMemoryVectorDB` is auto-created at construction time. Embedding, upsert, and search proceed normally but all data is lost on process restart. Use `pip install magic-agents[qdrant]` and inject a `QdrantVectorDB` via `deps` for persistent storage. The `vector_db_backend` debug field signals `"ephemeral"` vs `"qdrant"` vs `"external"`.
- **No client**: Extraction is skipped. Search is skipped. The message passes through unchanged.

This matches the existing pattern: the Codex competing-edge anti-pattern is also documented but not enforced at validation.

### 5. Single-Engine Embedding Only

Vector search depends on embedding quality from a single embedding model. No multi-engine embedding abstraction layer exists. The embedding model is determined by the MagicLLM client's engine configuration.

**Scope isolation** via `filter_scope={"session_id": ..., "node_id": ...}` ensures memory pools are independent across sessions and nodes. However, if the embedding model changes between sessions, the same text may produce different embedding vectors, affecting search relevance. This is expected behavior — no backward-compatibility guarantee across model changes.

### 6. NodeCodex Modifications Are Out of Scope

This documentation covers NodeMemory as a standalone node type. Any modifications to NodeCodex (`NodeCodex.py`, `CodexNodeModel.py`, or Codex touchpoints) for Codex-Memory integration require a separate SDD change. See [No Codex Trigger Integration](#2-no-codex-trigger-integration) above.
