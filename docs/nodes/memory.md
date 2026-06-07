# `memory`

## Purpose

Extract structured memories from user messages via LLM extraction, store them with embeddings in a vector database, search for relevant past memories on each invocation, and inject matching memories as `<memory_content>` prompt context.

Current runtime behavior is split: extraction and upsert run in a background task, while vector search and injection run synchronously for the current invocation. Newly extracted memories become available to later invocations, not to the current output.

NodeMemory is a **new first-class node** (`memory` type), **not** an upgrade to NodeCodex. Codex-Memory trigger integration is deferred to a follow-up.

## Runtime class

- `NodeMemory`
- model: `MemoryNodeModel`
- protocol: `VectorDB` (structural subtyping via `typing.Protocol`)

## Default handles

| Direction | Handle | Purpose | Mandatory |
|-----------|--------|---------|-----------|
| Input | `handle_memory_input` | User message input (from upstream Codex or UserInput node) | Yes |
| Input | `handle-client-provider` | MagicLLM client delivery for optional extraction (from NodeClientLLM) | No — extraction is skipped if missing |
| Output | `handle_memory_output` | Memory-enriched message output (distinct from `handle_user_message` — avoids competing-edge race with Codex) | N/A |

Handle overrides are supported via `handles` dict:
- `handles.get('input', DEFAULT_INPUT_HANDLE)` → custom input handle
- `handles.get('output', DEFAULT_OUTPUT_HANDLE)` → custom output handle (override to `"handle_user_message"` for zero-config downstream when no Codex coexistence)
- `handles.get('client', DEFAULT_CLIENT_HANDLE)` → custom client handle

## Model fields

### `MemoryNodeModel` top-level fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `instructions` | `string` | Optional | `null` | LLM extraction prompt template. When `null`, only vector search is performed (no LLM extraction). When set, NodeMemory uses this as the system prompt for the extraction LLM call. |
| `memory_entries` | `array` | Optional | `[]` | Seed memory entries (array of `MemoryEntry` objects). This is a seed/export snapshot only — the vector DB is the source of truth for runtime search. |
| `top_k` | `integer` | Optional | `5` | Number of top memory matches to retrieve from vector search. Valid range: 1–50. Configurable per graph in JSON (see [Graph wiring](#graph-wiring)). |
| `context_messages_count` | `integer` | Optional | `5` | Number of previous chat/history messages to include in the extraction prompt. Valid range: 0-100. This affects extraction only, not vector search. |

### `MemoryEntry` sub-model fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `string` | Optional | Auto-generated 32-char UUID4 hex | **STRICTLY auto-generated. DO NOT provide in JSON — raises `ValidationError`.** Auto-generated as `uuid.uuid4().hex` (32 hex chars). |
| `source_id` | `string` or `null` | Optional | `null` | Read-back field populated by vector DB search from stored metadata `doc_id`. Distinct from auto-generated `id`. `null` when no metadata is available. |
| `content` | `string` | **Required** | — | Memory content text (min 1 character). |
| `trigger` | `string` | Optional | `""` | Memory tag/category (singular string, unlike CodexEntry's `triggers` list). |
| `created_at` | `string (ISO 8601)` | Optional | Auto-generated UTC timestamp when omitted | Memory creation timestamp. |

### `MemoryEntry` validation rules

- `id` MUST NOT be provided manually. If set in JSON, the model raises `ValidationError`.
- `content` MUST be non-empty (min 1 character). Empty content raises `ValidationError`.
- Extra unknown fields in a `MemoryEntry` are rejected (inherited `extra='forbid'`).
- `id` is always auto-generated at construction time and never read from JSON input.
- `created_at` is auto-generated when omitted; a provided parseable datetime is accepted by Pydantic.

## Multi-phase async process

NodeMemory currently executes six logical phases with fail-open behavior:

| Phase | Operation | Fail-open behavior |
|-------|-----------|-------------------|
| **1 - Input** | Get user message from `handle_memory_input` and optional MagicLLM client from `handle-client-provider`. If no message is present, yield an empty passthrough. | N/A |
| **2 - Background extraction** | If `instructions` and a MagicLLM client are available, start `asyncio.create_task(...)` for extraction. The prompt includes the system instructions, up to `context_messages_count` injected history messages, and the current message. | Log warning in the background task; current invocation continues. |
| **3 - Background embedding + upsert** | The background task embeds extracted memories with injected `embedding_client.llm.async_embedding(...)` and upserts them into the vector DB with a UUID4 `doc_id` stored in metadata. | Log warning and skip that memory/batch. |
| **4 - Vector search** | If an injected `embedding_client` is available, embed the current message and call `vector_db.search(..., top_k=top_k, filter_scope={"session_id": str(chat_log.id_chat) or "", "node_id": self.node_id or ""})`. | Log warning and use the original message unchanged. |
| **5 - Injection** | Sort returned matches by `created_at` ascending, wrap each as `<memory_content>{content}</memory_content>`, and prepend them to the current user message. | Use the original message unchanged. |
| **6 - Yield** | Emit the final text on `handle_memory_output`. | N/A |

Important timing detail: phases 2 and 3 are fire-and-forget. Extracted memories are not available to phase 4 in the same invocation.

Embedding is not taken from `handle-client-provider`. Vector search and upsert require an injected `embedding_client` dependency whose `llm` object implements `async_embedding(...)`. The MagicLLM client delivered by `handle-client-provider` is used for extraction.

## Scope isolation

Every `search()` call enforces a compound key filter:

```python
filter_scope = {
    "session_id": chat_log.id_chat,  # From ModelAgentRunLog
    "node_id": self.node_id,          # From Node base class
}
```

- Two NodeMemory nodes with different `node_id` values produce independent search results within the same chat session.
- The same NodeMemory node across different chat sessions produces independent search results.
- The vector DB implementation MUST enforce this filter. The production `InMemoryVectorDB` fallback enforces both keys.

## Vector document IDs and `source_id`

The current background upsert path creates a fresh UUID4 string for each stored vector document:

```python
doc_id = str(uuid.uuid4())
```

That value is used as the vector DB point ID and is also stored in metadata under `"doc_id"`. Search results reconstruct `MemoryEntry` objects with `source_id=metadata.get("doc_id")`, which lets consumers trace a returned memory back to its vector-storage document.

This means the current runtime does **not** content-deduplicate extracted memories by hashing content/session/node. Scope isolation is enforced by search filters, not by deterministic document IDs.

## `<memory_content>` is prompt text — NOT a security boundary

- The runtime does NOT parse, validate, or escape XML. Memory content passes through unmodified between the hard-coded delimiter tags.
- Content containing `</memory_content>`, `<script>`, or any tag-like sequences is inserted verbatim.
- The wrapper is a prompt engineering hint for the LLM, not a structural or security boundary — matches the `<codex_content>` policy exactly.

## Constructor runtime deps

### Dependency injection via `run_agent(deps=...)`

Runtime dependencies (`vector_db`, `embedding_client`, and optional `history_messages`) are injected through `run_agent(raw_graph_dict, deps=...)` or `build(..., deps=...)`.

`run_agent(deps=...)` only forwards dependencies when the graph argument is a raw dict. If you pass an already built `AgentFlowModel`, dependencies were already bound during `build()` and new `deps` passed to `run_agent()` are ignored.

#### Ephemeral fallback (default)

If no `vector_db` is provided, NodeMemory auto-creates an ephemeral `InMemoryVectorDB` at construction time:

```python
from magic_agents import run_agent
from magic_agents.agt_flow import build

# No deps - NodeMemory auto-creates InMemoryVectorDB during build()
graph = build(graph_dict, message="hello")
async for result in run_agent(graph):
    ...
```

**⚠️ Data-loss warning**: The `InMemoryVectorDB` fallback stores all data in-memory only. **All memories are lost on process restart.** This is suitable for development, testing, or stateless scenarios. For persistent storage, provide an explicit `vector_db` (see below).

### Persistent storage via Qdrant (recommended for production)

For persistent vector storage, inject a `QdrantVectorDB` wrapper through `run_agent(deps=...)`:

```python
from magic_agents import run_agent
from magic_agents.vector_storage import QdrantVectorDB
from qdrant_client import AsyncQdrantClient

client = AsyncQdrantClient(url="http://localhost:6333")
vector_db = QdrantVectorDB(client=client, dim=1536)

# Inject vector_db via deps — use the Memory node's ID as the key
deps = {"memory-1": {"vector_db": vector_db}}
async for result in run_agent(graph_dict, deps=deps):
    ...
```

The `deps` dict uses the node ID from your graph JSON (`"memory-1"` in this example) as the key. NodeMemory receives the injected dependencies at construction time.

#### Custom embedding client injection

You can also inject a custom embedding client through the same `deps` mechanism:

```python
from magic_agents import run_agent

custom_embedder = MyCustomEmbeddingClient()

deps = {"memory-1": {"embedding_client": custom_embedder}}
async for result in run_agent(graph_dict, deps=deps):
    ...
```

This is useful for:
- Using a non-OpenAI embedding model
- Sharing a pre-configured embedding client across multiple nodes
- Injecting a mock embedder in tests

Both `vector_db` and `embedding_client` can be injected together:

```python
deps = {"memory-1": {"vector_db": qdrant_vdb, "embedding_client": custom_embedder}}
```

Key characteristics of `QdrantVectorDB`:

- **Single collection**: Uses one project-wide collection named `"magic_llm_memory"` (default). Node/session isolation is achieved via Qdrant payload filtering on `node_id` and `session_id` fields — not separate collections.
- **Lazy lifecycle**: The collection is NOT created at construction time. It is automatically created on first `upsert()` or `search()` via an internal lazy check.
- **FLOAT16 by default**: Vectors are stored with `Datatype.FLOAT16` (explicit, not the Qdrant default `float32`). This reduces storage footprint while preserving accuracy for LLM embeddings.
- **Minimal constructor**: Accepts `dim` (auto-detected from first embedding if `None`), `dtype` (defaults to `"float16"`), `collection_name`, and a `**kwargs` escape hatch passed through to `create_collection()` for advanced Qdrant config (e.g., `distance`, `hnsw_config`, `quantization_config`, `on_disk`).
- **Payload filtering**: Supports filtering on `node_id`, `session_id`, `tags`, `content`, and `creation_date` via Qdrant's `FieldCondition` mechanism. The `filter_scope` dict provided by NodeMemory is automatically translated to Qdrant filter conditions.
- **No Qdrant coupling in NodeMemory**: NodeMemory does NOT import, know about, or depend on Qdrant. The `QdrantVectorDB` wrapper implements the `VectorDB` Protocol; NodeMemory interacts with it purely through `upsert()` and `search()`. Collection lifecycle, payload schema, and Qdrant-specific logic are entirely encapsulated in the wrapper.

Requires `pip install magic-agents[qdrant]` (optional extra).

The MagicLLM client is NOT passed via `deps` — it arrives via the existing `handle-client-provider` edge mechanism, following the proven NodeLLM pattern.

## Qdrant rebuild guidance

`QdrantVectorDB` lazily creates the configured collection on first upsert or search. Existing Qdrant collections keep their original vector dimension and datatype; Qdrant collection shape is not migrated automatically.

For a clean state after changing embedding model, vector dimension, datatype, or collection options:

1. **Delete** the existing `"magic_llm_memory"` collection (or your custom collection name):
   ```python
   await client.delete_collection(collection_name="magic_llm_memory")
   ```
2. Let the system **repopulate** naturally as users interact. `QdrantVectorDB` auto-creates the collection on first `upsert()` or `search()`.
3. If preserving pre-upgrade memories is critical, back up the Qdrant storage directory before deleting.

### No action needed (development / test)

For development or testing environments where data loss is acceptable, no rebuild is needed unless the existing collection shape conflicts with the configured embedding dimension.

### Configuring `top_k` in graph JSON

Add the `top_k` field to the memory node's `data` object to control how many memory entries are retrieved on each invocation:

```json
{
  "nodes": [
    {"id": "memory-1", "type": "memory", "data": {"top_k": 10, "memory_entries": []}},
    ...
  ]
}
```

Valid values: `1` through `50`. Default: `5` (when `top_k` is absent from JSON).

### Memory alone (no Codex)

Override output to `handle_user_message` for zero-config downstream:

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
    {"source": "chat", "target": "end", "sourceHandle": "handle_chat_output", "targetHandle": "handle_flow_input"}
  ]
}
```

### Memory + Codex coexistence

Distinct handles prevent competing-edge race:

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
    {"source": "chat", "target": "end", "sourceHandle": "handle_chat_output", "targetHandle": "handle_flow_input"}
  ]
}
```

## Debugging: `_capture_internal_state`

| Field | Type | Description |
|-------|------|-------------|
| `memory_entry_count` | `int` | Total entries in the runtime memory list |
| `extracted_count` | `int` | Reserved extraction counter. Current background extraction path resets it to `0` and does not update it before yielding. |
| `injected_count` | `int` | Memories injected in the last `process()` run |
| `top_k` | `int` | Configured number of top matches to retrieve from vector search. Mirrors the `top_k` field from `MemoryNodeModel` (default 5, range 1–50). |
| `_context_messages_count` | `int` | Number of history messages considered for background extraction context. |
| `_history_messages_count` | `int` | Count of constructor-injected history messages available to background extraction. |
| `vector_db_available` | `bool` | Whether a vector DB instance is configured. Always `true` now — the fallback auto-creates an `InMemoryVectorDB` if none is provided. |
| `vector_db_backend` | `str` | Detected vector DB backend type. One of: `"ephemeral"` (auto-created `InMemoryVectorDB`), `"qdrant"` (`QdrantVectorDB` wrapper), or `"external"` (any other `VectorDB`-compatible implementation). |

## Known limitations

1. **Embedding client required for vector operations**: Vector search and background upsert require an injected `embedding_client` with `llm.async_embedding(...)`. Without it, extraction can still run, but vector enrichment is skipped.
2. **No Codex trigger integration**: No automatic mechanism to convert Codex triggers into memory tags. Manual trigger duplication required.
3. **No memory eviction/limit**: Vector DB accumulates indefinitely. No count limit, LRU eviction, or expiration.
4. **No memory deletion/forgetting**: NodeMemory only creates memories. No delete/forget UX or API.
5. **Ephemeral fallback is NOT persistent**: When no `vector_db` is provided via `deps`, NodeMemory auto-creates an `InMemoryVectorDB` that lives entirely in memory. All memories are lost on process restart. For persistent storage, install `pip install magic-agents[qdrant]` and inject a `QdrantVectorDB` wrapper via `deps` (see [Persistent storage via Qdrant](#persistent-storage-via-qdrant-recommended-for-production)). Use `vector_db_backend` in `_capture_internal_state()` to check which backend is active at runtime.
6. **Single embedding client**: No multi-engine embedding abstraction. Embedding model behavior is determined by the injected `embedding_client`.

## Testing

- **Unit tests**: `test/test_node_memory_process.py` — covers process paths, fail-open scenarios, scope isolation, background extraction context, `top_k` defaults/boundaries, `source_id` roundtrip, handle overrides, and `_capture_internal_state`.
- **Integration tests**: `test/test_node_memory_integration.py` (11 tests) — full graph execution with mocked vector DB and LLM client, client provider sharing, graph validation, `run_agent(deps=...)` injection (vector_db + embedding_client).
- **Vector DB tests**: `test/test_qdrant_vector_db.py` (14 tests) — `QdrantVectorDB` wrapper unit tests using `QdrantClient(location=":memory:")` (skip-guarded if `qdrant-client` is not installed), including `source_id` roundtrip and UUID-formatted ID acceptance.
- **Production vector storage**: `magic_agents.vector_storage.InMemoryVectorDB` — production-grade ephemeral storage with `asyncio.Lock` concurrency safety, validated `MemoryEntry` reconstruction, and dimension-mismatch guards. `magic_agents.vector_storage.QdrantVectorDB` — optional Qdrant wrapper (requires `pip install magic-agents[qdrant]`).
- **Fixtures**: `mock_magic_embedding` (deterministic SHA256-based 8-dim unit vector) and `in_memory_vector_db` (fresh `InMemoryVectorDB` from `magic_agents.vector_storage`) in `test/conftest.py`.
