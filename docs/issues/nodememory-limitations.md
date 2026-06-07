# NodeMemory Limitations

## Status

Documented limitation

## Summary

NodeMemory (`memory` node type) has several explicit design limitations documented in the spec, design, and implementation. These are intentional scope cuts, not bugs. They may be addressed in follow-up SDD changes.

## Detailed limitations

### 1. Embedding Client Required

Vector operations require an injected `embedding_client` whose `llm` object implements `async_embedding(...)`. The MagicLLM client delivered by `handle-client-provider` is used for extraction, not for embedding/search/upsert.

**Workaround**: Inject a compatible embedding client through `deps`. An OpenAI-compatible MagicLLM client is one option, but a custom/test client with `llm.async_embedding(...)` also satisfies the runtime contract.

### 2. No Codex Trigger Integration

NodeMemory and NodeCodex operate independently. There is no automatic mechanism to convert Codex triggers into memory tags. A `codex_entries` trigger like `"help"` does NOT automatically create a memory entry with `trigger="help"`.

**Workaround**: Graph authors can duplicate trigger values manually in both the `codex_entries` config and the `instructions` prompt text.

**Follow-up**: Codex-trigger-as-memory-tag integration requires a separate SDD that touches NodeCodex (`NodeCodex.py`, `CodexNodeModel.py`). This is explicitly out of scope for the NodeMemory change.

### 3. No Memory Eviction or Limit

The vector DB accumulates memories indefinitely. There is no count limit, LRU eviction, or expiration. This matches the spec non-goal: "No count limit, LRU eviction, or expiration. Vector DB accumulates indefinitely."

### 4. No Memory Deletion/Forgetting

NodeMemory only creates memories. There is no delete/forget UX or API. This matches the spec non-goal: "NodeMemory only creates memories. No delete/forget UX or API."

### 5. Vector Storage: Ephemeral Fallback by Default — Persistent Qdrant Requires Explicit Injection

NodeMemory **no longer degrades silently** when no `vector_db` is provided. It auto-creates an ephemeral `InMemoryVectorDB` instance in `__init__()` with an INFO-level log warning:

```
NodeMemory '{node_id}': No vector_db provided.
Auto-created ephemeral InMemoryVectorDB.
Memory data is LOST on process restart.
```

**Data loss**: The in-memory fallback does NOT persist to disk. All memories are lost when the process exits. Use persistent storage for production.

**Persistent storage**: Install the optional Qdrant extra and inject a `QdrantVectorDB` instance via graph deps:

```python
pip install magic-agents[qdrant]

from magic_agents.vector_storage import QdrantVectorDB
from qdrant_client import AsyncQdrantClient

client = AsyncQdrantClient(url="http://localhost:6333")
vector_db = QdrantVectorDB(client=client, dim=1536)

# Pass via build(..., deps=...) or run_agent(raw_graph_dict, deps=...)
deps = {"mem-1": {"vector_db": vector_db}}
```

**Diagnostic**: `vector_db_backend` in `_capture_internal_state()` reports the active backend — `"ephemeral"`, `"qdrant"`, or `"external"`.

**Remaining limitations** (intentional scope cuts, not bugs):

| Limitation | Details |
|-----------|---------|
| **No Qdrant archive/prune/migration** | There is no mechanism to archive old memories, prune by date/tag, or migrate between Qdrant collection configs. The single collection `magic_llm_memory` accumulates indefinitely. Archive/prune/migration are out of scope for the initial integration. |
| **Collection immutability / dimension mismatch** | Qdrant collections are immutable after creation (`dim`, `dtype`, `distance` cannot change). If the embedding model changes (e.g., 1536→3072 dims), the existing collection is incompatible. `QdrantVectorDB` raises a `ValueError` with a clear message. Manual intervention required: delete the collection or use a different `collection_name`. No automatic migration. |
| **Ephemeral persistence warning** | The `InMemoryVectorDB` fallback has zero durability guarantees. A process crash, restart, or container recycle destroys all stored memories. This is by design — `InMemoryVectorDB` is a development/default convenience, not a production storage tier. |
| **Single collection per project** | Both ephemeral and Qdrant backends use a single collection (`magic_llm_memory`). Node/session isolation relies on payload `FieldCondition` filtering. This is intentional to avoid per-node collection lifecycle overhead. |

### 6. Single Embedding Client Only

No multi-engine embedding abstraction layer exists. The embedding model is determined by the injected embedding client. If the embedding model changes between sessions, the same text may produce different embedding vectors, affecting search relevance.

## Related docs

- [../nodes/memory.md](../nodes/memory.md) — NodeMemory reference doc (includes QdrantVectorDB usage example)
- [../JSON_CONTRACT.md](../JSON_CONTRACT.md) — JSON graph contract with `vector_db_backend` and `vector_db_available` fields
- [`.dev/sdd/changes/nodememory-vector-storage/spec.md`](../../.dev/sdd/changes/nodememory-vector-storage/spec.md) — delta specification for vector storage
- [`.dev/sdd/changes/nodememory-vector-storage/design.md`](../../.dev/sdd/changes/nodememory-vector-storage/design.md) — architecture decisions and data flow
- `magic_agents/vector_storage/__init__.py` — package init: unconditional `InMemoryVectorDB`, conditional `QdrantVectorDB`
- `magic_agents/vector_storage/in_memory_vector_db.py` — production ephemeral `InMemoryVectorDB`
- `magic_agents/vector_storage/qdrant_vector_db.py` — `QdrantVectorDB` wrapper with lazy collection lifecycle
