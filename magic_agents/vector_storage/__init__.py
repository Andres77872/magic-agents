"""Vector storage implementations for NodeMemory.

Provides production-grade vector database implementations:

- InMemoryVectorDB — ephemeral, zero-dependency in-memory vector store.
  Auto-created when NodeMemory receives no vector_db dependency.
  All data is lost on process restart.

- QdrantVectorDB (optional) — Qdrant-backed persistent vector store.
  Requires qdrant-client >= 1.10.0 (pip install magic-agents[qdrant]).
  Single project-wide collection 'magic_llm_memory' with payload-based
  node/session isolation.
"""

from magic_agents.vector_storage.in_memory_vector_db import InMemoryVectorDB

__all__ = ["InMemoryVectorDB"]

# Conditional QdrantVectorDB: only made available when qdrant-client is
# installed. Import the flag first (the module always loads — it handles
# its own ImportError internally).
try:
    from magic_agents.vector_storage.qdrant_vector_db import _QDRANT_AVAILABLE
except ImportError:
    _QDRANT_AVAILABLE = False

if _QDRANT_AVAILABLE:
    from magic_agents.vector_storage.qdrant_vector_db import QdrantVectorDB  # noqa: F811

    __all__.append("QdrantVectorDB")
