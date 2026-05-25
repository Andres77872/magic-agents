"""Production in-memory vector DB with concurrency safety.

Ephemeral storage — all data is lost on process restart.
Satisfies the VectorDB Protocol (structural subtyping) with:

- In-memory dict storage protected by asyncio.Lock
- Dimension-mismatch guard on upsert
- Idempotent upsert (same ID replaces existing entry)
- Scope filtering via filter_scope (session_id AND node_id)
- Validated MemoryEntry reconstruction (no model_construct bypass)
- Search results sorted by cosine similarity DESC
- ensure_collection / delete_collection as no-ops
"""

import asyncio
import logging
import math
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryEntry


logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Brute-force cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either vector
        has zero magnitude.

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: len(a)={len(a)}, len(b)={len(b)}"
        )
    dot = sum(ax * bx for ax, bx in zip(a, b))
    norm_a = math.sqrt(sum(ax * ax for ax in a))
    norm_b = math.sqrt(sum(bx * bx for bx in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorDB:
    """Production in-memory vector DB with concurrency safety.

    NOT for persistent storage. All data lost on restart.
    Satisfies VectorDB Protocol via structural subtyping.

    - asyncio.Lock for thread-safe concurrent upsert/search
    - Dimension-mismatch guard on upsert
    - Validated MemoryEntry reconstruction (no model_construct bypass)
    - ensure_collection and delete_collection as no-ops
    """

    def __init__(self) -> None:
        """Initialize empty in-memory vector store."""
        self._entries: list[dict] = []
        self._lock: asyncio.Lock = asyncio.Lock()
        self._dim: int | None = None
        self._has_upserted: bool = False
        self._has_searched: bool = False

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Insert or update a vector entry.

        Idempotent — if an entry with the given id already exists,
        its embedding and metadata are replaced.

        Args:
            id: Document ID (content-hashed SHA256 for idempotency).
            embedding: Vector embedding as list of floats.
            metadata: Dict containing at minimum: memory_entry_id, content,
                     trigger, created_at, session_id, node_id.

        Raises:
            ValueError: If embedding dimension differs from previously
                       established dimension.
        """
        async with self._lock:
            # Dimension-mismatch guard: check inside lock for concurrent safety
            if self._dim is not None and len(embedding) != self._dim:
                raise ValueError(
                    f"Embedding dimension {len(embedding)} != expected {self._dim}"
                )

            # Auto-detect dimension on first upsert
            if self._dim is None:
                self._dim = len(embedding)

            # Log first upsert
            if not self._has_upserted:
                self._has_upserted = True
                logger.info(
                    "InMemoryVectorDB: first upsert — collection is accepting writes "
                    "(dim=%d)",
                    self._dim,
                )

            # Idempotent insert/replace
            for entry in self._entries:
                if entry["id"] == id:
                    entry["embedding"] = embedding
                    entry["metadata"] = metadata
                    return
            self._entries.append({
                "id": id,
                "embedding": embedding,
                "metadata": metadata,
            })

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_scope: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Search for top-k similar vectors, filtered by scope.

        Applies brute-force cosine similarity over all candidates.
        Results are sorted by similarity descending (highest first).

        Args:
            query_embedding: Query vector as list of floats.
            top_k: Maximum number of results (default 5).
            filter_scope: Dict with session_id and node_id for scope
                          isolation. When provided, only entries whose
                          metadata matches BOTH session_id AND node_id
                          are considered.

        Returns:
            List of MemoryEntry objects sorted by similarity DESC.
        """
        async with self._lock:
            # Log first search
            if not self._has_searched:
                self._has_searched = True
                logger.info(
                    "InMemoryVectorDB: first search — collection is being queried",
                )

            candidates = self._entries

            # Apply scope filter: match BOTH session_id AND node_id
            if filter_scope:
                session_id = filter_scope.get("session_id")
                node_id = filter_scope.get("node_id")
                candidates = [
                    e for e in candidates
                    if e["metadata"].get("session_id") == session_id
                    and e["metadata"].get("node_id") == node_id
                ]

            # Score: compute cosine similarity for each candidate
            scored: list[tuple[float, dict]] = []
            for entry in candidates:
                sim = _cosine_similarity(query_embedding, entry["embedding"])
                scored.append((sim, entry))

            # Sort DESC by similarity, take top_k
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:top_k]

            # Reconstruct MemoryEntry from stored metadata.
            # Uses validated MemoryEntry() constructor — NOT model_construct().
            # This ensures Pydantic validators run (e.g., content min_length=1).
            # The id field is NOT set, so Pydantic auto-generates a new UUID.
            results: list[MemoryEntry] = []
            for _sim, entry in scored:
                meta = entry["metadata"]
                results.append(MemoryEntry(
                    content=meta["content"],
                    trigger=meta.get("trigger", ""),
                    created_at=(
                        datetime.fromisoformat(meta["created_at"])
                        if "created_at" in meta
                        else datetime.now()
                    ),
                    source_id=meta.get("doc_id"),  # NEW: populate from stored doc_id
                ))
            return results

    async def rebuild(
        self,
        entries: list[MemoryEntry],
        embed_fn: Callable[[str], Awaitable[list[float]]],
    ) -> None:
        """Batch-embed and upsert seed entries.

        For each entry, computes embedding via embed_fn, then upserts
        using the entry's auto-generated id as the document ID.

        Args:
            entries: List of MemoryEntry to embed and index.
            embed_fn: Async callable that takes text and returns
                     embedding vector as list[float].
        """
        for entry in entries:
            embedding = await embed_fn(entry.content)
            await self.upsert(
                id=entry.id,
                embedding=embedding,
                metadata={
                    "memory_entry_id": entry.id,
                    "content": entry.content,
                    "trigger": entry.trigger,
                    "created_at": entry.created_at.isoformat(),
                },
            )

    async def ensure_collection(
        self,
        collection_name: str,
        dim: Optional[int] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        """No-op: in-memory store always exists.

        Logs at DEBUG level for lifecycle traceability.

        Args:
            collection_name: Ignored (no-op for in-memory).
            dim: Ignored (no-op for in-memory).
            dtype: Ignored (no-op for in-memory).
        """
        logger.debug(
            "InMemoryVectorDB.ensure_collection('%s') — no-op "
            "(flat in-memory store always exists)",
            collection_name,
        )

    async def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """No-op: in-memory store does not manage collections.

        Args:
            collection_name: Ignored (no-op for in-memory).
        """
        pass  # no-op
