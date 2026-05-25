"""
VectorDB Protocol for vector database operations.

Defines a structural subtyping Protocol (via typing.Protocol) for vector database
operations used by NodeMemory. Any object with matching method signatures is
accepted — no inheritance, no vector DB library coupling.

Imports only from the standard library and MagicMemoryNodeModel (which will be
created by Task 1.3). No vector DB libraries (ChromaDB, Qdrant, etc.) are
imported at the protocol level.
"""

from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Protocol,
)

from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryEntry


class VectorDB(Protocol):
    """Protocol for vector database operations in NodeMemory.

    Uses structural subtyping — any object with matching async method signatures
    is accepted as a VectorDB. This allows swapping implementations (ChromaDB,
    Qdrant, in-memory test double) without inheritance coupling.

    The filter_scope dict in search() MUST contain at minimum:
    - "session_id": str | int  — from ModelAgentRunLog.id_chat
    - "node_id": str           — from Node.node_id

    Implementations MUST enforce scope isolation — no global search without
    filter_scope.

    NOTE on rebuild(): rebuild() is retained in this Protocol for backward
    compatibility with existing implementations. NodeMemory NEVER calls rebuild().
    Collection lifecycle is managed by ensure_collection() — lazy on first data
    operation, called internally by upsert()/search().
    """

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_scope: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Search for top-k similar vectors, filtered by scope.

        Args:
            query_embedding: Query vector as list of floats.
            top_k: Maximum number of results (default 5).
            filter_scope: Dict with session_id and node_id for scope isolation.
                          MUST be enforced by implementations — no global search.

        Returns:
            List of MemoryEntry objects sorted by similarity DESC.
        """
        ...

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Insert or update a vector entry.

        Args:
            id: Document ID (content-hashed SHA256 for idempotency).
            embedding: Vector embedding as list of floats.
            metadata: Dict containing at minimum: memory_entry_id, content,
                     trigger, created_at, session_id, node_id.
        """
        ...

    async def rebuild(
        self,
        entries: list[MemoryEntry],
        embed_fn: Callable[[str], Awaitable[list[float]]],
    ) -> None:
        """Batch-embed and upsert seed entries on startup.

        rebuild() is retained for backward compatibility. NodeMemory NEVER calls
        rebuild(). Collection lifecycle is managed by ensure_collection() — lazy
        on first data operation.

        Args:
            entries: List of MemoryEntry to embed and index.
            embed_fn: Async callable that takes text and returns embedding
                      vector as list[float].
        """
        ...

    async def ensure_collection(
        self,
        collection_name: str,
        dim: Optional[int] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        """Ensure a collection exists, creating it if necessary.

        MUST be idempotent — calling multiple times with the same
        collection_name MUST NOT raise an error if the collection
        already exists.

        Args:
            collection_name: Name of the collection to ensure.
            dim: Vector dimension. Auto-detected from first embedding if None.
            dtype: Vector datatype. Implementation-specific default if None.
        """
        ...

    async def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """Delete a collection. Primarily for testing/admin cleanup.

        Args:
            collection_name: Name of the collection to delete.
        """
        ...
