"""QdrantVectorDB — Qdrant-backed VectorDB wrapper with lazy collection lifecycle.

Single project-wide collection 'magic_llm_memory' (default).
Node/session isolation via payload FieldCondition filtering.
Default vector datatype: models.Datatype.FLOAT16 (explicit — NOT float32).

Requires qdrant-client >= 1.10.0 (optional [qdrant] extra).
"""

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

try:
    from qdrant_client import AsyncQdrantClient, models

    _QDRANT_AVAILABLE = True
except ImportError:
    AsyncQdrantClient = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    _QDRANT_AVAILABLE = False

from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryEntry


logger = logging.getLogger(__name__)


class QdrantVectorDB:
    """Qdrant-backed VectorDB wrapper with lazy collection lifecycle.

    Single project-wide collection 'magic_llm_memory' (default).
    Node/session isolation via payload FieldCondition filtering.
    Default vector datatype: models.Datatype.FLOAT16 (explicit).

    Usage::

        from qdrant_client import AsyncQdrantClient
        from magic_agents.vector_storage import QdrantVectorDB

        client = AsyncQdrantClient(url="http://localhost:6333")
        vdb = QdrantVectorDB(client=client, dim=1536)
    """

    def __init__(
        self,
        client: Any,  # AsyncQdrantClient (avoid direct type ref when not available)
        collection_name: str = "magic_llm_memory",
        dim: Optional[int] = None,
        dtype: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the QdrantVectorDB wrapper.

        Does NOT create any collection — all lifecycle is deferred to
        the first upsert() or search() call.

        Args:
            client: AsyncQdrantClient instance (caller manages lifecycle).
            collection_name: Qdrant collection name (default "magic_llm_memory").
            dim: Vector dimension. If None, auto-detected from first embedding.
            dtype: Vector datatype. Defaults to Datatype.FLOAT16 (NOT float32).
            **kwargs: Passed through to create_collection() for advanced config
                     (e.g., hnsw_config, quantization_config, on_disk_payload).
        """
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required to use QdrantVectorDB. "
                "Install: pip install magic-agents[qdrant]"
            )
        self._client: AsyncQdrantClient = client
        self._collection_name: str = collection_name
        self._dim: Optional[int] = dim
        self._dtype: Any = dtype if dtype is not None else "float16"
        self._extra_kwargs: dict[str, Any] = kwargs
        self._collection_ensured: bool = False

    async def _ensure_collection(self) -> None:
        """Ensure the Qdrant collection exists, creating it if necessary.

        Internal method called by upsert(), search(), and the public
        ensure_collection(). Idempotent — subsequent calls are no-ops
        once the collection is confirmed to exist.

        Raises:
            ValueError: If dim is not set, or if an existing collection
                       has a different dimension.
        """
        if self._collection_ensured:
            return

        if self._dim is None:
            raise ValueError(
                "_dim must be set before ensure_collection. "
                "Auto-detect from first embedding in caller."
            )

        exists = await self._client.collection_exists(self._collection_name)

        if not exists:
            # Map dtype string to Qdrant Datatype enum
            if self._dtype == "float16" or self._dtype is None:
                datatype = models.Datatype.FLOAT16
            else:
                datatype = self._dtype

            # Extract distance from extra_kwargs if provided (not a top-level
            # create_collection param — it belongs to VectorParams)
            distance = self._extra_kwargs.get(
                "distance", models.Distance.COSINE
            )

            vector_params = models.VectorParams(
                size=self._dim,
                distance=distance,
                datatype=datatype,
            )

            # Create the collection, passing remaining kwargs
            # (exclude "distance" since it was already consumed for VectorParams)
            create_kwargs = {
                k: v
                for k, v in self._extra_kwargs.items()
                if k != "distance"
            }
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=vector_params,
                **create_kwargs,
            )
        else:
            # Collection exists — verify dimension compatibility
            info = await self._client.get_collection(self._collection_name)
            # For single anonymous vector config, .vectors is a VectorParams
            existing_dim = info.config.params.vectors.size
            if existing_dim != self._dim:
                logger.warning(
                    "Collection '%s' exists with dim=%d, but requested dim=%d",
                    self._collection_name,
                    existing_dim,
                    self._dim,
                )
                raise ValueError(
                    f"Collection '{self._collection_name}' exists with "
                    f"dim={existing_dim}, but requested dim={self._dim}. "
                    f"Use a different collection name or delete the "
                    f"existing one."
                )

        self._collection_ensured = True

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Insert or update a vector entry.

        Lazy collection creation on first call. Auto-detects dimension
        from the embedding if not set at construction.

        Args:
            id: Document ID (content-hashed SHA256 for idempotency).
            embedding: Vector embedding as list of floats.
            metadata: Dict containing at minimum: memory_entry_id, content,
                     trigger, created_at, session_id, node_id.
                     "tags" key: if missing, defaults to [].

        Raises:
            ValueError: If the embedding dimension differs from the
                       existing collection dimension.
        """
        # Auto-detect dimension on first data operation
        if self._dim is None:
            self._dim = len(embedding)

        # Ensure collection exists (lazy — idempotent after first call)
        await self._ensure_collection()

        # Normalize tags: if not present in metadata, default to empty list.
        # tags MUST come ONLY from an explicit "tags" key in metadata — never
        # derived from "trigger" or any other field.
        if "tags" not in metadata:
            metadata["tags"] = []

        # Construct PointStruct with payload = metadata
        point = models.PointStruct(
            id=id,
            vector=embedding,
            payload=metadata,
        )
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[point],
            wait=True,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_scope: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Search for top-k similar vectors, filtered by scope.

        Lazy collection creation on first call. Auto-detects dimension
        from the query vector if not set at construction.

        Args:
            query_embedding: Query vector as list of floats.
            top_k: Maximum number of results (default 5).
            filter_scope: Dict with session_id, node_id for scope isolation.
                          When provided, only entries matching ALL filter
                          keys are returned. When None or empty, no filtering
                          is applied.

        Returns:
            List of MemoryEntry objects sorted by similarity DESC.
        """
        # Auto-detect dimension from query embedding
        if self._dim is None:
            self._dim = len(query_embedding)

        # Ensure collection exists (lazy)
        await self._ensure_collection()

        # Build Qdrant Filter from filter_scope dict
        # Each k,v pair becomes a FieldCondition with MatchValue
        query_filter: Optional[models.Filter] = None
        if filter_scope:
            conditions = [
                models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v),
                )
                for k, v in filter_scope.items()
            ]
            query_filter = models.Filter(must=conditions)

        # Query Qdrant
        result = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        # Map ScoredPoint results to validated MemoryEntry objects.
        # Uses regular MemoryEntry() constructor — NOT model_construct().
        # This ensures Pydantic validators run (e.g., content min_length=1).
        # The id field is NOT set, so Pydantic auto-generates a new UUID.
        results: list[MemoryEntry] = []
        for hit in result.points:
            payload = hit.payload
            results.append(MemoryEntry(
                content=payload["content"],
                trigger=payload.get("trigger", ""),
                created_at=(
                    datetime.fromisoformat(payload["created_at"])
                    if "created_at" in payload
                    else datetime.now()
                ),
                source_id=payload.get("doc_id"),  # NEW: populate from stored doc_id
            ))
        return results

    async def rebuild(
        self,
        entries: list[MemoryEntry],
        embed_fn: Callable[[str], Awaitable[list[float]]],
    ) -> None:
        """Batch-embed and upsert seed entries.

        Retained for backward compatibility with the VectorDB Protocol.
        NodeMemory NEVER calls this method. Collection lifecycle is managed
        by ensure_collection() — lazy on first data operation.

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
                    "tags": [],
                },
            )

    async def ensure_collection(
        self,
        collection_name: str,
        dim: Optional[int] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        """Ensure a collection exists, creating it if necessary.

        Implements the VectorDB Protocol lifecycle method.
        Sets instance config if provided, then delegates to the internal
        _ensure_collection().

        NodeMemory NEVER calls this method — collection lifecycle is
        managed internally by upsert() and search().

        Args:
            collection_name: Collection name (must match the instance name
                            for consistency).
            dim: Vector dimension. Sets self._dim if provided.
            dtype: Vector datatype. Sets self._dtype if provided.
        """
        if dim is not None:
            self._dim = dim
        if dtype is not None:
            self._dtype = dtype
        await self._ensure_collection()

    async def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """Delete a collection. Primarily for testing/admin cleanup.

        Args:
            collection_name: Name of the collection to delete.
        """
        await self._client.delete_collection(collection_name)
        self._collection_ensured = False
