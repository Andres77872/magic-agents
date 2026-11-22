"""Tests for the QdrantVectorDB wrapper.

All tests that require real Qdrant are guarded by ``pytest.importorskip``
at module level — they skip when ``qdrant-client`` is not installed.

Most tests use mocked ``AsyncQdrantClient`` to verify QdrantVectorDB behavior
(filter translation, lazy lifecycle, payload schema, etc.) without requiring
a real Qdrant server or in-memory Qdrant instance.

One integration-style test (``test_upsert_search_roundtrip``) exercises the
real in-process (``location=":memory:"``) Qdrant path for end-to-end coverage.
"""

import pytest
from packaging.version import Version
from unittest.mock import AsyncMock, MagicMock

qdrant = pytest.importorskip("qdrant_client")
# qdrant-client does not expose __version__ on the module, so we check
# version independently via importlib.metadata to enforce the minimum.
from importlib.metadata import version as _pkg_version

if Version(_pkg_version("qdrant-client")) < Version("1.10.0"):
    pytest.skip(
        f"qdrant-client >= 1.10.0 required, "
        f"found {_pkg_version('qdrant-client')}"
    )

from qdrant_client import AsyncQdrantClient, models
from magic_agents.vector_storage import QdrantVectorDB
from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryEntry


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_query_result(points: list) -> MagicMock:
    """Build a mock query_points result with the given scored points."""
    result = MagicMock()
    result.points = points
    return result


def _make_scored_point(
    id: str,
    score: float,
    payload: dict,
) -> MagicMock:
    """Build a mock ScoredPoint with the given id, score, and payload."""
    pt = MagicMock()
    pt.id = id
    pt.score = score
    pt.payload = payload
    return pt


# ─── Base class: mock client ───────────────────────────────────────────────


class TestQdrantVectorDB:
    """Tests for QdrantVectorDB wrapping behavior.

    All async tests use a mocked AsyncQdrantClient so they do NOT require
    a real Qdrant server or in-memory Qdrant process.
    """

    # ── Fixtures ──────────────────────────────────────────────────────────

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock AsyncQdrantClient.

        Default behavior:
        - collection_exists → False (no pre-existing collection)
        - All async methods are AsyncMock stubs
        """
        client = MagicMock()
        client.collection_exists = AsyncMock(return_value=False)
        client.create_collection = AsyncMock()
        client.upsert = AsyncMock()
        client.query_points = AsyncMock(
            return_value=_make_query_result([])
        )
        client.delete_collection = AsyncMock()
        client.get_collection = AsyncMock()

        # get_collection returns an info with config.params.vectors.size
        coll_info = MagicMock()
        coll_info.config.params.vectors.size = 384
        client.get_collection = AsyncMock(return_value=coll_info)

        return client

    # ── Tests ─────────────────────────────────────────────────────────────

    def test_default_collection_name(self, mock_client: MagicMock) -> None:
        """Default collection_name is 'magic_llm_memory'."""
        vdb = QdrantVectorDB(client=mock_client, dim=384)
        assert vdb._collection_name == "magic_llm_memory"

    def test_custom_collection_name(self, mock_client: MagicMock) -> None:
        """Custom collection_name is stored correctly."""
        vdb = QdrantVectorDB(
            client=mock_client,
            collection_name="custom",
            dim=384,
        )
        assert vdb._collection_name == "custom"

    @pytest.mark.asyncio
    async def test_lazy_collection_creation(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Collection is NOT created on init; created lazily on first upsert."""
        vdb = QdrantVectorDB(client=mock_client, dim=384)
        assert vdb._collection_ensured is False

        # First upsert triggers lazy collection creation
        await vdb.upsert("id1", [0.1, 0.2], {"content": "test"})

        mock_client.collection_exists.assert_called_once_with(
            "magic_llm_memory"
        )
        mock_client.create_collection.assert_called_once()
        assert vdb._collection_ensured is True

        # Second upsert must NOT trigger another collection check
        mock_client.collection_exists.reset_mock()
        mock_client.create_collection.reset_mock()

        await vdb.upsert("id2", [0.3, 0.4], {"content": "test2"})

        mock_client.collection_exists.assert_not_called()
        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_float16_default(self, mock_client: MagicMock) -> None:
        """VectorParams is created with Datatype.FLOAT16 by default."""
        vdb = QdrantVectorDB(client=mock_client, dim=384)
        await vdb._ensure_collection()

        mock_client.create_collection.assert_called_once()
        _call_args, call_kwargs = mock_client.create_collection.call_args
        vectors_config = call_kwargs["vectors_config"]

        assert vectors_config.size == 384
        assert vectors_config.distance == models.Distance.COSINE
        assert vectors_config.datatype == models.Datatype.FLOAT16

    @pytest.mark.asyncio
    async def test_dimension_auto_detect_from_upsert(
        self,
        mock_client: MagicMock,
    ) -> None:
        """When dim=None, dimension is inferred from first upsert embedding."""
        vdb = QdrantVectorDB(client=mock_client, dim=None)
        assert vdb._dim is None

        await vdb.upsert("id1", [0.1, 0.2, 0.3], {"content": "test"})

        assert vdb._dim == 3

        mock_client.create_collection.assert_called_once()
        _call_args, call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs["vectors_config"].size == 3

    @pytest.mark.asyncio
    async def test_search_before_upsert_no_dim(
        self,
        mock_client: MagicMock,
    ) -> None:
        """When dim=None, first search infers dim from query, creates collection,
        and returns empty results (freshly created collection has no points)."""
        mock_client.query_points = AsyncMock(
            return_value=_make_query_result([])
        )

        vdb = QdrantVectorDB(client=mock_client, dim=None)
        assert vdb._dim is None

        results = await vdb.search(
            [0.1, 0.2, 0.3],
            top_k=5,
            filter_scope={"node_id": "n1"},
        )

        assert vdb._dim == 3
        mock_client.collection_exists.assert_called_once_with(
            "magic_llm_memory"
        )
        mock_client.create_collection.assert_called_once()
        assert results == []

    @pytest.mark.asyncio
    async def test_filter_scope_translation(
        self,
        mock_client: MagicMock,
    ) -> None:
        """filter_scope dict translates to FieldCondition entries in Filter."""
        mock_client.query_points = AsyncMock(
            return_value=_make_query_result([])
        )
        vdb = QdrantVectorDB(client=mock_client, dim=384)

        await vdb.search(
            [0.1, 0.2],
            top_k=5,
            filter_scope={"node_id": "n1", "session_id": "s1"},
        )

        mock_client.query_points.assert_called_once()
        _call_args, call_kwargs = mock_client.query_points.call_args
        query_filter = call_kwargs["query_filter"]

        assert query_filter is not None
        assert len(query_filter.must) == 2

        conditions = {
            (c.key, c.match.value) for c in query_filter.must
        }
        assert ("node_id", "n1") in conditions
        assert ("session_id", "s1") in conditions

    @pytest.mark.asyncio
    async def test_no_filter_scope(
        self,
        mock_client: MagicMock,
    ) -> None:
        """When filter_scope is None, query_filter is None (no filtering)."""
        mock_client.query_points = AsyncMock(
            return_value=_make_query_result([])
        )
        vdb = QdrantVectorDB(client=mock_client, dim=384)

        await vdb.search([0.1, 0.2], top_k=5, filter_scope=None)

        mock_client.query_points.assert_called_once()
        _call_args, call_kwargs = mock_client.query_points.call_args
        assert call_kwargs.get("query_filter") is None

    @pytest.mark.asyncio
    async def test_payload_schema(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Upsert passes metadata as PointStruct payload with all fields."""
        vdb = QdrantVectorDB(client=mock_client, dim=384)
        metadata = {
            "node_id": "n1",
            "content": "memory text",
            "tags": ["tag1", "tag2"],
            "creation_date": "2026-01-01T00:00:00",
            "trigger": "test_trigger",
        }

        await vdb.upsert("id1", [0.1, 0.2], metadata)

        mock_client.upsert.assert_called_once()
        _call_args, call_kwargs = mock_client.upsert.call_args
        points = call_kwargs["points"]
        assert len(points) == 1

        point = points[0]
        assert point.id == "id1"
        assert point.vector == [0.1, 0.2]

        payload = point.payload
        assert payload["node_id"] == "n1"
        assert payload["content"] == "memory text"
        assert payload["tags"] == ["tag1", "tag2"]
        assert payload["creation_date"] == "2026-01-01T00:00:00"
        assert payload["trigger"] == "test_trigger"

    @pytest.mark.asyncio
    async def test_tags_defaults_to_empty_list(
        self,
        mock_client: MagicMock,
    ) -> None:
        """Metadata without 'tags' key stores 'tags': [] in payload."""
        vdb = QdrantVectorDB(client=mock_client, dim=384)
        metadata = {
            "node_id": "n1",
            "content": "no tags here",
        }

        await vdb.upsert("id1", [0.1, 0.2], metadata)

        mock_client.upsert.assert_called_once()
        _call_args, call_kwargs = mock_client.upsert.call_args
        point = call_kwargs["points"][0]
        assert point.payload["tags"] == []

    @pytest.mark.asyncio
    async def test_upsert_search_roundtrip(self) -> None:
        """Full round-trip: upsert → search → reconstructed MemoryEntry.

        Uses real in-memory Qdrant (AsyncQdrantClient with location=":memory:")
        for end-to-end coverage.

        Qdrant local mode requires point IDs to be valid UUIDs.  We use
        uuid4().hex to match the format generated by MemoryEntry.
        """
        import uuid

        id_1 = uuid.uuid4().hex
        id_2 = uuid.uuid4().hex

        client = AsyncQdrantClient(location=":memory:")
        try:
            vdb = QdrantVectorDB(client=client, dim=3)

            await vdb.upsert(
                id_1,
                [1.0, 0.0, 0.0],
                {
                    "node_id": "n1",
                    "content": "hello",
                    "tags": ["greeting"],
                    "creation_date": "2026-01-01T00:00:00",
                    "trigger": "greeting",
                },
            )
            # Second point with different node_id (should not match n1 filter)
            await vdb.upsert(
                id_2,
                [0.0, 1.0, 0.0],
                {
                    "node_id": "n2",
                    "content": "other node memory",
                    "tags": [],
                    "creation_date": "2026-01-01T00:00:00",
                },
            )

            # Search with node_id "n1" should return only the first point
            results = await vdb.search(
                [1.0, 0.0, 0.0],
                top_k=5,
                filter_scope={"node_id": "n1"},
            )

            assert len(results) == 1
            assert isinstance(results[0], MemoryEntry)
            assert results[0].content == "hello"

            # Verify filter_scope correctly isolates by node_id
            n2_results = await vdb.search(
                [1.0, 0.0, 0.0],
                top_k=5,
                filter_scope={"node_id": "n2"},
            )
            assert len(n2_results) == 1
            assert n2_results[0].content == "other node memory"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_source_id_roundtrip(self) -> None:
        """Upsert with doc_id in payload → search → source_id matches.

        Verifies that QdrantVectorDB.search() correctly populates
        MemoryEntry.source_id from the stored doc_id payload field.
        Uses real in-memory Qdrant (AsyncQdrantClient with location=":memory:").
        """
        import hashlib
        import uuid

        doc_id = str(uuid.UUID(hashlib.sha256(b"test-content").hexdigest()[:32]))

        client = AsyncQdrantClient(location=":memory:")
        try:
            vdb = QdrantVectorDB(client=client, dim=3)

            await vdb.upsert(
                doc_id,
                [1.0, 0.0, 0.0],
                {
                    "doc_id": doc_id,
                    "node_id": "n1",
                    "content": "hello",
                    "tags": [],
                    "creation_date": "2026-01-01T00:00:00",
                    "trigger": "greeting",
                },
            )

            results = await vdb.search(
                [1.0, 0.0, 0.0],
                top_k=5,
                filter_scope={"node_id": "n1"},
            )

            assert len(results) == 1
            assert isinstance(results[0], MemoryEntry)
            assert results[0].source_id == doc_id, (
                f"Expected source_id={doc_id!r}, got {results[0].source_id!r}"
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_uuid_formatted_id_acceptance(self) -> None:
        """Qdrant local mode accepts UUID-formatted IDs from NodeMemory ID scheme.

        Constructs a UUID-formatted ID using the exact scheme from NodeMemory:
        str(uuid.UUID(sha256(input).hexdigest()[:32])). Verifies upsert
        succeeds and source_id matches the input ID on search.
        Uses real in-memory Qdrant (AsyncQdrantClient with location=":memory:").
        """
        import hashlib
        import uuid

        uuid_id = str(uuid.UUID(hashlib.sha256(b"test").hexdigest()[:32]))

        client = AsyncQdrantClient(location=":memory:")
        try:
            vdb = QdrantVectorDB(client=client, dim=3)

            await vdb.upsert(
                uuid_id,
                [1.0, 0.0, 0.0],
                {
                    "doc_id": uuid_id,
                    "node_id": "n1",
                    "content": "uuid formatted id test",
                    "tags": [],
                    "creation_date": "2026-01-01T00:00:00",
                    "trigger": "test",
                },
            )

            results = await vdb.search(
                [1.0, 0.0, 0.0],
                top_k=5,
                filter_scope={"node_id": "n1"},
            )

            assert len(results) == 1
            assert isinstance(results[0], MemoryEntry)
            assert results[0].source_id == uuid_id, (
                f"Expected source_id={uuid_id!r}, got {results[0].source_id!r}"
            )

            # Verify the ID is a valid UUID format
            parsed = uuid.UUID(uuid_id)
            assert str(parsed) == uuid_id, (
                f"UUID round-trip failed: {uuid_id} → {parsed}"
            )
        finally:
            await client.close()
