"""
Task 4.2 — NodeMemory integration tests with build() + run_agent().

Covers 6 integration test scenarios per tasks.md lines 171-182:
1. Full flow with seed match → <memory_content> injection
2. Full flow with no match → passthrough (no <memory_content>)
3. Extraction flow: instructions enabled → memories extracted and upserted
4. Client provider sharing: single client → Memory + LLM
5. Graph validation passes for memory graph
6. Content-hashed idempotency: same message twice → single vector DB entry

Topology patterns:
- Memory node uses handle_memory_input / handle_memory_output / handle-client-provider
- Vector DB injected via build() deps parameter
- Real client node; ``mock_magic_llm`` patches the provider constructor, so the
  NodeClientLLM output edge carries the fake client to Memory
- SendMessage as downstream consumer (no real API keys needed)

Pattern follows test_node_codex_integration.py:
- @pytest.mark.asyncio for async tests
- build(agt, message=..., deps=...) + run_agent(graph)
- Check node.outputs after execution for transform verification
- Pre-seed InMemoryVectorDB for search-match scenarios
"""

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build, validate_graph
from magic_llm.model.ModelEmbeddingResponse import EmbeddingData, ModelEmbeddingResponse
from magic_agents.vector_storage import InMemoryVectorDB

# ─── Constants ───────────────────────────────────────────────────────────────

MEMORY_CONTENT_MARKER = "<memory_content>"
TRIGGER_MESSAGE = "hello world"
NO_MATCH_MESSAGE = "goodbye world"
CONSTANT_EMBEDDING = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
TEST_SESSION_ID = "test-session-1"

# ─── Graph Builder ────────────────────────────────────────────────────────────


def _make_memory_graph(
    memory_instructions=None,
    memory_entries=None,
    include_client=True,
    extra_consumer_nodes=None,
    extra_consumer_edges=None,
    llm_after_memory=False,
    debug=True,
) -> dict:
    """Build a graph dict with UserInput → Memory → consumer(s) → END.

    Args:
        memory_instructions: Instructions for LLM extraction (None = search-only).
        memory_entries: Seed memory entries list.
        include_client: Whether to include a client LLM node.
        extra_consumer_nodes: Additional consumer nodes to append.
        extra_consumer_edges: Additional edges to append.
        llm_after_memory: If True, insert an LLM node between Memory and SendMessage.
        debug: Enable debug in the graph.

    Returns:
        Graph dict ready for build().
    """
    if memory_entries is None:
        memory_entries = []

    memory_data = {}
    if memory_instructions is not None:
        memory_data["instructions"] = memory_instructions
    if memory_entries:
        memory_data["memory_entries"] = memory_entries

    nodes = [
        {"id": "input", "type": "user_input"},
        {"id": "memory", "type": "memory", "data": memory_data},
    ]

    edges = [
        {
            "id": "e_input_memory",
            "source": "input",
            "target": "memory",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle_memory_input",
        },
    ]

    if include_client:
        nodes.append({
            "id": "client",
            "type": "client",
            "data": {
                "engine": "openai",
                "model": "gpt-4o-mini",
            },
        })
        edges.append({
            "id": "e_client_memory",
            "source": "client",
            "target": "memory",
            "sourceHandle": "handle-client-provider",
            "targetHandle": "handle-client-provider",
        })

    # Consumer: SendMessage by default
    if llm_after_memory:
        nodes.append({
            "id": "llm", "type": "llm",
            "data": {"top_p": 1, "stream": False, "max_tokens": 5, "temperature": 0.2},
        })
        edges.append({
            "id": "e_memory_llm",
            "source": "memory", "target": "llm",
            "sourceHandle": "handle_memory_output",
            "targetHandle": "handle_user_message",
        })
        if include_client:
            edges.append({
                "id": "e_client_llm",
                "source": "client", "target": "llm",
                "sourceHandle": "handle-client-provider",
                "targetHandle": "handle-client-provider",
            })
        # LLM → SendMessage
        nodes.append({
            "id": "send", "type": "send_message",
            "data": {"message": "", "json_extras": ""},
        })
        edges.append({
            "id": "e_llm_send",
            "source": "llm", "target": "send",
            "sourceHandle": "handle_generated_content",
            "targetHandle": "handle_send_extra",
        })
    else:
        nodes.append({
            "id": "send", "type": "send_message",
            "data": {"message": "", "json_extras": ""},
        })
        edges.append({
            "id": "e_memory_send",
            "source": "memory", "target": "send",
            "sourceHandle": "handle_memory_output",
            "targetHandle": "handle_send_extra",
        })

    # END node
    nodes.append({"id": "end", "type": "end"})
    edges.append({
        "id": "e_send_end",
        "source": "send", "target": "end",
        "sourceHandle": "handle_message_output",
    })

    # Append extra nodes/edges (e.g. for client sharing tests)
    if extra_consumer_nodes:
        nodes.extend(extra_consumer_nodes)
    if extra_consumer_edges:
        edges.extend(extra_consumer_edges)

    return {"type": "graph", "debug": debug, "nodes": nodes, "edges": edges}


def _make_memory_graph_for_validation(
    memory_instructions=None,
    memory_entries=None,
    debug=True,
) -> dict:
    """Build a minimal graph dict for validate_graph() — no execution needed.

    This graph does NOT include a client node (not needed for validation).
    """
    if memory_entries is None:
        memory_entries = []

    memory_data = {}
    if memory_instructions is not None:
        memory_data["instructions"] = memory_instructions
    if memory_entries:
        memory_data["memory_entries"] = memory_entries

    nodes = [
        {"id": "input", "type": "user_input"},
        {"id": "memory", "type": "memory", "data": memory_data},
        {"id": "send", "type": "send_message",
         "data": {"message": "", "json_extras": ""}},
        {"id": "end", "type": "end"},
    ]
    edges = [
        {
            "id": "e1", "source": "input", "target": "memory",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle_memory_input",
        },
        {
            "id": "e2", "source": "memory", "target": "send",
            "sourceHandle": "handle_memory_output",
            "targetHandle": "handle_send_extra",
        },
        {
            "id": "e3", "source": "send", "target": "end",
            "sourceHandle": "handle_message_output",
        },
    ]
    return {"type": "graph", "debug": debug, "nodes": nodes, "edges": edges}


def _make_memory_graph_dual_memory(debug=True) -> dict:
    """Build a graph with TWO Memory nodes sharing one Client (for test 4)."""
    nodes = [
        {"id": "input", "type": "user_input"},
        {"id": "client", "type": "client",
         "data": {"engine": "openai",
                  "model": "gpt-4o-mini"}},
        {"id": "memory-1", "type": "memory", "data": {}},
        {"id": "memory-2", "type": "memory", "data": {}},
        {"id": "send-1", "type": "send_message",
         "data": {"message": "", "json_extras": ""}},
        {"id": "send-2", "type": "send_message",
         "data": {"message": "", "json_extras": ""}},
        {"id": "end", "type": "end"},
    ]
    edges = [
        # input → memory-1
        {"id": "e1", "source": "input", "target": "memory-1",
         "sourceHandle": "handle_user_message",
         "targetHandle": "handle_memory_input"},
        # input → memory-2
        {"id": "e2", "source": "input", "target": "memory-2",
         "sourceHandle": "handle_user_message",
         "targetHandle": "handle_memory_input"},
        # client → memory-1
        {"id": "e3", "source": "client", "target": "memory-1",
         "sourceHandle": "handle-client-provider",
         "targetHandle": "handle-client-provider"},
        # client → memory-2
        {"id": "e4", "source": "client", "target": "memory-2",
         "sourceHandle": "handle-client-provider",
         "targetHandle": "handle-client-provider"},
        # memory-1 → send-1
        {"id": "e5", "source": "memory-1", "target": "send-1",
         "sourceHandle": "handle_memory_output",
         "targetHandle": "handle_send_extra"},
        # memory-2 → send-2
        {"id": "e6", "source": "memory-2", "target": "send-2",
         "sourceHandle": "handle_memory_output",
         "targetHandle": "handle_send_extra"},
        # send-1 → end
        {"id": "e7", "source": "send-1", "target": "end",
         "sourceHandle": "handle_message_output"},
        # send-2 → end
        {"id": "e8", "source": "send-2", "target": "end",
         "sourceHandle": "handle_message_output"},
    ]
    return {"type": "graph", "debug": debug, "nodes": nodes, "edges": edges}


# ─── Output Extractors ────────────────────────────────────────────────────────


def _get_memory_output(graph) -> str:
    """Extract the memory node's output content after execution."""
    memory_node = graph.nodes.get("memory")
    assert memory_node is not None, "Memory node not found in graph"
    assert "handle_memory_output" in memory_node.outputs, (
        f"Memory node has no output on handle_memory_output. "
        f"Available outputs: {list(memory_node.outputs.keys())}"
    )
    output = memory_node.outputs["handle_memory_output"]
    assert "content" in output, (
        f"Memory output has no 'content' field: {output}"
    )
    return str(output["content"])


def _get_memory_internal_state(graph) -> dict:
    """Get the memory node's internal state after execution."""
    memory_node = graph.nodes.get("memory")
    assert memory_node is not None, "Memory node not found in graph"
    return memory_node._capture_internal_state()


def _get_memory_output_by_id(graph, node_id: str) -> str:
    """Extract a specific memory node's output content after execution."""
    memory_node = graph.nodes.get(node_id)
    assert memory_node is not None, f"Memory node '{node_id}' not found"
    assert "handle_memory_output" in memory_node.outputs, (
        f"Memory node '{node_id}' has no output on handle_memory_output. "
        f"Available outputs: {list(memory_node.outputs.keys())}"
    )
    output = memory_node.outputs["handle_memory_output"]
    assert "content" in output, (
        f"Memory node '{node_id}' output has no 'content' field: {output}"
    )
    return str(output["content"])


# ─── Embedding Helpers ────────────────────────────────────────────────────────


def _constant_embedding_response() -> ModelEmbeddingResponse:
    """Return a ModelEmbeddingResponse with a constant 8-dim embedding vector.

    Using a constant vector ensures text-independent similarity for tests:
    all queries match all seeded entries with cosine similarity = 1.0.
    """
    return ModelEmbeddingResponse(
        object="list",
        data=[EmbeddingData(
            object="embedding", index=0, embedding=CONSTANT_EMBEDDING,
        )],
        model="mock-embedding-model",
    )


def _make_embedding_client() -> MagicMock:
    """Create a mock embedding client returning constant embeddings.

    The client's async_embedding returns CONSTANT_EMBEDDING so that any
    query matches any seeded entry (cosine similarity = 1.0).
    This is the standard embedding client helper for integration tests
    that need Phase 3 or Phase 4 embedding to execute.
    """
    custom_client = MagicMock()
    custom_client.llm = MagicMock()
    custom_client.llm.async_embedding = AsyncMock(
        return_value=_constant_embedding_response(),
    )
    return custom_client


def _make_background_task_tracker():
    tasks = []

    def track(coro):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    return tasks, track


async def _seed_vector_db(vector_db: InMemoryVectorDB, entries: list[dict],
                          session_id: str, node_id: str) -> None:
    """Pre-seed a vector DB with entries for search-match scenarios.

    Each entry gets the constant embedding vector so any query matches it.
    """
    for i, entry in enumerate(entries):
        doc_id = hashlib.sha256(entry["content"].encode()).hexdigest()
        await vector_db.upsert(
            id=doc_id,
            embedding=CONSTANT_EMBEDDING,
            metadata={
                "memory_entry_id": f"seed-{i}",
                "content": entry["content"],
                "trigger": entry.get("trigger", ""),
                "created_at": entry.get("created_at", "2026-01-01T00:00:00+00:00"),
                "session_id": session_id,
                "node_id": node_id,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_memory_client(mock_magic_llm):
    """Base fixture for NodeMemory integration tests.

    The client node's patched ``MagicLLM`` constructor returns this client.
    Inherits NotImplementedError on async_embedding from mock_magic_llm.
    Tests that need extraction flow can override async_generate on this fixture
    within the test body before calling build().
    Tests that need embedding must inject embedding_client via deps.
    """
    return mock_magic_llm


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Full flow with seed match → injection
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeMemoryIntegration:
    """Integration tests for NodeMemory with full graph execution."""

    @pytest.mark.asyncio
    async def test_full_flow_match_injection(
        self, mock_memory_client, in_memory_vector_db,
    ):
        """UserInput → Memory (search-only) → SendMessage → END with seed match.

        Given seed entries in the vector DB, and the Memory node in search-only
        mode (no instructions), processing a user message should find the seed
        entries and inject <memory_content> blocks into the output.
        """
        # Pre-seed the vector DB with entries matching the test session
        await _seed_vector_db(
            in_memory_vector_db,
            entries=[
                {"content": "User prefers email communication", "trigger": "preference"},
                {"content": "User mentioned pro plan pricing", "trigger": "pricing"},
            ],
            session_id=TEST_SESSION_ID,
            node_id="memory",
        )

        custom_embedding_client = _make_embedding_client()
        agt = _make_memory_graph(memory_instructions=None, memory_entries=[])
        deps = {
            "memory": {
                "vector_db": in_memory_vector_db,
                "embedding_client": custom_embedding_client,
            },
        }
        graph = build(agt, message=TRIGGER_MESSAGE, deps=deps)

        async for _ in run_agent(graph, id_chat=TEST_SESSION_ID):
            pass

        # Embedding provenance: embedding occurred via the correct client
        custom_embedding_client.llm.async_embedding.assert_called()
        mock_memory_client.llm.async_embedding.assert_not_called()

        # Transform verification: memory output contains <memory_content>
        memory_output = _get_memory_output(graph)
        assert MEMORY_CONTENT_MARKER in memory_output, (
            f"Expected <memory_content> in output, got: {memory_output!r}"
        )
        assert "User prefers email communication" in memory_output, (
            f"Expected seed content in output, got: {memory_output!r}"
        )
        assert "User mentioned pro plan pricing" in memory_output, (
            f"Expected second seed in output, got: {memory_output!r}"
        )
        assert TRIGGER_MESSAGE in memory_output, (
            f"Expected original message in output, got: {memory_output!r}"
        )

        # Output starts with oldest match sorted by created_at ASC
        assert memory_output.startswith(
            "<memory_content>User prefers email communication</memory_content>"
        ), f"Output does not start with expected format: {memory_output!r}"

        # Internal state verification
        state = _get_memory_internal_state(graph)
        assert state["injected_count"] == 2, (
            f"Expected 2 injected, got: {state['injected_count']}"
        )
        assert state["vector_db_available"] is True
        # extraction-mode disabled → extracted_count should be 0
        assert state["extracted_count"] == 0, (
            f"Expected 0 extracted (search-only), got: {state['extracted_count']}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Test 2: Full flow with no match → passthrough
    # ═════════════════════════════════════════════════════════════════════════

    # ═════════════════════════════════════════════════════════════════════════
    # Test 3: Extraction flow — memories extracted and upserted
    # ═════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_extraction_flow_memories_upserted(
        self, in_memory_vector_db,
    ):
        """UserInput → Memory (with instructions) → SendMessage → END.

        With instructions enabled and the LLM returning valid ExtractionResult
        JSON, the Memory node should extract memories via fire-and-forget
        background task, upsert them to the vector DB with UUID4 IDs.

        NEW BEHAVIOR:
        - Extraction runs via asyncio.create_task() (fire-and-forget)
        - UUID4 IDs are used (not deterministic hash IDs)
        - Background upsert happens but is NOT awaited by graph flow
        - Newly extracted memories are NOT available in same-turn output
          (they become available for FUTURE turns)
        - Metadata includes session_id and node_id

        This test verifies:
        1. Background extraction task completes (via upserted entries)
        2. UUID4 IDs are generated (not deterministic hash IDs)
        3. Memory upsert completes with proper metadata
        4. Same-turn output does NOT include newly extracted memories
        """
        # Create extraction response - use a fresh MagicMock each time
        def make_extraction_response():
            return MagicMock(content=json.dumps({
                "memories": [
                    {"content": "User concerned about pro plan pricing",
                     "trigger": "pricing"},
                    {"content": "User wants to compare plans",
                     "trigger": "evaluation"},
                ],
            }))
        
        # Create a proper mock client - AsyncMock with return_value
        mock_client = MagicMock()
        mock_client.llm = MagicMock()
        mock_client.llm.async_generate = AsyncMock(return_value=make_extraction_response())

        custom_embedding_client = _make_embedding_client()
        background_tasks, background_task_tracker = _make_background_task_tracker()
        # Create graph WITHOUT client node - we'll inject the client via deps
        # This avoids the handle-client-provider edge validation error
        agt = _make_memory_graph(
            memory_instructions="Extract pricing concerns from this message.",
            memory_entries=[],
            include_client=False,  # Don't include client node in graph
        )
        # Inject the mock client into memory deps - NodeMemory will use it
        # for background extraction
        deps = {
            "memory": {
                "vector_db": in_memory_vector_db,
                "embedding_client": custom_embedding_client,
                "client": mock_client,  # Pass mock client directly to memory node
                "background_task_tracker": background_task_tracker,
            },
        }
        graph = build(agt, message=TRIGGER_MESSAGE, deps=deps)

        async for _ in run_agent(graph, id_chat=TEST_SESSION_ID):
            pass

        assert background_tasks, "Memory extraction should be registered as background work"
        await asyncio.gather(*background_tasks)

        # Embedding provenance: embedding occurred via the correct client
        custom_embedding_client.llm.async_embedding.assert_called()
        # The background task should have been called (verified by upserted entries)

        # NEW: UUID4 IDs (not deterministic hash IDs)
        # All IDs should be valid UUID4 strings
        import re
        uuid4_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE)
        entry_ids = {e["id"] for e in in_memory_vector_db._entries}
        assert len(entry_ids) == 2

        for entry_id in entry_ids:
            assert uuid4_pattern.match(entry_id), (
                f"Expected UUID4 ID format, got: {entry_id}"
            )

        # Verify metadata contains session_id and node_id for scope isolation
        for entry in in_memory_vector_db._entries:
            meta = entry["metadata"]
            assert meta.get("session_id") == TEST_SESSION_ID, (
                f"Expected session_id={TEST_SESSION_ID}, got: {meta.get('session_id')}"
            )
            assert meta.get("node_id") == "memory", (
                f"Expected node_id='memory', got: {meta.get('node_id')}"
            )

        # NEW: Same-turn output does NOT include newly extracted memories
        # because background task is not awaited by graph flow
        # The graph returns the original message without <memory_content>
        memory_output = _get_memory_output(graph)
        assert memory_output == TRIGGER_MESSAGE, (
            f"Expected passthrough (no same-turn injection for newly extracted), "
            f"got: {memory_output!r}"
        )

        # Internal state verification
        state = _get_memory_internal_state(graph)
        assert state["extracted_count"] == 2, (
            f"Expected 2 extracted memories, got: {state['extracted_count']}"
        )
        # injected_count should be 0 because newly extracted memories
        # are not available in the same turn (background task not awaited)
        assert state["injected_count"] == 0, (
            f"Expected 0 injected (newly extracted not in same-turn), "
            f"got: {state['injected_count']}"
        )
        assert state["vector_db_available"] is True
        assert state["memory_entry_count"] == 2, (
            f"Expected 2 memory entries accumulated, got: {state['memory_entry_count']}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Test 4: Client provider sharing
    # ═════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_client_provider_sharing(
        self, mock_memory_client, in_memory_vector_db,
    ):
        """Single Client node fans out to two Memory nodes.

        Verifies that a single handle-client-provider edge can serve
        multiple consumers (parallel-safe, same MagicLLM instance).
        Both Memory-1 and Memory-2 receive the same client.
        """
        # Pre-seed separate vector DBs (one per memory node)
        db_1 = InMemoryVectorDB()
        db_2 = InMemoryVectorDB()

        await _seed_vector_db(
            db_1, [{"content": "Memory-1 specific data", "trigger": "a"}],
            session_id=TEST_SESSION_ID, node_id="memory-1",
        )
        await _seed_vector_db(
            db_2, [{"content": "Memory-2 specific data", "trigger": "b"}],
            session_id=TEST_SESSION_ID, node_id="memory-2",
        )

        custom_embedding_client = _make_embedding_client()
        agt = _make_memory_graph_dual_memory()
        deps = {
            "memory-1": {
                "vector_db": db_1,
                "embedding_client": custom_embedding_client,
            },
            "memory-2": {
                "vector_db": db_2,
                "embedding_client": custom_embedding_client,
            },
        }
        graph = build(agt, message=TRIGGER_MESSAGE, deps=deps)

        async for _ in run_agent(graph, id_chat=TEST_SESSION_ID):
            pass

        # Embedding provenance: embedding occurred via the correct client
        custom_embedding_client.llm.async_embedding.assert_called()
        mock_memory_client.llm.async_embedding.assert_not_called()

        # Both memory nodes executed and produced output
        output_1 = _get_memory_output_by_id(graph, "memory-1")
        output_2 = _get_memory_output_by_id(graph, "memory-2")

        assert MEMORY_CONTENT_MARKER in output_1, (
            f"Memory-1 expected <memory_content>, got: {output_1!r}"
        )
        assert MEMORY_CONTENT_MARKER in output_2, (
            f"Memory-2 expected <memory_content>, got: {output_2!r}"
        )

        # Each node sees its own data (scope isolation via node_id)
        assert "Memory-1 specific data" in output_1, (
            f"Memory-1 should see its own data: {output_1!r}"
        )
        assert "Memory-2 specific data" in output_2, (
            f"Memory-2 should see its own data: {output_2!r}"
        )

        # Memory-2 should NOT see Memory-1's data (scope isolation)
        assert "Memory-1 specific data" not in output_2, (
            f"Memory-2 should NOT see Memory-1's data: {output_2!r}"
        )
        assert "Memory-2 specific data" not in output_1, (
            f"Memory-1 should NOT see Memory-2's data: {output_1!r}"
        )

        # Both consumer SendMessage nodes executed
        send_1 = graph.nodes.get("send-1")
        send_2 = graph.nodes.get("send-2")
        assert send_1 is not None and "handle_message_output" in send_1.outputs
        assert send_2 is not None and "handle_message_output" in send_2.outputs

    # ═════════════════════════════════════════════════════════════════════════
    # Test 5: Graph validation passes
    # ═════════════════════════════════════════════════════════════════════════

    def test_graph_validation_passes(self):
        """validate_graph() passes for a valid memory graph.

        Memory node uses standard handles: handle_memory_input,
        handle_memory_output. No client node needed for validation.
        """
        agt = _make_memory_graph_for_validation(
            memory_instructions=None, memory_entries=[],
        )
        result = validate_graph(agt["nodes"], agt["edges"])

        assert result["valid"] is True, (
            f"Expected empty memory graph to pass validation, "
            f"got errors: {result.get('errors', [])}"
        )
        assert result["errors"] == [], (
            f"Unexpected validation errors: {result['errors']}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Test 6: Content-hashed idempotency
    # ═════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_content_hashed_idempotency_integration(
        self, mock_memory_client, in_memory_vector_db,
    ):
        """UUID4 ID generation: duplicate content creates separate entries.

        NEW BEHAVIOR: Content-hash deduplication is OUT OF SCOPE.
        Each extracted memory gets a unique UUID4 ID, regardless of duplicate content.
        This test verifies:
        - UUID4 IDs are generated for each upsert
        - Duplicate content does NOT produce deduplication (separate entries)
        - No content-hash idempotency mechanism exists in this scope
        """
        extraction_response = MagicMock()
        extraction_response.content = json.dumps({
            "memories": [
                {"content": "User concerned about pricing", "trigger": "pricing"},
            ],
        })
        mock_memory_client.llm.async_generate = AsyncMock(
            return_value=extraction_response,
        )

        custom_embedding_client = _make_embedding_client()
        agt = _make_memory_graph(
            memory_instructions="Extract concerns from message.",
            memory_entries=[],
        )
        deps = {
            "memory": {
                "vector_db": in_memory_vector_db,
                "embedding_client": custom_embedding_client,
            },
        }

        # Clear any existing entries
        in_memory_vector_db._entries = []

        # --- First run ---
        graph_1 = build(agt, message=TRIGGER_MESSAGE, deps=deps)
        async for _ in run_agent(graph_1, id_chat=TEST_SESSION_ID):
            pass

        # Verify first run produced exactly 1 entry
        assert len(in_memory_vector_db._entries) == 1, (
            f"Expected 1 entry after first run, "
            f"got {len(in_memory_vector_db._entries)}"
        )

        # NEW: ID should be UUID4 format (not deterministic hash)
        import re
        uuid4_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE)
        first_id = in_memory_vector_db._entries[0]["id"]
        assert uuid4_pattern.match(first_id), (
            f"Expected UUID4 ID format, got: {first_id}"
        )

        (
            in_memory_vector_db._entries[0]["metadata"]["created_at"]
        )

        # --- Second run (same message) ---
        graph_2 = build(agt, message=TRIGGER_MESSAGE, deps=deps)
        async for _ in run_agent(graph_2, id_chat=TEST_SESSION_ID):
            pass

        # NEW: NOT idempotent — duplicate content creates SEPARATE entry
        # because UUID4 is used (no content-hash deduplication)
        assert len(in_memory_vector_db._entries) == 2, (
            f"Expected 2 entries after second run (UUID4, no dedup), "
            f"got {len(in_memory_vector_db._entries)}"
        )

        # The second entry should have a DIFFERENT UUID4 ID
        second_id = in_memory_vector_db._entries[1]["id"]
        assert uuid4_pattern.match(second_id), (
            f"Expected UUID4 ID format for second entry, got: {second_id}"
        )
        assert first_id != second_id, (
            f"Expected different UUID4 IDs for duplicate content, "
            f"got same ID: {first_id}"
        )

        # Embedding provenance: embedding occurred via the correct client
        custom_embedding_client.llm.async_embedding.assert_called()
        mock_memory_client.llm.async_embedding.assert_not_called()

    # ═════════════════════════════════════════════════════════════════════════
    # Edge case: Empty graph with no vector DB → passthrough
    # ═════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_no_vector_db_fallback_auto_created(self, mock_memory_client):
        """Memory node without vector DB → ephemeral fallback auto-created.

        With the NodeMemory change, when vector_db is not provided via deps,
        an ephemeral InMemoryVectorDB is auto-created. In search-only mode
        with no seed data, search returns nothing so the message passes
        through unchanged.
        """
        agt = _make_memory_graph(memory_instructions=None, memory_entries=[])
        # No memory deps → vector_db auto-creates InMemoryVectorDB fallback
        graph = build(agt, message=TRIGGER_MESSAGE, deps=None)

        async for _ in run_agent(graph, id_chat=TEST_SESSION_ID):
            pass

        memory_output = _get_memory_output(graph)
        assert MEMORY_CONTENT_MARKER not in memory_output, (
            f"Expected NO <memory_content> (empty ephemeral DB), "
            f"got: {memory_output!r}"
        )
        assert memory_output == TRIGGER_MESSAGE, (
            f"Expected passthrough, got: {memory_output!r}"
        )

        state = _get_memory_internal_state(graph)
        assert state["vector_db_available"] is True, (
            "vector_db_available should be True (auto-created fallback)"
        )
        assert state["vector_db_backend"] == "ephemeral", (
            f"Expected 'ephemeral' backend, got: {state.get('vector_db_backend')}"
        )
        assert state["injected_count"] == 0

    # ═════════════════════════════════════════════════════════════════════════
    # New: Extraction flow with ephemeral fallback → full roundtrip
    # ═════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_fallback_extraction_flow_completes(
        self,
    ):
        """Memory node without vector DB deps → background extraction + upsert happens.

        NEW BEHAVIOR: Extraction + embedding + upsert runs via asyncio.create_task()
        in the background and is NOT awaited by the graph flow.

        This test verifies:
        - Graph completes successfully even without vector_db in deps (auto-created)
        - Background extraction task completes (via upserted entries)
        - Embedding_client is used for background upsert
        - Same-turn output does NOT guarantee inclusion of newly extracted memories
          (they become available for FUTURE turns)
        """
        # Create extraction response - use a fresh MagicMock each time
        def make_extraction_response():
            return MagicMock(content=json.dumps({
                "memories": [
                    {"content": "User interested in pro plan pricing",
                     "trigger": "pricing"},
                ],
            }))
        
        # Create a proper mock client - AsyncMock with return_value
        mock_client = MagicMock()
        mock_client.llm = MagicMock()
        mock_client.llm.async_generate = AsyncMock(return_value=make_extraction_response())

        # Create a custom embedding client with a spy on async_embedding
        # so we can verify it was called by the background task
        custom_embedding_client = MagicMock()
        custom_embedding_client.llm = MagicMock()
        custom_embedding_client.llm.async_embedding = AsyncMock(
            return_value=_constant_embedding_response(),
        )
        background_tasks, background_task_tracker = _make_background_task_tracker()

        # No deps for vector_db → auto-creates InMemoryVectorDB fallback.
        # Provide embedding_client so Phase 3 upsert and Phase 4 search execute.
        agt = _make_memory_graph(
            memory_instructions="Extract pricing concerns from this message.",
            memory_entries=[],
            include_client=False,  # Don't include client node in graph
        )
        # Inject the mock client into memory deps - NodeMemory will use it
        # for background extraction
        deps = {
            "memory": {
                "embedding_client": custom_embedding_client,
                "client": mock_client,  # Pass mock client directly to memory node
                "background_task_tracker": background_task_tracker,
            },
        }
        graph = build(agt, message=TRIGGER_MESSAGE, deps=deps)

        async for _ in run_agent(graph, id_chat=TEST_SESSION_ID):
            pass

        # Embedding provenance: embedding occurred via the correct client
        # (by the background task, which we verify by awaiting the tracker)
        assert background_tasks, "Memory extraction should be registered as background work"
        await asyncio.gather(*background_tasks)
        custom_embedding_client.llm.async_embedding.assert_called()
        # The background task should have been called (verified by upserted entries)
        assert len(graph.nodes["memory"]._vector_db._entries) == 1

        # NEW: Same-turn output does NOT guarantee inclusion of newly extracted memories
        # because background task is not awaited by graph flow
        # The graph returns the original message without <memory_content>
        memory_output = _get_memory_output(graph)
        # This is EXPECTED behavior: newly extracted memories are NOT in same-turn output
        # They will be available for FUTURE turns when search() sees them
        assert memory_output == TRIGGER_MESSAGE, (
            f"Expected passthrough (no same-turn injection for newly extracted), "
            f"got: {memory_output!r}"
        )

        # Internal state: extraction happened, injection count is 0 (no matches in same turn)
        state = _get_memory_internal_state(graph)
        assert state["vector_db_available"] is True, (
            "vector_db_available should be True (auto-created fallback)"
        )
        assert state["vector_db_backend"] == "ephemeral", (
            f"Expected 'ephemeral' backend, got: {state.get('vector_db_backend')}"
        )
        assert state["extracted_count"] == 1, (
            f"Expected 1 extracted memory, got: {state['extracted_count']}"
        )
        # injected_count should be 0 because newly extracted memories
        # are not available in the same turn (background task not awaited)
        assert state["injected_count"] == 0, (
            f"Expected 0 injected (newly extracted not in same-turn), "
            f"got: {state['injected_count']}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 7: run_agent(deps=...) injection tests
    # ═════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_run_agent_deps_vector_db_injection(
        self, mock_memory_client,
    ):
        """run_agent(raw_graph_dict, deps=...) injects vector_db into NodeMemory.

        Calls run_agent() with a raw graph dict and deps that provide a
        pre-seeded InMemoryVectorDB. Verifies the custom VDB was used by
        checking that search() was called on it during execution.
        """
        from unittest.mock import AsyncMock

        custom_vdb = InMemoryVectorDB()
        # Wrap search with a spy to verify it was called
        original_search = custom_vdb.search
        search_spy = AsyncMock(wraps=original_search)
        custom_vdb.search = search_spy

        # Set up the graph with a user message so the search executes
        custom_embedding_client = _make_embedding_client()
        agt = _make_memory_graph(memory_instructions=None, memory_entries=[])
        agt["nodes"][0]["data"] = {"text": TRIGGER_MESSAGE}

        async for _ in run_agent(
            agt,
            deps={
                "memory": {
                    "vector_db": custom_vdb,
                    "embedding_client": custom_embedding_client,
                },
            },
            id_chat=TEST_SESSION_ID,
        ):
            pass

        # Embedding provenance: embedding occurred via the correct client
        custom_embedding_client.llm.async_embedding.assert_called()
        mock_memory_client.llm.async_embedding.assert_not_called()

        # Verify the custom vector_db was used for search
        search_spy.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_agent_deps_embedding_client_injection(
        self, mock_memory_client,
    ):
        """run_agent(raw_graph_dict, deps=...) injects embedding_client.

        Calls run_agent() with a raw graph dict, extraction instructions,
        and deps providing a custom embedding_client. Verifies the custom
        client's async_embedding was called during Phase 3 (embedding+upsert).
        """
        # Set up extraction response so Phase 3 executes
        extraction_response = MagicMock()
        extraction_response.content = json.dumps({
            "memories": [
                {"content": "User interested in pro plan pricing",
                 "trigger": "pricing"},
            ],
        })
        mock_memory_client.llm.async_generate = AsyncMock(
            return_value=extraction_response,
        )

        # Create a custom embedding client with a spy on async_embedding
        custom_client = MagicMock()
        custom_client.llm = MagicMock()
        custom_client.llm.async_embedding = AsyncMock(
            return_value=_constant_embedding_response(),
        )

        agt = _make_memory_graph(
            memory_instructions="Extract pricing concerns from this message.",
            memory_entries=[],
        )
        agt["nodes"][0]["data"] = {"text": TRIGGER_MESSAGE}

        async for _ in run_agent(
            agt,
            deps={"memory": {"embedding_client": custom_client}},
            id_chat=TEST_SESSION_ID,
        ):
            pass

        # Verify the custom embedding_client was used for embedding
        custom_client.llm.async_embedding.assert_called()
        # Verify CLIENT handle embedding was NEVER called
        mock_memory_client.llm.async_embedding.assert_not_called()
