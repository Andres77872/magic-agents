"""
Shared pytest fixtures for magic-agents TDD tests.

This conftest.py provides reusable fixtures to avoid duplication
across the growing test suite.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_agents.agt_flow import build, validate_graph, create_node
from magic_agents.execution.event_dispatcher import GraphEventDispatcher, NodeState
from magic_agents.execution.input_tracker import NodeInputTracker, InputInfo
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from magic_agents.vector_storage import InMemoryVectorDB
from magic_llm.model.ModelEmbeddingResponse import (
    EmbeddingData,
    ModelEmbeddingResponse,
)
from test_support import (
    _populate_os_environ_from_dotenv,
    _resolve_api_keys,
)


# Populate os.environ from .env.test at import time so all downstream
# code (env_resolver, LLM clients, etc.) sees the test keys.
_populate_os_environ_from_dotenv()


# ─── Helpers ────────────────────────────────────────────────────────────────


def make_node(node_id: str, node_type: str, data: dict = None) -> dict:
    """Create a minimal node dict for JSON graph definitions."""
    node = {"id": node_id, "type": node_type}
    if data is not None:
        node["data"] = data
    return node


def make_edge(
    edge_id: str,
    source: str,
    target: str,
    source_handle: str = None,
    target_handle: str = None,
) -> dict:
    """Create a minimal edge dict."""
    edge = {"id": edge_id, "source": source, "target": target}
    if source_handle is not None:
        edge["sourceHandle"] = source_handle
    if target_handle is not None:
        edge["targetHandle"] = target_handle
    return edge


def make_minimal_graph(
    extra_nodes: list = None,
    extra_edges: list = None,
    debug: bool = False,
) -> dict:
    """
    Create a minimal valid graph with one USER_INPUT and one END node.

    Args:
        extra_nodes: Additional nodes to include.
        extra_edges: Additional edges to include.
        debug: Whether to enable debug mode.

    Returns:
        A dict suitable for passing to build().
    """
    nodes = [
        make_node("user_input", ModelAgentFlowTypesModel.USER_INPUT),
        make_node("end_node", ModelAgentFlowTypesModel.END),
    ]
    edges = [
        make_edge("e1", "user_input", "end_node"),
    ]
    if extra_nodes:
        nodes.extend(extra_nodes)
    if extra_edges:
        edges.extend(extra_edges)
    return {"type": "graph", "debug": debug, "nodes": nodes, "edges": edges}


# ─── Debug Summary Helpers (consolidated from test_loop_execution.py,
#     test_conditional_enhanced.py, test_loop_refactor.py) ────────────────────

def extract_streamed_content(item):
    """Extract streamed content from send_message or LLM output."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "content":
        return ""
    content = item.get("content")
    if content is None:
        return ""
    if hasattr(content, "choices") and content.choices:
        delta = content.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            return delta.content
    return ""


def get_executed_nodes(debug_summary: dict) -> set:
    """Extract set of executed node IDs from debug summary."""
    executed = set()
    if not debug_summary:
        return executed
    for node in debug_summary.get("nodes", []):
        if node.get("was_executed"):
            executed.add(node.get("node_id"))
    return executed


def get_bypassed_nodes(debug_summary: dict) -> set:
    """Extract set of bypassed node IDs from debug summary."""
    bypassed = set()
    if not debug_summary:
        return bypassed
    for node in debug_summary.get("nodes", []):
        if node.get("was_bypassed"):
            bypassed.add(node.get("node_id"))
    return bypassed


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def load_chat_stub():
    """A stub load_chat callable that does nothing."""
    return lambda **kwargs: None


@pytest.fixture
def minimal_graph():
    """A minimal valid graph dict (deepcopied to prevent mutation)."""
    return deepcopy(make_minimal_graph())


@pytest.fixture
def make_tracker():
    """Factory fixture for creating NodeInputTracker instances."""
    def _make(node_id: str = "test_node", expected_inputs: list = None):
        if expected_inputs is None:
            expected_inputs = []
        return NodeInputTracker(node_id=node_id, expected_inputs=expected_inputs)
    return _make


@pytest.fixture
def make_edge_model():
    """Factory fixture for creating EdgeNodeModel instances."""
    def _make(
        edge_id: str = "e1",
        source: str = "src",
        target: str = "tgt",
        source_handle: str = None,
        target_handle: str = None,
    ):
        return EdgeNodeModel(
            id=edge_id,
            source=source,
            target=target,
            sourceHandle=source_handle,
            targetHandle=target_handle,
        )
    return _make


@pytest.fixture
def make_input_info():
    """Factory fixture for creating InputInfo instances."""
    def _make(
        handle: str = "input_1",
        source_node: str = "src",
        source_handle: str = "output_1",
        content: Any = None,
    ):
        return InputInfo(
            handle=handle,
            source_node=source_node,
            source_handle=source_handle,
            content=content,
        )
    return _make


@pytest.fixture
def mock_llm_response():
    """
    Create a mock ChatCompletionModel-like object.

    This produces an object that looks like a real ChatCompletionModel
    to pass through the executor without needing real API calls.
    """
    mock_choice = MagicMock()
    mock_choice.delta = MagicMock()
    mock_choice.delta.content = "mocked LLM response"

    mock_response = MagicMock()
    mock_response.id = "mock-id"
    mock_response.model = "mock-model"
    mock_response.choices = [mock_choice]
    return mock_response


# ─── API Key Fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def api_keys() -> dict[str, str]:
    """Load API keys once per session with priority: real env > .env.test.

    Returns:
        Dict with available keys (openai_key, serper_key).
        May be empty if no keys are configured.
    """
    return _resolve_api_keys()


# ─── Mock MagicLLM Fixture ──────────────────────────────────────────────────

@pytest.fixture
def mock_magic_llm():
    """Patch MagicLLM so graph execution (run_agent) works without real API keys.

    Patches at ``magic_agents.node_system.NodeClientLLM.MagicLLM`` — the import
    site where NodeClientLLM creates the client during ``build()``.

    The mock client provides:
    - ``client.llm.async_generate()`` — returns a mock ModelChatResponse
    - ``client.llm.async_stream_generate()`` — yields valid ChatCompletionModel chunks

    Usage:
        @pytest.mark.asyncio
        async def test_something(self, mock_magic_llm, image_json_config):
            graph = build(image_json_config, "Hello", load_chat=None)
            async for result in run_agent(graph):
                events.append(result)
            # ... assertions ...
    """
    from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

    mock_llm_engine = MagicMock()

    # async_generate returns a ModelChatResponse-like object
    mock_response = MagicMock()
    mock_response.content = "mocked LLM response"
    mock_llm_engine.async_generate = AsyncMock(return_value=mock_response)

    # async_stream_generate yields valid ChatCompletionModel chunks
    # NOTE: usage is omitted — ChatCompletionModel has a sensible default
    async def fake_stream(*args, **kwargs):
        chunks = ["mocked ", "LLM ", "response"]
        for i, text in enumerate(chunks):
            delta = DeltaModel(content=text)
            choice = ChoiceModel(index=0, delta=delta, logprobs=None, finish_reason=None)
            yield ChatCompletionModel(
                id=f"mock-chunk-{i}",
                model="mock-model",
                choices=[choice],
            )

    mock_llm_engine.async_stream_generate = fake_stream

    # Embedding MUST NOT be called via the CLIENT handle — only via _embedding_client.
    # Any accidental call fails LOUDLY with NotImplementedError.
    mock_llm_engine.async_embedding = AsyncMock(
        side_effect=NotImplementedError(
            "CLIENT handle should never be used for embedding. Use _embedding_client."
        )
    )

    mock_client = MagicMock()
    mock_client.llm = mock_llm_engine
    mock_client.model = "mock-model"

    with patch("magic_agents.node_system.NodeClientLLM.MagicLLM", return_value=mock_client):
        yield mock_client


# ─── NodeMemory / Vector Embedding Fixtures ─────────────────────────────────

@pytest.fixture
def mock_magic_embedding():
    """Return a deterministic async embedding function for NodeMemory tests.

    Produces an 8-dimensional unit vector from input text using SHA256.
    The first 8 bytes of the hash digest are mapped to the range [-1, 1]
    and normalized to unit length. This guarantees:
    - Deterministic output: same text → same embedding vector
    - Reproducible tests: no randomness between runs
    - Stable similarity: identical texts produce identical vectors

    Returns:
        Async callable ``_embed(text: str) -> ModelEmbeddingResponse``.
        The response contains a single ``EmbeddingData`` entry at index 0.
    """
    async def _embed(text: str) -> ModelEmbeddingResponse:
        digest = hashlib.sha256(text.encode()).digest()
        # Map first 8 bytes from [0, 255] to [-1, 1]
        vector: list[float] = []
        for i in range(8):
            val = (digest[i] / 127.5) - 1.0
            vector.append(val)
        # Normalize to unit vector
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        return ModelEmbeddingResponse(
            object="list",
            data=[EmbeddingData(object="embedding", index=0, embedding=vector)],
            model="mock-embedding-model",
        )
    return _embed


@pytest.fixture
def in_memory_vector_db():
    """Return a fresh InMemoryVectorDB instance for NodeMemory tests.

    Each test gets an empty, isolated vector store — no state leakage
    between tests. Use together with ``mock_magic_embedding`` for
    ``rebuild()`` or insert seed entries directly via ``upsert()``.

    Returns:
        InMemoryVectorDB instance with empty internal state.
    """
    return InMemoryVectorDB()
