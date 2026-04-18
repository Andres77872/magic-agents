"""
Shared pytest fixtures for magic-agents TDD tests.

This conftest.py provides reusable fixtures to avoid duplication
across the growing test suite.
"""
from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_agents.agt_flow import build, validate_graph, create_node
from magic_agents.execution.event_dispatcher import GraphEventDispatcher, NodeState
from magic_agents.execution.input_tracker import NodeInputTracker, InputInfo
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel


# ─── Helpers ────────────────────────────────────────────────────────────────

def _parse_dotenv(path: str | Path) -> dict[str, str]:
    """Parse a simple .env file into a dict.

    Handles:
    - KEY=VALUE lines
    - Lines with surrounding quotes: KEY="value with spaces"
    - Comment lines starting with #
    - Blank lines
    - Values containing = (splits on first = only)

    Args:
        path: Path to the .env file.

    Returns:
        Dict of key-value pairs. Empty dict if file doesn't exist.
    """
    result = {}
    env_path = Path(path)
    if not env_path.exists():
        return result

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip blank lines and comments
            if not line or line.startswith("#"):
                continue
            # Split on first = only
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes (single or double)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key:
                result[key] = value
    return result


def _is_placeholder_value(value: str) -> bool:
    """Check if a value looks like a placeholder rather than a real key.

    Placeholder patterns:
    - Contains 'your-', '-here', 'placeholder', 'xxx' (case-insensitive)
    - Empty string
    - Very short (< 10 chars)

    Real OpenAI keys are 50+ chars starting with 'sk-'.
    Real Serper keys are hex-like strings.
    """
    if not value:
        return True
    if len(value) < 10:
        return True
    lower = value.lower()
    placeholder_patterns = ["your-", "-here", "placeholder", "xxx", "changeme", "insert-"]
    return any(pattern in lower for pattern in placeholder_patterns)


def _load_env_test() -> dict[str, str]:
    """Load environment variables from .env.test file at project root.

    Returns:
        Dict of key-value pairs from .env.test. Empty dict if file missing.
    """
    # Project root is two levels up from test/conftest.py
    project_root = Path(__file__).parent.parent
    dotenv_path = project_root / ".env.test"
    return _parse_dotenv(dotenv_path)


def _populate_os_environ_from_dotenv() -> None:
    """Load .env.test values into os.environ for the pytest process.

    Shell environment variables take priority — only keys absent from
    os.environ are filled in from .env.test. This ensures runtime code
    that reads os.environ (e.g. env_resolver.py) sees the same keys
    that conftest skip-guards check.
    """
    dotenv_vars = _load_env_test()
    for key, value in dotenv_vars.items():
        if key not in os.environ:
            os.environ[key] = value


# Populate os.environ from .env.test at import time so all downstream
# code (env_resolver, LLM clients, etc.) sees the test keys.
_populate_os_environ_from_dotenv()


def _resolve_api_keys() -> dict[str, str]:
    """Resolve API keys with priority: real env vars > .env.test > empty.

    Returns:
        Dict with keys: openai_key, serper_key (if available).
    """
    dotenv_vars = _load_env_test()
    keys = {}

    # Priority 1: Real environment variables
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    serper_key = os.environ.get("SERPER_API_KEY", "")

    # Priority 2: Fall back to .env.test values
    if not openai_key:
        openai_key = dotenv_vars.get("OPENAI_API_KEY", "")
    if not serper_key:
        serper_key = dotenv_vars.get("SERPER_API_KEY", "")

    if openai_key:
        keys["openai_key"] = openai_key
    if serper_key:
        keys["serper_key"] = serper_key

    return keys


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


async def collect_all_from_generator(async_gen):
    """
    Consume an async generator and return all yielded items as a list.

    This is the correct way to collect results from run_agent() and
    similar async generators — avoids the asyncio.wait_for misuse
    that plagued test_edge_cases.py.
    """
    results = []
    async for item in async_gen:
        results.append(item)
    return results


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


def skip_if_no_api_keys(api_keys: dict[str, str] | None = None) -> dict[str, str]:
    """Skip the current test if no real API keys are available.

    Checks that openai_key exists and is not a placeholder value.

    Args:
        api_keys: Pre-resolved keys dict. If None, resolves fresh.

    Returns:
        The api_keys dict if available.

    Raises:
        pytest.skip: If no real API key is configured.
    """
    if api_keys is None:
        api_keys = _resolve_api_keys()

    openai_key = api_keys.get("openai_key", "")
    if not openai_key or _is_placeholder_value(openai_key):
        pytest.skip("No real OPENAI_API_KEY configured (env var or .env.test)")
    return api_keys


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

    mock_client = MagicMock()
    mock_client.llm = mock_llm_engine
    mock_client.model = "mock-model"

    with patch("magic_agents.node_system.NodeClientLLM.MagicLLM", return_value=mock_client):
        yield mock_client
