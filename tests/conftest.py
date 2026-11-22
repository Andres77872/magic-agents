"""Root pytest configuration for the magic-agents suite.

Shared fixtures only; helpers that tests import directly live in
``tests.support`` because pytest does not treat ``conftest.py`` as an
importable module.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magic_agents.vector_storage import InMemoryVectorDB


@pytest.fixture(autouse=True)
def _default_debug_kill_switch(monkeypatch):
    """Run every test with the executor's default ``DEBUG_ENABLED`` (on).

    A developer shell exporting ``DEBUG_ENABLED=false`` would otherwise silently
    change the debug events many tests assert on.
    """
    monkeypatch.delenv("DEBUG_ENABLED", raising=False)


@pytest.fixture
def mock_magic_llm():
    """Replace the provider client NodeClientLLM constructs during ``build()``.

    The patched ``MagicLLM`` returns a client whose engine streams three
    ``ChatCompletionModel`` chunks ("mocked LLM response") and whose
    ``async_generate`` returns a response with that content. Embedding through
    the client handle fails loudly: memory nodes must use their injected
    embedding client.
    """
    from magic_llm.model.ModelChatResponse import Choice, Message, ModelChatResponse
    from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

    engine = MagicMock()
    engine.model = "mock-model"
    engine.async_generate = AsyncMock(return_value=ModelChatResponse(
        id="mock-response", object="chat.completion", created=0.0, model="mock-model",
        choices=[Choice(index=0, message=Message(role="assistant", content="mocked LLM response"),
                        finish_reason="stop")],
    ))

    async def fake_stream(*args, **kwargs):
        for index, text in enumerate(["mocked ", "LLM ", "response"]):
            choice = ChoiceModel(index=0, delta=DeltaModel(content=text), logprobs=None, finish_reason=None)
            yield ChatCompletionModel(id=f"mock-chunk-{index}", model="mock-model", choices=[choice])

    engine.async_stream_generate = fake_stream
    engine.async_embedding = AsyncMock(
        side_effect=NotImplementedError("CLIENT handle should never be used for embedding.")
    )

    client = MagicMock()
    client.llm = engine
    client.model = "mock-model"
    with patch("magic_agents.node_system.NodeClientLLM.MagicLLM", return_value=client) as constructor:
        client.constructor = constructor
        yield client


@pytest.fixture
def in_memory_vector_db():
    """A fresh, empty ``InMemoryVectorDB``."""
    return InMemoryVectorDB()
