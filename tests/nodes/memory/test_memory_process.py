"""Behavior tests for NodeMemory.process()."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryEntry, MemoryNodeModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.NodeMemory import NodeMemory


class FakeLLM:
    def __init__(self, extraction_payload=None, embedding=None, fail_generate=False):
        self.extraction_payload = extraction_payload or {"memories": []}
        self.embedding = embedding or [0.1, 0.2, 0.3]
        self.fail_generate = fail_generate
        self.generated_chats = []
        self.embedding_inputs = []

    async def async_generate(self, chat, json_output=False):
        self.generated_chats.append((chat, json_output))
        if self.fail_generate:
            raise RuntimeError("LLM extraction failed")
        return SimpleNamespace(content=json.dumps(self.extraction_payload))

    async def async_embedding(self, text):
        self.embedding_inputs.append(text)
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.embedding)])


class FakeVectorDB:
    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self.search_calls = []
        self.upserts = []

    async def search(self, query_embedding, top_k, filter_scope):
        self.search_calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "filter_scope": filter_scope,
            }
        )
        return self.search_results

    async def upsert(self, **kwargs):
        self.upserts.append(kwargs)


def client_for(llm):
    return SimpleNamespace(llm=llm)


def memory_node(**kwargs):
    data = MemoryNodeModel(
        instructions=kwargs.pop("instructions", "Extract memory"),
        context_messages_count=kwargs.pop("context_messages_count", 0),
        top_k=kwargs.pop("top_k", 3),
    )
    return NodeMemory(data=data, node_id="mem-test", **kwargs)


async def run_node(node, message="Remember this", id_chat="session-1"):
    node.inputs[node.INPUT_HANDLE] = message
    outputs = [
        event
        async for event in node.process(ModelAgentRunLog(id_chat=id_chat))
    ]
    business_outputs = [event for event in outputs if event["type"] == node.OUTPUT_HANDLE]
    assert len(business_outputs) == 1
    return business_outputs[0]["content"]["content"], outputs


@pytest.mark.asyncio
async def test_process_injects_search_matches_and_uses_session_filter():
    older = MemoryEntry(
        content="Older memory",
        trigger="profile",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    newer = MemoryEntry(
        content="Newer memory",
        trigger="profile",
        created_at=datetime(2024, 1, 2, tzinfo=UTC),
    )
    vector_db = FakeVectorDB(search_results=[newer, older])
    embedding_llm = FakeLLM(embedding=[0.5, 0.6])
    node = memory_node(
        instructions=None,
        vector_db=vector_db,
        embedding_client=client_for(embedding_llm),
    )
    assert isinstance(node, NodeMemory)

    content, _ = await run_node(node, "Current message", id_chat="chat-123")

    assert content == (
        "<memory_content>Older memory</memory_content>\n"
        "<memory_content>Newer memory</memory_content>\n"
        "Current message"
    )
    assert vector_db.search_calls == [
        {
            "query_embedding": [0.5, 0.6],
            "top_k": 3,
            "filter_scope": {"session_id": "chat-123", "node_id": "mem-test"},
        }
    ]


@pytest.mark.asyncio
async def test_background_extraction_uses_history_window_and_upserts_memory():
    extraction_llm = FakeLLM(
        extraction_payload={
            "memories": [{"content": "User likes concise answers", "trigger": "preference"}],
        }
    )
    embedding_llm = FakeLLM(embedding=[0.9, 0.1])
    vector_db = FakeVectorDB()
    tracked_tasks = []

    def track_background(coro):
        tracked_tasks.append(coro)

    node = memory_node(
        context_messages_count=2,
        history_messages=[
            {"role": "user", "content": "Ignored old user"},
            {"role": "assistant", "content": "Recent assistant"},
            {"role": "user", "content": "Recent user"},
        ],
        vector_db=vector_db,
        embedding_client=client_for(embedding_llm),
        background_task_tracker=track_background,
    )
    node.inputs[node.CLIENT_HANDLE] = client_for(extraction_llm)

    content, _ = await run_node(node, "Current message", id_chat="chat-456")
    assert content == "Current message"
    assert len(tracked_tasks) == 1

    await tracked_tasks[0]

    extraction_chat, json_output = extraction_llm.generated_chats[0]
    messages = extraction_chat.messages
    assert json_output is True
    assert [message["role"] for message in messages] == [
        "system",
        "assistant",
        "user",
        "user",
    ]
    assert [message["content"] for message in messages[1:]] == [
        "Recent assistant",
        "Recent user",
        "Current message",
    ]
    assert embedding_llm.embedding_inputs == [
        "Current message",
        "User likes concise answers",
    ]
    assert len(vector_db.upserts) == 1
    assert vector_db.upserts[0]["embedding"] == [0.9, 0.1]
    assert vector_db.upserts[0]["metadata"]["session_id"] == "chat-456"
    assert vector_db.upserts[0]["metadata"]["node_id"] == "mem-test"


@pytest.mark.asyncio
async def test_background_extraction_failure_is_fail_open(caplog):
    extraction_llm = FakeLLM(fail_generate=True)
    embedding_llm = FakeLLM()
    vector_db = FakeVectorDB()
    tracked_tasks = []
    node = memory_node(
        vector_db=vector_db,
        embedding_client=client_for(embedding_llm),
        background_task_tracker=tracked_tasks.append,
    )
    node.inputs[node.CLIENT_HANDLE] = client_for(extraction_llm)

    content, _ = await run_node(node, "Do not block response")
    assert content == "Do not block response"

    await tracked_tasks[0]

    assert vector_db.upserts == []
    assert "Background extraction failed" in caplog.text


@pytest.mark.asyncio
async def test_empty_input_yields_empty_without_search_or_background():
    vector_db = FakeVectorDB()
    tracked_tasks = []
    node = memory_node(
        vector_db=vector_db,
        embedding_client=client_for(FakeLLM()),
        background_task_tracker=tracked_tasks.append,
    )

    content, _ = await run_node(node, "")

    assert content == ""
    assert tracked_tasks == []
    assert vector_db.search_calls == []


def test_capture_internal_state_reports_memory_configuration():
    vector_db = FakeVectorDB()
    node = memory_node(
        context_messages_count=4,
        top_k=7,
        history_messages=[{"role": "user", "content": "hello"}],
        vector_db=vector_db,
    )

    state = node._capture_internal_state()

    assert state["vector_db_available"] is True
    assert state["vector_db_backend"] == "external"
    assert state["top_k"] == 7
    assert state["_context_messages_count"] == 4
    assert state["_history_messages_count"] == 1


@pytest.mark.asyncio
async def test_naive_and_aware_timestamps_from_the_store_still_order_chronologically():
    """Regression: sorting mixed naive/aware ``created_at`` raised TypeError outside fail-open."""
    legacy = MemoryEntry(content="Legacy memory", trigger="t", created_at=datetime(2024, 1, 1))  # naive
    recent = MemoryEntry(content="Recent memory", trigger="t", created_at=datetime(2024, 6, 1, tzinfo=UTC))
    node = memory_node(
        instructions=None,
        vector_db=FakeVectorDB(search_results=[recent, legacy]),
        embedding_client=client_for(FakeLLM()),
    )

    content, _ = await run_node(node, "Current message")

    assert content == (
        "<memory_content>Legacy memory</memory_content>\n"
        "<memory_content>Recent memory</memory_content>\n"
        "Current message"
    )


@pytest.mark.asyncio
async def test_in_memory_store_backfills_missing_created_at_as_utc():
    from magic_agents.vector_storage import InMemoryVectorDB

    store = InMemoryVectorDB()
    # An entry persisted before created_at was recorded.
    store._entries.append({"embedding": [1.0, 0.0], "metadata": {
        "content": "no timestamp", "session_id": "s", "node_id": "n"}})

    [entry] = await store.search([1.0, 0.0], top_k=1, filter_scope={"session_id": "s", "node_id": "n"})

    assert entry.created_at.tzinfo is not None
