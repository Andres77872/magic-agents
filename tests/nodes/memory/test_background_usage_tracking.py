import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from magic_agents.hooks.hook_relay import HookRelay
from magic_agents.models.factory.Nodes import LlmNodeModel
from magic_agents.models.factory.Nodes.MemoryNodeModel import MemoryNodeModel
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.node_system.NodeMemory import NodeMemory


class _RecordingFlowHooks:
    def __init__(self) -> None:
        self.last_context = None

    def on_llm_end(self, context):
        self.last_context = context


class _UsageWithDetails:
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18
    cached_tokens_read = 3
    cached_tokens_write = 2
    reasoning_tokens = 5
    audio_tokens = 4
    accepted_prediction_tokens = 6
    rejected_prediction_tokens = 1
    provider_request_id = "usage-provider-req"
    service_tier = "default"
    provider_extra = {"safe_counter": 9}
    usage_source = "provider"
    prompt_tokens_details = None
    completion_tokens_details = None

    def model_dump(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens_read": self.cached_tokens_read,
            "cached_tokens_write": self.cached_tokens_write,
            "reasoning_tokens": self.reasoning_tokens,
            "audio_tokens": self.audio_tokens,
            "accepted_prediction_tokens": self.accepted_prediction_tokens,
            "rejected_prediction_tokens": self.rejected_prediction_tokens,
            "provider_request_id": self.provider_request_id,
            "service_tier": self.service_tier,
            "provider_extra": self.provider_extra,
            "usage_source": self.usage_source,
        }


@pytest.mark.asyncio
async def test_node_memory_background_task_tracker_waits_before_terminal_usage():
    tracked_tasks: list[asyncio.Task] = []

    def background_task_tracker(coro):
        task = asyncio.create_task(coro, name="tracked-memory-background")
        tracked_tasks.append(task)
        return task

    vector_db = MagicMock()
    vector_db.search = AsyncMock(return_value=[])
    vector_db.upsert = AsyncMock()

    embedding_response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])
    embedding_llm = MagicMock()
    embedding_llm.async_embedding = AsyncMock(return_value=embedding_response)
    embedding_client = MagicMock(llm=embedding_llm)

    extraction_response = SimpleNamespace(
        content=json.dumps({"memories": [{"content": "likes red dragons", "trigger": "preference"}]})
    )
    client_llm = MagicMock()
    client_llm.async_generate = AsyncMock(return_value=extraction_response)
    client = MagicMock(llm=client_llm)

    node = NodeMemory(
        data=MemoryNodeModel(instructions="Extract memories"),
        node_id="mem-usage",
        vector_db=vector_db,
        embedding_client=embedding_client,
        background_task_tracker=background_task_tracker,
    )
    node.inputs[node.INPUT_HANDLE] = "remember I like red dragons"
    node.inputs[node.CLIENT_HANDLE] = client

    results = []
    async for result in node.process(ModelAgentRunLog(id_chat="chat-1")):
        results.append(result)

    assert tracked_tasks, "NodeMemory must register background extraction with the provided tracker"
    await asyncio.gather(*tracked_tasks)
    assert any(result.get("content", {}).get("content") == "remember I like red dragons" for result in results)
    assert client_llm.async_generate.await_count == 1
    assert embedding_llm.async_embedding.await_count >= 2  # foreground search + background upsert
    assert vector_db.upsert.await_count == 1


def test_hook_relay_forwards_new_usage_detail_fields_and_usage_provider_request_id():
    hooks = _RecordingFlowHooks()
    relay = HookRelay(flow_hooks=hooks, node_id="llm-1")
    response = SimpleNamespace(
        id="response-id-fallback",
        model="gpt-4o-mini",
        content="hello",
        finish_reason="stop",
        usage=_UsageWithDetails(),
    )
    state = SimpleNamespace(step=0)

    relay.on_llm_response(response=response, state=state)

    outputs = hooks.last_context.outputs
    assert outputs["provider_request_id"] == "usage-provider-req"
    assert outputs["cached_tokens_read"] == 3
    assert outputs["cached_tokens_write"] == 2
    assert outputs["reasoning_tokens"] == 5
    assert outputs["audio_tokens"] == 4
    assert outputs["accepted_prediction_tokens"] == 6
    assert outputs["rejected_prediction_tokens"] == 1
    assert outputs["provider_extra"] == {"safe_counter": 9}
    assert outputs["raw_usage_json"]["provider_request_id"] == "usage-provider-req"


def test_node_llm_generation_event_relays_usage_detail_fields():
    node = NodeLLM(data=LlmNodeModel(stream=False), node_id="llm-usage")
    response = SimpleNamespace(
        id="response-id-fallback",
        model="gpt-4o-mini",
        usage=_UsageWithDetails(),
    )

    event = node._emit_llm_generation(response)
    payload = event["content"]

    assert payload["provider_request_id"] == "usage-provider-req"
    assert payload["cached_tokens_read"] == 3
    assert payload["cached_tokens_write"] == 2
    assert payload["reasoning_tokens"] == 5
    assert payload["audio_tokens"] == 4
    assert payload["accepted_prediction_tokens"] == 6
    assert payload["rejected_prediction_tokens"] == 1
    assert payload["provider_extra"] == {"safe_counter": 9}
