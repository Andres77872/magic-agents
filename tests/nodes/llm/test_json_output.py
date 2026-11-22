"""Real NodeLLM runtime paths with fakes only at the provider boundary."""

from __future__ import annotations

from typing import Any

import pytest

from magic_llm.agent.bundle import SubagentBundle
from magic_llm.model import ModelChat
from magic_llm.model.ModelChatResponse import Choice, Message, ModelChatResponse, UsageModel
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel, DeltaModel

from magic_agents.agt_flow import (
    disable_task_subagents,
    enable_task_subagents,
    is_task_subagents_enabled,
)
from magic_agents.hooks.flow_hooks import FlowHooks
from magic_agents.hooks.hook_registry import HookRegistry
from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel
from magic_agents.node_system.NodeLLM import NodeLLM


def response(content: str, *, response_id: str = "response-1") -> ModelChatResponse:
    return ModelChatResponse(
        id=response_id,
        object="chat.completion",
        created=1.0,
        model="local-model",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=UsageModel(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def chunk(content: str, *, response_id: str = "chunk-1") -> ChatCompletionModel:
    return ChatCompletionModel(
        id=response_id,
        model="local-model",
        choices=[ChoiceModel(delta=DeltaModel(content=content), finish_reason="stop")],
    )


class LocalProvider:
    """Deterministic provider engine implementing the real NodeLLM boundary."""

    model = "local-model"
    engine_name = "local"

    def __init__(self, responses: list[ModelChatResponse] | None = None, streams=None):
        self.responses = list(responses or [response("ok")])
        self.streams = list(streams or [])
        self.generated_chats: list[list[dict]] = []
        self.streamed_chats: list[list[dict]] = []

    async def async_generate(self, chat: ModelChat, **kwargs):
        self.generated_chats.append([dict(message) for message in chat.get_messages()])
        return self.responses.pop(0)

    async def async_stream_generate(self, chat: ModelChat, **kwargs):
        self.streamed_chats.append([dict(message) for message in chat.get_messages()])
        for item in self.streams.pop(0):
            yield item


class LocalClient:
    def __init__(self, provider: LocalProvider):
        self.llm = provider
        self.agent_calls: list[dict[str, Any]] = []
        self.agent_stream_calls: list[dict[str, Any]] = []
        self.agent_response = response("agent complete", response_id="agent-response")
        self.agent_chunks = [chunk("agent "), chunk("stream")]
        self.subagent_bundle = SubagentBundle()

    async def run_agent_async(self, **kwargs):
        self.agent_calls.append(kwargs)
        return self.agent_response

    async def run_agent_stream_async(self, **kwargs):
        self.agent_stream_calls.append(kwargs)
        for item in self.agent_chunks:
            yield item

    async def load_subagents(self, manifest_dir, code_registry):
        return self.subagent_bundle

    def reset_depths(self):
        return None


class EchoTool:
    @property
    def __name__(self) -> str:
        return "echo"

    @property
    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }

    @property
    def tool_callable(self):
        return self

    async def __call__(self, text: str) -> str:
        return text


class RecordingLLMHooks(FlowHooks):
    def __init__(self):
        self.events: list[tuple[str, Any]] = []

    async def on_llm_start(self, context, llm_config=None):
        self.events.append(("start", context))

    async def on_llm_end(self, context, response=None):
        self.events.append(("end", context))

    async def on_llm_loop_end(self, context):
        self.events.append(("loop_end", context))


def make_node(*, stream: bool = False, json_output: bool = False, iterate: bool = False):
    return NodeLLM(
        data=LlmNodeModel(stream=stream, json_output=json_output, iterate=iterate),
        node_id="llm-runtime",
        node_type="llm",
        debug=False,
    )


async def collect(node: NodeLLM) -> list[dict]:
    return [event async for event in node.process(None)]


def generated_values(node: NodeLLM, events: list[dict]) -> list[Any]:
    return [
        event["content"]["content"]
        for event in events
        if event.get("type") == node.OUTPUT_HANDLE_GENERATED
    ]


@pytest.mark.asyncio
async def test_callable_agent_loop_preserves_assembled_chat_history():
    provider = LocalProvider()
    client = LocalClient(provider)
    chat = ModelChat("original system")
    chat.add_user_message("old question")
    chat.add_assistant_message("old answer")
    node = make_node()
    node.inputs = {
        node.INPUT_HANDLER_CLIENT_PROVIDER: client,
        node.INPUT_HANDLER_CHAT: chat,
        node.INPUT_HANDLER_USER_MESSAGE: "new question",
        "handle-tool-echo": EchoTool(),
    }

    events = await collect(node)

    assert len(client.agent_calls) == 1
    call = client.agent_calls[0]
    assert call["user_input"] == "new question"
    assert call["extra_messages"] == [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]
    assert call["system_prompt"].endswith("original system")
    assert "## Available Tools" in call["system_prompt"]
    assert set(call["tool_functions"]) == {"echo"}
    assert generated_values(node, events) == ["agent complete"]


@pytest.mark.asyncio
async def test_callable_streaming_path_runs_real_node_branch():
    client = LocalClient(LocalProvider())
    node = make_node(stream=True)
    node.inputs = {
        node.INPUT_HANDLER_CLIENT_PROVIDER: client,
        node.INPUT_HANDLER_USER_MESSAGE: "stream with a tool",
        "handle-tool-echo": EchoTool(),
    }

    events = await collect(node)

    assert len(client.agent_stream_calls) == 1
    assert client.agent_stream_calls[0]["user_input"] == "stream with a tool"
    assert set(client.agent_stream_calls[0]["tool_functions"]) == {"echo"}
    assert generated_values(node, events) == ["agent stream"]


@pytest.mark.asyncio
async def test_subagent_only_configuration_uses_agent_loop():
    was_enabled = is_task_subagents_enabled()
    enable_task_subagents()
    try:
        provider = LocalProvider()
        client = LocalClient(provider)

        async def research(topic: str) -> str:
            return f"researched {topic}"

        client.subagent_bundle = SubagentBundle(
            tool_schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "research",
                        "description": "Research a topic",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_callables={"research": research},
            registered_count=1,
        )
        node = make_node()
        node.inputs = {
            node.INPUT_HANDLER_CLIENT_PROVIDER: client,
            node.INPUT_HANDLER_USER_MESSAGE: "delegate this",
        }

        await collect(node)

        assert len(client.agent_calls) == 1
        assert client.agent_calls[0]["tools"] == client.subagent_bundle.tool_schemas
        assert client.agent_calls[0]["tool_functions"] == {}
        assert provider.generated_chats == []
    finally:
        if not was_enabled:
            disable_task_subagents()


@pytest.mark.asyncio
async def test_streaming_iterate_resets_generated_content_each_run():
    provider = LocalProvider(streams=[[chunk("first")], [chunk("second")]])
    client = LocalClient(provider)
    node = make_node(stream=True, iterate=True)
    node.inputs = {
        node.INPUT_HANDLER_CLIENT_PROVIDER: client,
        node.INPUT_HANDLER_USER_MESSAGE: "repeat",
    }

    first = await collect(node)
    second = await collect(node)

    assert generated_values(node, first) == ["first"]
    assert generated_values(node, second) == ["second"]
    assert node.generated == "second"


@pytest.mark.parametrize(
    "provider_content",
    [
        '{"answer": 42}',
        'before\n```json\n{"answer": 42}\n```\nafter',
        '```\n{"answer": 42}\n```',
        'prefix {"answer": 42} suffix',
    ],
    ids=["plain", "fenced_json", "fenced_bare", "embedded"],
)
async def test_json_output_extracts_and_parses_the_json_object(provider_content):
    client = LocalClient(LocalProvider(responses=[response(provider_content)]))
    node = make_node(json_output=True)
    node.inputs = {
        node.INPUT_HANDLER_CLIENT_PROVIDER: client,
        node.INPUT_HANDLER_USER_MESSAGE: "return json",
    }

    events = await collect(node)

    assert generated_values(node, events) == [{"answer": 42}]
    assert node.generated == {"answer": 42}


@pytest.mark.parametrize(
    ("provider_content", "error_type"),
    [("not valid json", "JSONParseError"), ('{"answer": 42', "JSONParseError"), ("", "JSONExtractionError")],
)
async def test_unparseable_json_output_is_a_diagnostic_without_generated_result(provider_content, error_type):
    client = LocalClient(LocalProvider(responses=[response(provider_content)]))
    node = make_node(json_output=True)
    node.inputs = {
        node.INPUT_HANDLER_CLIENT_PROVIDER: client,
        node.INPUT_HANDLER_USER_MESSAGE: "return json",
    }

    events = await collect(node)

    error_types = [e["content"].get("error_type") for e in events
                   if e.get("type") == "debug" and isinstance(e.get("content"), dict) and e["content"].get("error_type")]
    assert error_types == [error_type]
    assert generated_values(node, events) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_real_node_llm_fires_single_call_lifecycle(stream):
    provider = LocalProvider(
        responses=[response("nonstream")],
        streams=[[chunk("stream")]],
    )
    client = LocalClient(provider)
    node = make_node(stream=stream)
    registry = HookRegistry()
    hooks = RecordingLLMHooks()
    registry.execution_id = "execution-llm"
    registry.run_id = "run-llm"
    registry.register_graph(hooks)
    node._hooks = registry
    node.inputs = {
        node.INPUT_HANDLER_CLIENT_PROVIDER: client,
        node.INPUT_HANDLER_USER_MESSAGE: "hello",
    }

    await collect(node)

    assert [name for name, _context in hooks.events] == ["start", "end", "loop_end"]
    assert all(context.execution_id == "execution-llm" for _name, context in hooks.events)
    assert hooks.events[1][1].outputs["content"] in {"nonstream", "stream"}
    assert hooks.events[2][1].outputs["total_iterations"] == 1
