"""RED tests for no-CHAT NodeLLM tool replay history preservation.

These tests lock the custom-agent `/completions` replay contract from
api.magic_llm's SDD change `custom-agent-completions-tools-streaming`.
The no-CHAT NodeLLM path is used by inline/custom-agent graphs that do not
include an explicit CHAT node, so backend-injected OpenAI messages must survive
as provider-ready ModelChat history.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel
from magic_agents.node_system.NodeLLM import NodeLLM
from magic_llm.model.ModelChatResponse import Choice, Message, ModelChatResponse, UsageModel


def _build_mock_response(content: str = "continued answer", model: str = "test/model") -> ModelChatResponse:
    """Build a minimal non-streaming response for NodeLLM.process()."""
    return ModelChatResponse(
        id="resp-history-1",
        object="chat.completion",
        created=1234567890.0,
        model=model,
        choices=[Choice(index=0, message=Message(role="assistant", content=content))],
        usage=UsageModel(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )


def _build_capturing_client(response: ModelChatResponse):
    """Return a mock MagicLLM client and the ModelChat objects it receives."""
    captured_chats = []
    mock_client = MagicMock()
    mock_client.llm.model = response.model

    async def capture_generate(chat, **kwargs):
        captured_chats.append(chat)
        return response

    mock_client.llm.async_generate = AsyncMock(side_effect=capture_generate)
    return mock_client, captured_chats


def _schema_only_tool_call() -> dict:
    return {
        "index": 0,
        "id": "call_weather_1",
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "arguments": '{"city":"Buenos Aires"}',
        },
        "execution": "client",
        "source": "schema_only",
        "node_id": "llm-weather-router",
    }


@pytest.mark.asyncio
async def test_no_chat_history_preserves_assistant_tool_calls_with_node_id_metadata():
    """assistant.tool_calls replay metadata must reach the provider chat unchanged."""
    tool_call = _schema_only_tool_call()
    history = [
        {"role": "user", "content": "What is the weather in Buenos Aires?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call],
        },
    ]
    mock_client, captured_chats = _build_capturing_client(_build_mock_response())
    node = NodeLLM(
        node_id="llm-no-chat-history",
        node_type="llm",
        data=LlmNodeModel(stream=False, history_messages=history, max_messages=10),
        debug=False,
    )
    node.inputs = {
        "handle-client-provider": mock_client,
        "handle_user_message": "Continue after the tool call.",
    }

    async for _ in node.process(chat_log=None):
        pass

    assert len(captured_chats) == 1
    chat = captured_chats[0]
    assistant_messages = [message for message in chat.messages if message["role"] == "assistant"]
    assert assistant_messages, f"Expected replayed assistant message, got {chat.messages!r}"
    replayed_assistant = assistant_messages[0]

    assert replayed_assistant["tool_calls"] == [tool_call]
    assert replayed_assistant["tool_calls"][0]["id"] == "call_weather_1"
    assert replayed_assistant["tool_calls"][0]["index"] == 0
    assert replayed_assistant["tool_calls"][0]["node_id"] == "llm-weather-router"
    assert replayed_assistant["tool_calls"][0]["execution"] == "client"
    assert replayed_assistant["tool_calls"][0]["source"] == "schema_only"


@pytest.mark.asyncio
async def test_no_chat_history_preserves_tool_result_order_and_optional_name():
    """role=tool results must follow their assistant tool_calls in no-CHAT history."""
    tool_call = _schema_only_tool_call()
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_weather_1",
        "name": "lookup_weather",
        "content": '{"temperature":"18C","condition":"clear"}',
    }
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call],
        },
        tool_result,
        {"role": "user", "content": "Use the weather result in Spanish."},
    ]
    mock_client, captured_chats = _build_capturing_client(_build_mock_response())
    node = NodeLLM(
        node_id="llm-no-chat-tool-result",
        node_type="llm",
        data=LlmNodeModel(stream=False, history_messages=history, max_messages=10),
        debug=False,
    )
    node.inputs = {
        "handle-client-provider": mock_client,
        "handle_user_message": "Answer now.",
    }

    async for _ in node.process(chat_log=None):
        pass

    assert len(captured_chats) == 1
    chat = captured_chats[0]
    replayed_roles = [message["role"] for message in chat.messages]
    assert replayed_roles == ["assistant", "tool", "user", "user"]

    assistant_message = chat.messages[0]
    replayed_tool_result = chat.messages[1]
    assert assistant_message["tool_calls"][0]["id"] == "call_weather_1"
    assert assistant_message["tool_calls"][0]["node_id"] == "llm-weather-router"
    assert replayed_tool_result["tool_call_id"] == "call_weather_1"
    assert replayed_tool_result["name"] == "lookup_weather"
    assert replayed_tool_result["content"] == tool_result["content"]
