"""
Prove non-streaming NodeLLM yields ChatCompletionModel with full delta content.

Regression test for: Playground assistant messages persisted with empty `text`.

Root cause: non-streaming path at NodeLLM.py:583-590 yielded
ChatCompletionModel(choices=[ChoiceModel()]) — ChoiceModel() defaults to
DeltaModel(content='') which is falsy. The persistence wrapper in agt_flow.py
accumulates deltas via `if delta:` (line 520), so empty string content
was skipped → content_text='' in DB → retrieval returned {"text": ""}.

Fix: include intention.content in the non-streaming ChoiceModel delta.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from magic_llm.model.ModelChat import ModelChat
from magic_llm.model.ModelChatStream import ChatCompletionModel
from magic_llm.model.ModelChatResponse import (
    ModelChatResponse,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    UsageModel,
)

from magic_agents.node_system.NodeLLM import NodeLLM
from magic_agents.models.factory.Nodes.LlmNodeModel import LlmNodeModel


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_mock_response(content: str, model: str = "test/model"):
    """Build a ModelChatResponse with given content string."""
    return ModelChatResponse(
        id="resp-1",
        object="chat.completion",
        created=1234567890.0,
        model=model,
        choices=[Choice(
            index=0,
            message=Message(role="assistant", content=content),
        )],
        usage=UsageModel(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _build_tool_call_response(model: str = "test/model"):
    """Build a valid tool-call-only response with nullable content."""
    return ModelChatResponse(
        id="resp-tool-call-1",
        object="chat.completion",
        created=1234567890.0,
        model=model,
        choices=[Choice(
            index=0,
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(
                    id="call-create-deck-1",
                    type="function",
                    function=FunctionCall(
                        name="create_deck",
                        arguments='{"title":"Tool deck","slides":[]}',
                    ),
                )],
            ),
            finish_reason="tool_calls",
        )],
        usage=UsageModel(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _build_mock_client(response: ModelChatResponse):
    """Build a mock MagicLLM instance whose llm.async_generate returns `response`."""
    mock_client = MagicMock()
    mock_client.llm.model = response.model
    mock_client.llm.async_generate = AsyncMock(return_value=response)
    return mock_client


def _extract_chat_completion(event: dict) -> ChatCompletionModel:
    """
    Extract the ChatCompletionModel from an event yielded by NodeLLM.process().

    yield_static wraps content in a dict: {'type': handle, 'content': ...}.
    The prep() method wraps non-dict content in {'content': ..., 'node': 'NodeLLM'}.
    For ChatCompletionModel instances (which are dict-like via pydantic), prep()
    returns them directly wrapped in the inner wrapper: event['content'] is a dict
    with 'content' key containing the ChatCompletionModel.
    """
    payload = event.get("content", {})
    if isinstance(payload, ChatCompletionModel):
        return payload
    if isinstance(payload, dict):
        inner = payload.get("content")
        if isinstance(inner, ChatCompletionModel):
            return inner
    raise AssertionError(
        f"Cannot extract ChatCompletionModel from event: "
        f"payload type={type(payload).__name__}, keys={getattr(payload, 'keys', None) and list(payload.keys())}"
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestNonStreamingContentDelta:
    """Prove non-streaming OUTPUT_HANDLE_CONTENT has full delta content."""

    @pytest.mark.asyncio
    async def test_nonstreaming_content_delta_is_populated(self):
        """Non-streaming LLM yield must carry intention.content in choices[0].delta.content."""
        expected_content = "Hello from a non-streaming LLM call!"
        response = _build_mock_response(expected_content)
        mock_client = _build_mock_client(response)

        node_data = LlmNodeModel(stream=False)
        node = NodeLLM(
            node_id="llm-1",
            node_type="llm",
            data=node_data,
            debug=False,
        )

        node.inputs = {
            "handle-client-provider": mock_client,
            "handle_user_message": "Say hello",
        }

        outputs: list[dict] = []
        async for event in node.process(chat_log=None):
            outputs.append(event)

        content_events = [
            e for e in outputs
            if e.get("type") == node.OUTPUT_HANDLE_CONTENT
        ]

        assert len(content_events) >= 1, (
            f"Expected at least one content event on handle "
            f"'{node.OUTPUT_HANDLE_CONTENT}', got {[e['type'] for e in outputs]}"
        )

        ccm = _extract_chat_completion(content_events[0])
        assert len(ccm.choices) >= 1, "Expected at least one choice"
        choice = ccm.choices[0]
        assert choice.delta is not None, "Choice delta must not be None"

        actual_content = choice.delta.content
        assert actual_content == expected_content, (
            f"Delta content mismatch: expected {expected_content!r}, got {actual_content!r}"
        )

    @pytest.mark.asyncio
    async def test_nonstreaming_content_delta_not_empty_string(self):
        """Non-streaming delta content must be a non-empty, truthy string (not '')."""
        response = _build_mock_response("Real generated content")
        mock_client = _build_mock_client(response)

        node_data = LlmNodeModel(stream=False)
        node = NodeLLM(
            node_id="llm-2",
            node_type="llm",
            data=node_data,
            debug=False,
        )
        node.inputs = {
            "handle-client-provider": mock_client,
            "handle_user_message": "Hi",
        }

        async for event in node.process(chat_log=None):
            if event.get("type") == node.OUTPUT_HANDLE_CONTENT:
                ccm = _extract_chat_completion(event)
                delta_content = ccm.choices[0].delta.content
                assert delta_content, (
                    f"delta.content must be truthy (non-empty string), "
                    f"got {delta_content!r}"
                )
                assert isinstance(delta_content, str), (
                    f"delta.content must be str, got {type(delta_content)}"
                )
                return

        pytest.fail("No content event yielded")

    @pytest.mark.asyncio
    async def test_nonstreaming_empty_response_still_yields(self):
        """Empty LLM response still yields a content event (delta.content='')."""
        empty_response = _build_mock_response("")
        mock_client = _build_mock_client(empty_response)

        node_data = LlmNodeModel(stream=False)
        node = NodeLLM(
            node_id="llm-3",
            node_type="llm",
            data=node_data,
            debug=False,
        )
        node.inputs = {
            "handle-client-provider": mock_client,
            "handle_user_message": "Hi",
        }

        content_yielded = False
        async for event in node.process(chat_log=None):
            if event.get("type") == node.OUTPUT_HANDLE_CONTENT:
                content_yielded = True
                ccm = _extract_chat_completion(event)
                delta_content = ccm.choices[0].delta.content
                assert delta_content == "", (
                    f"Empty response should yield '' delta, got {delta_content!r}"
                )

        assert content_yielded, (
            "Empty responses should still yield a content event"
        )

    @pytest.mark.asyncio
    async def test_delta_content_matches_self_generated(self):
        """Delta content must match self.generated, which is set from intention.content."""
        expected = "Verification that self.generated and delta match"
        response = _build_mock_response(expected)
        mock_client = _build_mock_client(response)

        node_data = LlmNodeModel(stream=False)
        node = NodeLLM(
            node_id="llm-v",
            node_type="llm",
            data=node_data,
            debug=False,
        )
        node.inputs = {
            "handle-client-provider": mock_client,
            "handle_user_message": "Verify",
        }

        last_content = None
        async for event in node.process(chat_log=None):
            if event.get("type") == node.OUTPUT_HANDLE_CONTENT:
                ccm = _extract_chat_completion(event)
                last_content = ccm.choices[0].delta.content

        assert node.generated == expected, (
            f"node.generated should be {expected!r}, got {node.generated!r}"
        )
        assert last_content == expected, (
            f"delta.content should match node.generated: "
            f"{expected!r} != {last_content!r}"
        )

    @pytest.mark.asyncio
    async def test_schema_only_tool_call_response_normalizes_null_content(self):
        """Tool-call-only responses may have content=None and must not crash."""
        response = _build_tool_call_response()
        mock_client = _build_mock_client(response)

        node_data = LlmNodeModel(stream=False)
        node = NodeLLM(
            node_id="llm-tool-call-only",
            node_type="llm",
            data=node_data,
            debug=False,
        )
        node.inputs = {
            "handle-client-provider": mock_client,
            "handle_user_message": "Create a deck",
            "handle-tool-definition-0": {
                "type": "function",
                "function": {
                    "name": "create_deck",
                    "description": "Create a reveal presentation deck",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "slides": {"type": "array"},
                        },
                        "required": ["title", "slides"],
                    },
                },
            },
        }

        outputs: list[dict] = []
        async for event in node.process(chat_log=None):
            outputs.append(event)

        tool_events = [
            event for event in outputs
            if event.get("type") == node.OUTPUT_HANDLE_TOOL_CALLS
        ]
        assert len(tool_events) == 1
        envelope = tool_events[0]["content"]["content"]
        assert envelope == {
            "execution": "client",
            "source": "schema_only",
            "node_id": "llm-tool-call-only",
            "tool_calls": [{
                "id": "call-create-deck-1",
                "type": "function",
                "function": {
                    "name": "create_deck",
                    "arguments": '{"title":"Tool deck","slides":[]}',
                },
            }],
        }

        generated_events = [
            event for event in outputs
            if event.get("type") == node.OUTPUT_HANDLE_GENERATED
        ]
        assert len(generated_events) == 1
        assert generated_events[0]["content"]["content"] == ""
        assert node.generated == ""

        state = node._capture_internal_state()
        assert state["generated"] == ""

    def test_schema_only_tool_call_envelope_preserves_contract_with_node_id(self):
        """The runtime envelope adds node_id without rewriting tool call IDs/indexes."""
        node = NodeLLM(
            node_id="llm-router-node",
            node_type="llm",
            data=LlmNodeModel(stream=False),
            debug=False,
        )
        tool_calls = [{
            "index": 7,
            "id": "call-route-7",
            "type": "function",
            "function": {
                "name": "route_request",
                "arguments": '{"destination":"billing"}',
            },
        }]

        envelope = node._schema_only_tool_call_envelope(tool_calls)

        assert envelope["execution"] == "client"
        assert envelope["source"] == "schema_only"
        assert envelope["node_id"] == "llm-router-node"
        assert envelope["tool_calls"] == tool_calls
        assert envelope["tool_calls"][0]["id"] == "call-route-7"
        assert envelope["tool_calls"][0]["index"] == 7

    def test_capture_internal_state_normalizes_none_generated(self):
        """State capture must be resilient to legacy None generated values."""
        node = NodeLLM(
            node_id="llm-state-none",
            node_type="llm",
            data=LlmNodeModel(stream=False),
            debug=False,
        )
        node.generated = None

        state = node._capture_internal_state()

        assert state["generated"] == ""


# ── No-CHAT Windowing Integration Tests (Task 6.3) ────────────────────────────

class TestNoChatWindowing:
    """Prove NodeLLM no-CHAT fallback applies STM windowing correctly (spec X3)."""

    def _make_capturing_client(self, mock_response):
        """Build a mock client whose async_generate captures the chat argument."""
        mock_client = _build_mock_client(mock_response)
        captured_chat = []

        async def capturing_generate(chat, **kw):
            captured_chat.append(chat)
            return mock_response

        mock_client.llm.async_generate = AsyncMock(side_effect=capturing_generate)
        return mock_client, captured_chat

    # ── Test 1: DEFAULT_MAX_MESSAGES=30 applied with no config ─────────────────

    @pytest.mark.asyncio
    async def test_default_max_messages_30(self):
        """No-CHAT path applies DEFAULT_MAX_MESSAGES=30 when no explicit max_messages (L2)."""
        # 50 history msgs → windowing keeps at most 30 + 1 current turn = 31
        history = [
            {"role": "user", "content": f"history-msg-{i}"} for i in range(50)
        ]

        mock_response = _build_mock_response("Hello from LLM")
        mock_client, captured_chat = self._make_capturing_client(mock_response)

        node_data = LlmNodeModel(
            stream=False,
            history_messages=history,
        )
        node = NodeLLM(
            node_id="llm-nc-w1",
            node_type="llm",
            data=node_data,
            debug=False,
        )
        node.inputs = {
            "handle-client-provider": mock_client,
            "handle_user_message": "Current turn prompt",
        }

        async for _ in node.process(chat_log=None):
            pass

        assert len(captured_chat) == 1, "async_generate should be called once"
        chat = captured_chat[0]

        user_msgs = [m for m in chat.messages if m["role"] == "user"]
        # 30 history (default) + 1 current turn = 31
        assert len(user_msgs) == 31, (
            f"Expected 31 user messages (30 windowed + 1 turn), got {len(user_msgs)}. "
            f"All roles: {[m['role'] for m in chat.messages]}"
        )
        # Last message must be the current turn
        assert user_msgs[-1]["content"] == "Current turn prompt", (
            f"Last user msg should be current turn, got {user_msgs[-1]['content']!r}"
        )
        # First history message (msg-0) should NOT appear (dropped by windowing)
        first_history = next(
            (m for m in chat.messages if "history-msg-0" in str(m.get("content", ""))),
            None,
        )
        assert first_history is None, (
            "Oldest history message (msg-0) should have been dropped by windowing"
        )

    # ── Test 2: Explicit max_messages overrides default ────────────────────────

    @pytest.mark.asyncio
    async def test_explicit_max_messages(self):
        """No-CHAT path respects explicit max_messages=5 on LlmNodeModel."""
        history = [
            {"role": "user", "content": f"history-msg-{i}"} for i in range(20)
        ]

        mock_response = _build_mock_response("Hello")
        mock_client, captured_chat = self._make_capturing_client(mock_response)

        node_data = LlmNodeModel(
            stream=False,
            history_messages=history,
            max_messages=5,
        )
        node = NodeLLM(
            node_id="llm-nc-w2",
            node_type="llm",
            data=node_data,
            debug=False,
        )
        node.inputs = {
            "handle-client-provider": mock_client,
            "handle_user_message": "Turn",
        }

        async for _ in node.process(chat_log=None):
            pass

        chat = captured_chat[0]
        user_msgs = [m for m in chat.messages if m["role"] == "user"]
        assert len(user_msgs) == 6, (
            f"Expected 6 user messages (5 windowed + 1 turn), got {len(user_msgs)}"
        )
        # Verify the kept messages are the newest ones
        kept_contents = [m["content"] for m in user_msgs[:-1]]  # exclude current turn
        expected_newest = [f"history-msg-{i}" for i in range(15, 20)]
        assert kept_contents == expected_newest, (
            f"Expected newest 5 messages, got {kept_contents}"
        )

    # ── Test 3: System messages preserved ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_system_preservation_no_chat(self):
        """System messages preserved during no-CHAT windowing (W2)."""
        history = [
            {"role": "system", "content": "system-rule-1"},
            {"role": "system", "content": "system-rule-2"},
            {"role": "system", "content": "system-rule-3"},
        ] + [
            {"role": "user", "content": f"user-msg-{i}"} for i in range(50)
        ]

        mock_response = _build_mock_response("OK")
        mock_client, captured_chat = self._make_capturing_client(mock_response)

        node_data = LlmNodeModel(
            stream=False,
            history_messages=history,
            max_messages=10,
        )
        node = NodeLLM(
            node_id="llm-nc-w3",
            node_type="llm",
            data=node_data,
            debug=False,
        )
        node.inputs = {
            "handle-client-provider": mock_client,
            "handle_user_message": "Hi",
        }

        async for _ in node.process(chat_log=None):
            pass

        chat = captured_chat[0]
        system_msgs = [m for m in chat.messages if m["role"] == "system"]
        user_msgs = [m for m in chat.messages if m["role"] == "user"]

        # All 3 system messages must be present
        assert len(system_msgs) == 3, (
            f"Expected 3 system messages preserved, got {len(system_msgs)}"
        )
        assert system_msgs[0]["content"] == "system-rule-1"
        assert system_msgs[1]["content"] == "system-rule-2"
        assert system_msgs[2]["content"] == "system-rule-3"

        # 10 user msgs from history + 1 current turn = 11
        assert len(user_msgs) == 11, (
            f"Expected 11 user msgs (10 windowed + 1 turn), got {len(user_msgs)}"
        )

    # ── Test 4: Last user message preserved ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_last_user_preservation_no_chat(self):
        """Last user message in history is preserved during no-CHAT windowing (W3)."""
        # 20 user msgs — the last one (msg-19) must survive with max_messages=5
        history = [
            {"role": "user", "content": f"pre-turn-msg-{i}"} for i in range(20)
        ]

        mock_response = _build_mock_response("OK")
        mock_client, captured_chat = self._make_capturing_client(mock_response)

        node_data = LlmNodeModel(
            stream=False,
            history_messages=history,
            max_messages=5,
        )
        node = NodeLLM(
            node_id="llm-nc-w4",
            node_type="llm",
            data=node_data,
            debug=False,
        )
        node.inputs = {
            "handle-client-provider": mock_client,
            "handle_user_message": "Final turn",
        }

        async for _ in node.process(chat_log=None):
            pass

        chat = captured_chat[0]
        user_msgs = [m for m in chat.messages if m["role"] == "user"]

        # Total: 5 windowed + 1 current = 6 user msgs
        assert len(user_msgs) == 6, (
            f"Expected 6 user msgs (5 windowed + 1 turn), got {len(user_msgs)}"
        )
        # The last user message from history (pre-turn-msg-19) must be present
        history_in_chat = [m["content"] for m in user_msgs[:-1]]
        assert "pre-turn-msg-19" in history_in_chat, (
            f"Last history message 'pre-turn-msg-19' should be preserved, "
            f"got history contents: {history_in_chat}"
        )
        # The very last user message is the current turn
        assert user_msgs[-1]["content"] == "Final turn", (
            f"Last user msg should be current turn, got {user_msgs[-1]['content']!r}"
        )

    # ── Test 5: No windowing in CHAT path ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_no_windowing_in_chat_path(self):
        """CHAT input → no LLM-side windowing; history_messages NOT injected (M1)."""
        # 50 history msgs + max_messages=5 should NOT matter when CHAT input provided
        history = [
            {"role": "user", "content": f"history-msg-{i}"} for i in range(50)
        ]

        mock_response = _build_mock_response("Hello from LLM")
        mock_client, captured_chat = self._make_capturing_client(mock_response)

        node_data = LlmNodeModel(
            stream=False,
            history_messages=history,
            max_messages=5,
        )
        node = NodeLLM(
            node_id="llm-nc-w5",
            node_type="llm",
            data=node_data,
            debug=False,
        )

        # Provide a pre-built ModelChat as the CHAT input
        user_chat = ModelChat()
        user_chat.add_user_message("A single user msg from chat path")

        node.inputs = {
            "handle-client-provider": mock_client,
            "handle-chat": user_chat,
            "handle_user_message": "Should not add this (chat path uses chat directly)",
        }

        async for _ in node.process(chat_log=None):
            pass

        chat = captured_chat[0]
        # The Chat input is shared with other consumers, so NodeLLM sends a copy
        # and never mutates the caller's ModelChat.
        assert chat is not user_chat
        assert [m["content"] for m in user_chat.messages] == ["A single user msg from chat path"]
        # The copy gets the handle_user_message turn; no history is injected.
        assert len(chat.messages) == 2, (
            f"Chat path should have 2 user msgs (initial + added handle), "
            f"no history injected, got {len(chat.messages)}: {chat.messages}"
        )
        assert chat.messages[0]["role"] == "user"
        assert chat.messages[0]["content"] == "A single user msg from chat path"
        assert chat.messages[1]["role"] == "user"
        assert "chat path uses chat" in chat.messages[1]["content"]
        # Crucially: NO history messages were injected
        assert all("history-msg" not in str(m.get("content", "")) for m in chat.messages), (
            "History messages should NOT be injected in CHAT path"
        )
