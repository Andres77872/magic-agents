"""Tests for Chat node merge order verification.

Phase 5.5: Merge order tests per spec.md scenarios:
- Full merge path with append mode
- Legacy replace path preserves old behavior
- Custom messages appended before user
- System context override at index 0
- Merge order invariant: persisted → runtime → custom → user

The merge order MUST always follow the strict sequence:
1. history_messages from backend (persisted + runtime) - injected via build()
2. Runtime messages input (APPEND or REPLACE based on mode) - from input handles
3. custom_messages from config (APPEND)
4. System context (INSERT at index 0)
5. User message (APPEND, final slot)

ARCHITECTURE NOTE:
- Slot 1 (history_messages) is BACKEND-AUTHORITATIVE, injected via build() call.
- Backend (api.magic_llm) builds agent_messages = [persisted] + [runtime] + [user]
- Backend passes history_messages = agent_messages[:-1] to frontend build()
- Frontend stamps history_messages into Chat node's data
- NodeChat reads history_messages from ChatNodeModel and uses as Slot 1 base
- This eliminates the need for callback architecture or frontend DB calls.

UPDATED: 2026-04-26 - Cross-repo wiring fix implemented.
"""
import pytest

from magic_agents.models.factory.Nodes.ChatNodeModel import ChatNodeModel
from magic_agents.node_system.NodeChat import NodeChat
from magic_agents.node_system.utils import estimate_tokens


class MockChatLog:
    """Mock chat_log object for testing."""
    def __init__(self, id_chat=None):
        self.id_chat = id_chat


def extract_model_chat(outputs):
    """Return the single routed ModelChat and fail if NodeChat emitted none."""
    chat_outputs = [
        output for output in outputs
        if output.get("type") == "handle_chat_output"
    ]
    assert len(chat_outputs) == 1, (
        f"Expected one handle_chat_output, got {len(chat_outputs)}: {outputs}"
    )
    envelope = chat_outputs[0]["content"]
    assert envelope["node"] == "NodeChat"
    model_chat = envelope["content"]
    assert hasattr(model_chat, "messages")
    return model_chat


@pytest.mark.asyncio
class TestChatAttachments:
    """Real ``ModelChat`` behavior for the declared image and file handles."""

    async def test_process_builds_multimodal_user_message_from_images(self):
        node = NodeChat(data=ChatNodeModel())
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Describe these images"
        node.inputs[node.INPUT_HANDLER_USER_IMAGES] = [
            "https://images.invalid/one.png",
            "https://images.invalid/two.png",
        ]

        outputs = [item async for item in node.process(MockChatLog())]
        chat = extract_model_chat(outputs)

        assert chat.messages == [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe these images"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://images.invalid/one.png"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://images.invalid/two.png"},
                },
            ],
        }]

    async def test_process_accepts_json_encoded_image_list(self):
        node = NodeChat(data=ChatNodeModel())
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Inspect"
        node.inputs[node.INPUT_HANDLER_USER_IMAGES] = (
            '["https://images.invalid/encoded.png"]'
        )

        outputs = [item async for item in node.process(MockChatLog())]
        chat = extract_model_chat(outputs)

        assert chat.messages[0]["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "https://images.invalid/encoded.png"},
        }

    async def test_process_consumes_file_descriptors_before_current_prompt(self):
        node = NodeChat(data=ChatNodeModel())
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Answer using the attachments"
        node.inputs[node.INPUT_HANDLER_USER_FILES] = [
            ["Extracted page one", "https://files.invalid/page-one.png"],
            {"text": "Extracted text document"},
        ]

        outputs = [item async for item in node.process(MockChatLog())]
        chat = extract_model_chat(outputs)

        assert chat.messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extracted page one"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://files.invalid/page-one.png"},
                    },
                ],
            },
            {"role": "user", "content": "Extracted text document"},
            {"role": "user", "content": "Answer using the attachments"},
        ]

    async def test_process_preserves_legacy_file_pairs_on_images_handle(self):
        node = NodeChat(data=ChatNodeModel())
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Current prompt"
        node.inputs[node.INPUT_HANDLER_USER_IMAGES] = [
            ["Legacy extracted page", "https://files.invalid/legacy-page.png"],
        ]

        outputs = [item async for item in node.process(MockChatLog())]
        chat = extract_model_chat(outputs)

        assert chat.messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Legacy extracted page"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://files.invalid/legacy-page.png"},
                    },
                ],
            },
            {"role": "user", "content": "Current prompt"},
        ]

    async def test_real_graph_routes_declared_file_handle_into_chat(self):
        """UserInput -> Chat proves file consumption through production edges."""
        from magic_agents.agt_flow import build, run_agent

        graph = build(
            {
                "type": "graph",
                "contract_config": {"mode": "strict"},
                "nodes": [
                    {
                        "id": "input",
                        "type": "user_input",
                        "data": {
                            "files": [{"text": "Graph-routed document text"}],
                        },
                    },
                    {"id": "chat", "type": "chat"},
                    {"id": "end", "type": "end"},
                ],
                "edges": [
                    {
                        "id": "message-chat",
                        "source": "input",
                        "target": "chat",
                        "sourceHandle": "handle_user_message",
                        "targetHandle": "handle_user_message",
                    },
                    {
                        "id": "files-chat",
                        "source": "input",
                        "target": "chat",
                        "sourceHandle": "handle_user_files",
                        "targetHandle": "handle_user_files",
                    },
                    {
                        "id": "chat-end",
                        "source": "chat",
                        "target": "end",
                        "sourceHandle": "handle_chat_output",
                        "targetHandle": "handle_flow_input",
                    },
                ],
            },
            message="Current graph prompt",
        )

        assert not graph.contract_report.has_errors()
        async for _ in run_agent(graph):
            pass

        assert graph.nodes["chat"].chat.messages == [
            {"role": "user", "content": "Graph-routed document text"},
            {"role": "user", "content": "Current graph prompt"},
        ]
        assert graph.nodes["end"].response is not None

    async def test_process_combines_files_and_images_on_the_same_turn(self):
        node = NodeChat(data=ChatNodeModel(), node_id="chat-attachments")
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Inspect"
        node.inputs[node.INPUT_HANDLER_USER_IMAGES] = ["https://images.invalid/one.png"]
        node.inputs[node.INPUT_HANDLER_USER_FILES] = [{"text": "file text"}]

        outputs = [item async for item in node.process(MockChatLog())]
        chat = extract_model_chat(outputs)
        assert chat.messages == [
            {"role": "user", "content": "file text"},
            {"role": "user", "content": [
                {"type": "text", "text": "Inspect"},
                {"type": "image_url", "image_url": {"url": "https://images.invalid/one.png"}},
            ]},
        ]
        assert not any(item.get("type") == "debug" for item in outputs)

    async def test_process_rejects_unresolved_file_reference_instead_of_ignoring_it(self):
        node = NodeChat(data=ChatNodeModel(), node_id="chat-invalid-file")
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Inspect"
        node.inputs[node.INPUT_HANDLER_USER_FILES] = ["unresolved-file.txt"]

        outputs = [item async for item in node.process(MockChatLog())]

        assert not any(item.get("type") == node.OUTPUT_HANDLE for item in outputs)
        diagnostic = next(item for item in outputs if item.get("type") == "debug")
        assert diagnostic["content"]["error_type"] == "ValidationError"
        assert "must be a [text, image] pair" in diagnostic["content"]["error_message"]


class TestHistoryMessagesInjection:
    """Tests for backend-authoritative history_messages injection (Slot 1)."""

    @pytest.mark.asyncio
    async def test_full_merge_order_with_history_messages(self):
        """Full merge order: history_messages → runtime → custom → user."""
        model = ChatNodeModel(
            history_messages=[
                {"role": "user", "content": "HISTORY: question 1"},
                {"role": "assistant", "content": "HISTORY: answer 1"}
            ],
            messages_append_mode=True,
            custom_messages=[
                {"role": "assistant", "content": "CUSTOM: context"}
            ]
        )
        node = NodeChat(data=model)
        
        # Runtime messages (Slot 2)
        node.inputs[node.INPUT_HANDLER_MESSAGES] = [
            {"role": "user", "content": "RUNTIME: injected"}
        ]
        # User message (Slot 5)
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "USER: current"
        
        chat_log = MockChatLog(id_chat="conv-test")
        
        outputs = []
        async for output in node.process(chat_log):
            outputs.append(output)
        
        messages = extract_model_chat(outputs).messages
        contents = [str(message.get("content", "")) for message in messages]
        history_pos = [i for i, content in enumerate(contents) if "HISTORY:" in content]
        runtime_pos = [i for i, content in enumerate(contents) if "RUNTIME:" in content]
        custom_pos = [i for i, content in enumerate(contents) if "CUSTOM:" in content]
        user_pos = [i for i, content in enumerate(contents) if "USER:" in content]

        assert len(history_pos) == 2
        assert len(runtime_pos) == 1
        assert len(custom_pos) == 1
        assert len(user_pos) == 1
        assert max(history_pos) < min(runtime_pos)
        assert max(runtime_pos) < min(custom_pos)
        assert max(custom_pos) < min(user_pos)

    @pytest.mark.asyncio
    async def test_replace_mode_still_discards_history_messages(self):
        """Replace mode (messages_append_mode=False) discards history_messages per legacy semantics."""
        model = ChatNodeModel(
            history_messages=[
                {"role": "user", "content": "This should be replaced"},
                {"role": "assistant", "content": "This too"}
            ],
            messages_append_mode=False  # REPLACE mode
        )
        node = NodeChat(data=model)
        
        # Runtime messages that REPLACE base
        node.inputs[node.INPUT_HANDLER_MESSAGES] = [
            {"role": "user", "content": "New runtime message"}
        ]
        
        chat_log = MockChatLog(id_chat="conv-test")
        
        outputs = []
        async for output in node.process(chat_log):
            outputs.append(output)
        
        messages = extract_model_chat(outputs).messages
        contents = [message.get("content") for message in messages]

        assert "This should be replaced" not in contents
        assert "This too" not in contents
        assert contents == ["New runtime message"]


class TestChatMergeOrder:
    """Test Chat node merge order per spec.md scenarios."""

    def test_base_messages_empty_initialization(self):
        """NodeChat initializes with empty messages."""
        model = ChatNodeModel()
        node = NodeChat(data=model)
        assert node.chat.messages == []

        # Old history discarded (legacy REPLACE semantics)

    def test_handles_override_default_names(self):
        """Handle names can be overridden via config."""
        model = ChatNodeModel(
            handles={
                "messages": "custom_messages_input",
                "user_message": "custom_user_input"
            }
        )
        node = NodeChat(data=model)
        
        assert node.INPUT_HANDLER_MESSAGES == "custom_messages_input"
        assert node.INPUT_HANDLER_USER_MESSAGE == "custom_user_input"


@pytest.mark.asyncio
class TestChatProcessAsync:
    """Async tests for NodeChat.process() method."""

    async def test_process_yields_model_chat(self):
        """NodeChat.process() yields ModelChat with merged messages."""
        model = ChatNodeModel(
            messages_append_mode=True,
            custom_messages=[
                {"role": "assistant", "content": "Pre-user context"}
            ]
        )
        node = NodeChat(data=model)
        
        # Set inputs via node.inputs dict (NOT set_input method)
        node.inputs[node.INPUT_HANDLER_MESSAGES] = [
            {"role": "user", "content": "Runtime input"}
        ]
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Current question"
        
        # Create mock chat_log
        chat_log = MockChatLog(id_chat="conv-test")
        
        # Collect yielded outputs
        outputs = []
        async for output in node.process(chat_log):
            outputs.append(output)
        
        messages = extract_model_chat(outputs).messages
        contents = [message.get("content") for message in messages]

        assert contents == ["Runtime input", "Pre-user context", "Current question"]

    async def test_process_with_system_context(self):
        """System context is inserted at index 0."""
        model = ChatNodeModel(messages_append_mode=True)
        node = NodeChat(data=model)
        
        # Set system context input via node.inputs dict
        node.inputs[node.INPUT_HANDLER_SYSTEM_CONTEXT] = "You are a helpful assistant."
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Hello"
        
        chat_log = MockChatLog(id_chat="conv-test")
        
        outputs = []
        async for output in node.process(chat_log):
            outputs.append(output)
        
        messages = extract_model_chat(outputs).messages

        assert messages
        assert messages[0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }

    async def test_process_merge_order_append_mode(self):
        """Verify merge order in append mode: runtime → custom → user."""
        model = ChatNodeModel(
            messages_append_mode=True,
            custom_messages=[
                {"role": "assistant", "content": "Custom context A"},
                {"role": "assistant", "content": "Custom context B"}
            ]
        )
        node = NodeChat(data=model)
        
        # Set runtime messages input
        node.inputs[node.INPUT_HANDLER_MESSAGES] = [
            {"role": "user", "content": "Runtime A"},
            {"role": "assistant", "content": "Runtime B"}
        ]
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Final user question"
        
        chat_log = MockChatLog(id_chat="conv-test")
        
        outputs = []
        async for output in node.process(chat_log):
            outputs.append(output)
        
        messages = extract_model_chat(outputs).messages
        contents = [str(message.get("content", "")) for message in messages]
        runtime_positions = [
            i for i, content in enumerate(contents)
            if content in {"Runtime A", "Runtime B"}
        ]
        custom_positions = [
            i for i, content in enumerate(contents)
            if content in {"Custom context A", "Custom context B"}
        ]
        user_positions = [
            i for i, content in enumerate(contents)
            if "Final user question" in content
        ]

        assert len(runtime_positions) == 2
        assert len(custom_positions) == 2
        assert len(user_positions) == 1
        assert max(runtime_positions) < min(custom_positions)
        assert max(custom_positions) < user_positions[0]

    async def test_process_replace_mode_discards_runtime_messages(self):
        """Replace mode discards runtime messages and starts fresh."""
        model = ChatNodeModel(messages_append_mode=False)  # REPLACE mode
        node = NodeChat(data=model)
        
        # Set runtime messages input (should be REPLACED, not extended)
        node.inputs[node.INPUT_HANDLER_MESSAGES] = [
            {"role": "user", "content": "This replaces everything"}
        ]
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "New question"
        
        chat_log = MockChatLog(id_chat="conv-test")
        
        outputs = []
        async for output in node.process(chat_log):
            outputs.append(output)
        
        messages = extract_model_chat(outputs).messages
        contents = [message.get("content") for message in messages]

        assert contents == ["This replaces everything", "New question"]


@pytest.mark.asyncio
class TestPersistedHistoryArchitectureAsync:
    """Async tests for persisted history architecture."""

    async def test_base_messages_starts_empty(self):
        """NodeChat starts with empty base_messages in process()."""
        model = ChatNodeModel(messages_append_mode=True)
        node = NodeChat(data=model)
        
        chat_log = MockChatLog(id_chat="conv-test")
        
        # Process without any inputs
        outputs = []
        async for output in node.process(chat_log):
            outputs.append(output)
        
        messages = extract_model_chat(outputs).messages

        assert messages == []

    async def test_persisted_history_via_runtime_messages(self):
        """Persisted history is received via runtime messages input (backend-provided)."""
        model = ChatNodeModel(messages_append_mode=True)
        node = NodeChat(data=model)
        
        # Simulate backend providing persisted history via runtime messages
        # This is how the backend-backend contract works in production
        persisted_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        node.inputs[node.INPUT_HANDLER_MESSAGES] = persisted_history
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "New question"
        
        chat_log = MockChatLog(id_chat="conv-test")
        
        outputs = []
        async for output in node.process(chat_log):
            outputs.append(output)
        
        messages = extract_model_chat(outputs).messages
        contents = [message.get("content") for message in messages]

        assert contents == ["Previous question", "Previous answer", "New question"]

    # end of TestPersistedHistoryArchitectureAsync


@pytest.mark.asyncio
class TestChatWindowingIntegration:
    """Integration tests for STM windowing in full NodeChat pipeline.

    Verifies that ``apply_windowing()`` is correctly invoked between the
    5-slot merge and the yield in ``NodeChat.process()``.  These tests
    exercise the full pipeline: Slot 1 (history) → Slot 2 (runtime) →
    Slot 3 (custom) → Slot 4 (system) → Slot 5 (user) → Windowing.

    See spec X5 and tasks 6.2 for the complete scenario list.
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_chat(outputs: list[dict]) -> object:
        """Extract the ModelChat instance from process() output."""
        for output in outputs:
            if output.get('type') == 'handle_chat_output':
                return output['content']['content']
        return None

    # ------------------------------------------------------------------
    # X5: Integration scenarios
    # ------------------------------------------------------------------

    async def test_full_pipeline_windowing(self):
        """50 msgs │ max_messages=10 → ≤10 non-system msgs.

        Post-merge windowing reduces yielded message count to the
        configured max_messages limit. (N1, X5)
        """
        history: list[dict] = []
        for i in range(25):
            history.append({"role": "user", "content": f"History message {i}"})
            history.append({"role": "assistant", "content": f"History response {i}"})

        model = ChatNodeModel(
            history_messages=history,
            messages_append_mode=True,
            max_messages=10,
        )
        node = NodeChat(data=model)
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Current turn question"

        outputs: list[dict] = []
        async for output in node.process(MockChatLog(id_chat="conv-stm")):
            outputs.append(output)

        chat = self._extract_chat(outputs)
        assert chat is not None, "No ModelChat output found"

        messages = chat.messages
        non_system = [m for m in messages if m.get("role") != "system"]

        # The newest 9 history messages plus the current turn.
        assert len(non_system) == 10
        assert non_system[0]["content"] == "History response 20"

        # Last message must be the user's current turn (Slot 5)
        assert messages[-1]["role"] == "user"
        assert "Current turn question" in str(messages[-1].get("content", ""))

    async def test_system_preservation_end_to_end(self):
        """System context survives 5-slot merge + windowing.

        System messages injected via Slot 4 are preserved after
        max_messages truncation. (W2, X5)
        """
        history = [{"role": "user", "content": f"M{i}"} for i in range(20)]

        model = ChatNodeModel(
            history_messages=history,
            messages_append_mode=True,
            max_messages=5,
        )
        node = NodeChat(data=model)

        # Slot 4: Inject system context
        node.inputs[node.INPUT_HANDLER_SYSTEM_CONTEXT] = (
            "You are a helpful assistant."
        )
        # Slot 5: User message
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Hello"

        outputs: list[dict] = []
        async for output in node.process(MockChatLog(id_chat="conv-stm")):
            outputs.append(output)

        chat = self._extract_chat(outputs)
        assert chat is not None, "No ModelChat output found"

        messages = chat.messages

        # System message MUST be present (W2)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        assert len(system_msgs) >= 1, "System message should survive windowing"
        assert any(
            "helpful assistant" in m.get("content", "") for m in system_msgs
        ), "System content should match input"

        # Non-system messages ≤ max_messages
        non_system = [m for m in messages if m.get("role") != "system"]
        assert len(non_system) <= 5, (
            f"Expected ≤5 non-system messages, got {len(non_system)}"
        )

        # Last message should be the current user turn
        assert messages[-1]["role"] == "user"
        assert "Hello" in str(messages[-1].get("content", ""))

    async def test_tool_chain_end_to_end(self):
        """Tool-call chain preserved atomically through full pipeline.

        An assistant + tool message pair must not be split by
        windowing. (W6, X5)
        """
        history: list[dict] = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {"role": "tool", "content": "72°F, sunny", "tool_call_id": "call_1"},
            {"role": "user", "content": "Thanks!"},
        ]

        model = ChatNodeModel(
            history_messages=history,
            messages_append_mode=True,
            max_messages=3,
        )
        node = NodeChat(data=model)
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Current question"

        outputs: list[dict] = []
        async for output in node.process(MockChatLog(id_chat="conv-stm")):
            outputs.append(output)

        chat = self._extract_chat(outputs)
        assert chat is not None, "No ModelChat output found"

        messages = chat.messages
        tool_calls_msgs = [m for m in messages if m.get("tool_calls")]
        tool_result_msgs = [m for m in messages if m.get("role") == "tool"]

        assert len(tool_calls_msgs) == 1
        assert len(tool_result_msgs) == 1
        assert tool_calls_msgs[0]["tool_calls"][0]["id"] == "call_1"
        assert tool_result_msgs[0]["tool_call_id"] == "call_1"

    async def test_all_defaults_no_change(self):
        """50 messages at STM defaults → all preserved.

        With both limits at None, windowing MUST NOT be applied.
        (W8, N1)
        """
        history = [{"role": "user", "content": f"M{i}"} for i in range(50)]

        model = ChatNodeModel(
            history_messages=history,
            messages_append_mode=True,
        )
        node = NodeChat(data=model)
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "More data?"

        outputs: list[dict] = []
        async for output in node.process(MockChatLog(id_chat="conv-stm")):
            outputs.append(output)

        chat = self._extract_chat(outputs)
        assert chat is not None, "No ModelChat output found"

        messages = chat.messages
        assert [m["content"] for m in messages] == [f"M{i}" for i in range(50)] + ["More data?"]

    async def test_max_input_tokens_applied(self, monkeypatch):
        """Token budget enforced post-merge.

        When max_input_tokens is below total, output must fit
        within budget. (W5, X5, N1)
        """
        class LocalEncoding:
            def encode(self, value):
                return str(value).split()

        monkeypatch.setattr(
            "magic_agents.node_system.utils._get_encoding",
            lambda: LocalEncoding(),
        )
        history = [{"role": "user", "content": f"M{i}"} for i in range(50)]

        full_list = history + [{"role": "user", "content": "Latest"}]
        total_tokens = estimate_tokens(full_list)
        budget = total_tokens // 4

        model = ChatNodeModel(
            history_messages=history,
            messages_append_mode=True,
            max_input_tokens=budget,
        )
        node = NodeChat(data=model)
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Latest"

        outputs: list[dict] = []
        async for output in node.process(MockChatLog(id_chat="conv-stm")):
            outputs.append(output)

        chat = self._extract_chat(outputs)
        assert chat is not None, "No ModelChat output found"

        messages = chat.messages
        output_tokens = estimate_tokens(messages)

        assert output_tokens <= budget, (
            f"Expected ≤{budget} tokens, got {output_tokens}"
        )
        assert messages[-1]["role"] == "user"
        assert "Latest" in str(messages[-1].get("content", ""))

    async def test_messages_append_mode_with_windowing(self):
        """APPEND mode with runtime messages + windowing.

        Slot 1 + Slot 2 messages merged then windowed.  (X5)
        """
        history = [{"role": "user", "content": f"OLD{i}"} for i in range(20)]
        runtime = [{"role": "user", "content": f"NEW{i}"} for i in range(15)]

        model = ChatNodeModel(
            history_messages=history,
            messages_append_mode=True,
            max_messages=5,
        )
        node = NodeChat(data=model)
        node.inputs[node.INPUT_HANDLER_MESSAGES] = runtime
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Final"

        outputs: list[dict] = []
        async for output in node.process(MockChatLog(id_chat="conv-stm")):
            outputs.append(output)

        chat = self._extract_chat(outputs)
        assert chat is not None, "No ModelChat output found"

        messages = chat.messages
        non_system = [m for m in messages if m.get("role") != "system"]

        assert len(non_system) <= 5, (
            f"Expected ≤5 non-system messages, got {len(non_system)}"
        )
        assert "Final" in str(messages[-1].get("content", ""))

    # ------------------------------------------------------------------
    # Diagnostics verification
    # ------------------------------------------------------------------

    async def test_diagnostics_after_windowing(self):
        """_capture_internal_state reports correct windowing stats.

        After process() with max_messages=10 on 50 history msgs,
        diagnostics show accurate counts.  (N3)
        """
        history = [{"role": "user", "content": f"M{i}"} for i in range(50)]

        model = ChatNodeModel(
            history_messages=history,
            messages_append_mode=True,
            max_messages=10,
        )
        node = NodeChat(data=model)
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "Test"

        outputs: list[dict] = []
        async for output in node.process(MockChatLog(id_chat="conv-stm")):
            outputs.append(output)

        state = node._capture_internal_state()

        assert state["max_messages"] == 10
        assert state["max_input_tokens"] is None
        assert state["truncation_strategy"] == "tail"
        assert state["windowing_applied"] is True
        assert state["messages_before_windowing"] == 51  # 50 history + 1 user
        assert state["messages_after_windowing"] == 10
        assert state["messages_discarded"] == 41
        assert state["total_tokens_estimated"] is None

    async def test_diagnostics_defaults_no_windowing(self):
        """Diagnostics show None/False when windowing not configured.

        Nodes with all-default STM fields report
        windowing_applied=False.  (N3)
        """
        model = ChatNodeModel(
            history_messages=[{"role": "user", "content": "Hello"}],
            messages_append_mode=True,
        )
        node = NodeChat(data=model)
        node.inputs[node.INPUT_HANDLER_USER_MESSAGE] = "World"

        outputs: list[dict] = []
        async for output in node.process(MockChatLog(id_chat="conv-stm")):
            outputs.append(output)

        state = node._capture_internal_state()

        assert state["max_messages"] is None
        assert state["max_input_tokens"] is None
        assert state["truncation_strategy"] == "tail"
        assert state["windowing_applied"] is False
        assert state["messages_before_windowing"] == 0
        assert state["messages_discarded"] == 0
        assert state["total_tokens_estimated"] is None
