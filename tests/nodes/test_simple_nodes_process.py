"""Focused process tests for lightweight node implementations."""

import pytest
from magic_llm.model.ModelChatStream import ChatCompletionModel, ChoiceModel

from magic_agents.agt_flow import create_node
from magic_agents.models.factory.Nodes import (
    ModelAgentFlowTypesModel,
    ParserNodeModel,
    SendMessageNodeModel,
    TextNodeModel,
    UserInputNodeModel,
)
from magic_agents.models.model_agent_run_log import ModelAgentRunLog
from magic_agents.node_system.NodeEND import NodeEND
from magic_agents.node_system.NodeVoid import NodeVoid
from magic_agents.node_system.NodeParser import NodeParser
from magic_agents.node_system.NodeSendMessage import NodeSendMessage
from magic_agents.node_system.NodeText import NodeText
from magic_agents.node_system.NodeUserInput import NodeUserInput


async def collect_business_events(node, chat_log=None):
    events = [
        event
        async for event in node.process(chat_log or ModelAgentRunLog())
    ]
    return [event for event in events if event["type"] != "content"]


async def collect_events(node, chat_log=None):
    return [
        event
        async for event in node.process(chat_log or ModelAgentRunLog())
    ]


@pytest.mark.asyncio
async def test_text_node_emits_configured_text_handle():
    node = NodeText(
        data=TextNodeModel(text="hello"),
        handles={"output": "custom-text"},
        node_id="text-1",
    )

    events = await collect_business_events(node)

    assert events == [
        {"type": "custom-text", "content": {"node": "NodeText", "content": "hello"}}
    ]


@pytest.mark.asyncio
async def test_user_input_emits_message_files_images_extras_and_sets_session():
    chat_log = ModelAgentRunLog()
    node = NodeUserInput(
        data=UserInputNodeModel(
            text="hello",
            files=["file.txt"],
            images=["image.png"],
            extras={"locale": "en"},
            session_id="thread-1",
        ),
        node_id="input-1",
    )

    events = await collect_business_events(node, chat_log)

    assert chat_log.id_chat == "thread-1"
    assert chat_log.id_thread == "thread-1"
    assert [(event["type"], event["content"]["content"]) for event in events] == [
        ("handle_user_message", "hello"),
        ("handle_user_files", ["file.txt"]),
        ("handle_user_images", ["image.png"]),
        ("handle_client_extras", {"locale": "en"}),
    ]


@pytest.mark.asyncio
async def test_user_input_normalizes_scalar_files_and_images_before_emitting():
    model = UserInputNodeModel(
        text="hello",
        files="file.txt",
        images={"url": "image.png"},
    )
    node = NodeUserInput(data=model, node_id="input-scalars")

    events = await collect_business_events(node)

    assert model.files == ["file.txt"]
    assert model.images == [{"url": "image.png"}]
    assert [(event["type"], event["content"]["content"]) for event in events] == [
        ("handle_user_message", "hello"),
        ("handle_user_files", ["file.txt"]),
        ("handle_user_images", [{"url": "image.png"}]),
        # No extras in the request and no default: Extras still fires, with {}.
        ("handle_client_extras", {}),
    ]


@pytest.mark.asyncio
async def test_user_input_session_required_leaves_backend_session_creation_pending():
    chat_log = ModelAgentRunLog()
    node = NodeUserInput(
        data=UserInputNodeModel(text="hello", session_required=True),
        node_id="input-session-required",
    )

    events = await collect_business_events(node, chat_log)

    assert chat_log.id_chat is None
    assert chat_log.id_thread is None
    assert events[0]["content"]["content"] == "hello"


@pytest.mark.asyncio
async def test_user_input_preserves_backend_session_when_no_session_is_configured():
    chat_log = ModelAgentRunLog(
        id_chat="backend-chat",
        id_thread="backend-thread",
    )
    node = NodeUserInput(
        data=UserInputNodeModel(text="hello"),
        node_id="input-backend-session",
    )

    await collect_business_events(node, chat_log)

    assert chat_log.id_chat == "backend-chat"
    assert chat_log.id_thread == "backend-thread"


@pytest.mark.asyncio
async def test_user_input_configured_session_does_not_replace_existing_thread():
    chat_log = ModelAgentRunLog(
        id_chat="backend-chat",
        id_thread="backend-thread",
    )
    node = NodeUserInput(
        data=UserInputNodeModel(text="hello", session_id="configured-chat"),
        node_id="input-existing-thread",
    )

    await collect_business_events(node, chat_log)

    assert chat_log.id_chat == "configured-chat"
    assert chat_log.id_thread == "backend-thread"


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["content", "message"])
async def test_user_input_text_aliases_emit_the_aliased_value(alias):
    node = create_node(
        {
            "id": f"input-{alias}",
            "type": ModelAgentFlowTypesModel.USER_INPUT,
            "data": {alias: "aliased input"},
        },
        load_chat=None,
    )

    events = await collect_business_events(node)

    assert [(event["type"], event["content"]["content"]) for event in events] == [
        ("handle_user_message", "aliased input"),
        ("handle_user_files", None),
        ("handle_user_images", None),
        ("handle_client_extras", {}),
    ]


@pytest.mark.asyncio
async def test_user_input_without_text_emits_none_and_empty_extras():
    node = create_node(
        {"id": "input-empty", "type": ModelAgentFlowTypesModel.USER_INPUT},
        load_chat=None,
    )

    events = await collect_business_events(node)

    assert [(event["type"], event["content"]["content"]) for event in events] == [
        ("handle_user_message", None),
        ("handle_user_files", None),
        ("handle_user_images", None),
        ("handle_client_extras", {}),
    ]


@pytest.mark.asyncio
async def test_user_input_custom_handles_are_used_for_each_output():
    node = create_node(
        {
            "id": "input-custom-handles",
            "type": ModelAgentFlowTypesModel.USER_INPUT,
            "data": {
                "text": "hello",
                "files": ["file.txt"],
                "images": ["image.png"],
                "extras": {"locale": "en"},
                "handles": {
                    "user_message": "custom-message",
                    "user_files": "custom-files",
                    "user_images": "custom-images",
                    "client_extras": "custom-extras",
                },
            },
        },
        load_chat=None,
    )

    events = await collect_business_events(node)

    assert [(event["type"], event["content"]["content"]) for event in events] == [
        ("custom-message", "hello"),
        ("custom-files", ["file.txt"]),
        ("custom-images", ["image.png"]),
        ("custom-extras", {"locale": "en"}),
    ]


@pytest.mark.asyncio
async def test_parser_node_parses_json_input_before_rendering_template():
    node = NodeParser(
        data=ParserNodeModel(text="Name={{ handle_parser_input.name }}"),
        node_id="parser-1",
    )
    node.inputs["handle_parser_input"] = '{"name": "Ada"}'

    events = await collect_business_events(node)

    assert events == [
        {
            "type": "handle_parser_output",
            "content": {"node": "NodeParser", "content": "Name=Ada"},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["content", "template"])
async def test_parser_template_aliases_render_routed_input(alias):
    node = create_node(
        {
            "id": f"parser-{alias}",
            "type": ModelAgentFlowTypesModel.PARSER,
            "data": {alias: "Name={{ arbitrary_input.name }}"},
        },
        load_chat=None,
    )
    node.inputs["arbitrary_input"] = '{"name": "Ada"}'

    events = await collect_business_events(node)

    assert events == [
        {
            "type": "handle_parser_output",
            "content": {"node": "NodeParser", "content": "Name=Ada"},
        }
    ]


@pytest.mark.asyncio
async def test_parser_without_template_emits_empty_string():
    node = create_node(
        {"id": "parser-empty", "type": ModelAgentFlowTypesModel.PARSER},
        load_chat=None,
    )

    events = await collect_business_events(node)

    assert events == [
        {
            "type": "handle_parser_output",
            "content": {"node": "NodeParser", "content": ""},
        }
    ]


@pytest.mark.asyncio
async def test_send_message_emits_stream_and_routing_events_with_extras():
    node = NodeSendMessage(
        data=SendMessageNodeModel(message="visible message", json_extras="legacy fallback"),
        node_id="send-1",
    )
    node.inputs[node.INPUT_HANDLE_SEND_EXTRA] = '{"source": "test"}'

    events = await collect_events(node)
    stream_events = [
        event for event in events
        if event["type"] == "content"
        and getattr(event["content"]["content"], "extras", None) == {"source": "test"}
    ]
    route_events = [event for event in events if event["type"] == "handle_message_output"]
    business_events = stream_events + route_events

    assert [event["type"] for event in business_events] == ["content", "handle_message_output"]
    for event in business_events:
        chunk = event["content"]["content"]
        assert chunk.choices[0].delta.content == "visible message"
        assert chunk.extras == {"source": "test"}


@pytest.mark.asyncio
async def test_send_message_uses_json_extras_as_legacy_content_fallback():
    node = NodeSendMessage(
        data=SendMessageNodeModel(message="", json_extras="legacy content"),
        node_id="send-legacy",
    )

    events = await collect_events(node)
    chunks = [
        event["content"]["content"]
        for event in events
        if event["type"] in {"content", "handle_message_output"}
        and hasattr(event["content"]["content"], "choices")
        and event["content"]["content"].choices[0].delta.content == "legacy content"
    ]

    assert len(chunks) == 2
    assert [chunk.choices[0].delta.content for chunk in chunks] == [
        "legacy content",
        "legacy content",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_data",
    [
        {"content": "content alias"},
        {"extras": "extras alias"},
    ],
)
async def test_send_message_content_aliases_emit_stream_and_route_events(node_data):
    node = create_node(
        {
            "id": "send-alias",
            "type": ModelAgentFlowTypesModel.SEND_MESSAGE,
            "data": node_data,
        },
        load_chat=None,
    )
    expected = next(iter(node_data.values()))

    events = await collect_events(node)
    business_events = [
        event
        for event in events
        if event["type"] in {"content", "handle_message_output"}
        and hasattr(event["content"]["content"], "choices")
        and event["content"]["content"].choices[0].delta.content == expected
    ]

    assert [event["type"] for event in business_events] == [
        "content",
        "handle_message_output",
    ]
    assert [
        event["content"]["content"].choices[0].delta.content
        for event in business_events
    ] == [expected, expected]


@pytest.mark.asyncio
async def test_end_node_emits_terminal_handle():
    node = NodeEND(handles={"output": "done"}, node_id="end-1")

    events = await collect_business_events(node)

    assert events == [
        {
            "type": "done",
            "content": {
                "node": "NodeEND",
                "content": ChatCompletionModel(
                    id="",
                    model="",
                    choices=[ChoiceModel()],
                ),
            },
        }
    ]


@pytest.mark.asyncio
async def test_text_content_alias_emits_aliased_value():
    node = create_node(
        {
            "id": "text-content",
            "type": ModelAgentFlowTypesModel.TEXT,
            "data": {"content": "aliased text"},
        },
        load_chat=None,
    )

    events = await collect_business_events(node)

    assert events == [
        {
            "type": "handle_text_output",
            "content": {"node": "NodeText", "content": "aliased text"},
        }
    ]


@pytest.mark.asyncio
async def test_text_without_content_emits_empty_string():
    node = create_node(
        {"id": "text-empty", "type": ModelAgentFlowTypesModel.TEXT},
        load_chat=None,
    )

    events = await collect_business_events(node)

    assert events == [
        {
            "type": "handle_text_output",
            "content": {"node": "NodeText", "content": ""},
        }
    ]


@pytest.mark.asyncio
async def test_void_node_uses_terminal_sink_behavior():
    node = create_node(
        {"id": "void-1", "type": ModelAgentFlowTypesModel.VOID},
        load_chat=None,
    )

    events = await collect_business_events(node)

    assert isinstance(node, NodeVoid)
    assert node.node_type == ModelAgentFlowTypesModel.VOID
    assert events == []
    assert node._capture_internal_state()["is_silent_sink"] is True
