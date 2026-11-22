"""Text -> Chat system context in the tracked ``image.json`` graph.

The provider client is faked at the ``MagicLLM`` constructor (``mock_magic_llm``);
build, routing, the Chat node and the LLM node all run for real.
"""

import copy

import pytest

from magic_agents.agt_flow import build, run_agent
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from tests.support import debug_errors, load_fixture_graph

SYSTEM_PROMPT = "Write always on French"


@pytest.fixture
def image_graph():
    return load_fixture_graph("image.json")


def _node_id(graph_def, node_type):
    return next(node["id"] for node in graph_def["content"]["nodes"] if node["type"] == node_type)


def _streamed_text(events):
    text = ""
    for event in events:
        content = event.get("content") if event.get("type") == "content" else None
        if getattr(content, "choices", None) and content.choices[0].delta.content:
            text += content.choices[0].delta.content
    return text


def test_chat_tracker_waits_for_system_context_message_and_images(image_graph):
    graph = build(image_graph, "Hello", load_chat=None)

    tracker = GraphEventDispatcher(graph.nodes, graph.edges).get_tracker(_node_id(image_graph, "chat"))

    assert {"handle-system-context", "handle_user_message", "handle_user_images"} <= set(tracker.expected_handles)


@pytest.mark.parametrize("ordering", ["authored", "reversed_edges", "reversed_nodes"])
async def test_system_context_reaches_chat_regardless_of_authoring_order(image_graph, mock_magic_llm, ordering):
    graph_def = copy.deepcopy(image_graph)
    if ordering == "reversed_edges":
        graph_def["content"]["edges"].reverse()
    elif ordering == "reversed_nodes":
        graph_def["content"]["nodes"].reverse()
    graph = build(graph_def, "Hi", load_chat=None)
    chat_node = graph.nodes[_node_id(image_graph, "chat")]

    events = [event async for event in run_agent(graph)]

    assert debug_errors(events) == []
    assert chat_node.inputs["handle-system-context"] == SYSTEM_PROMPT
    assert [(m["role"], m["content"]) for m in chat_node.chat.messages[:2]] == [
        ("system", SYSTEM_PROMPT),
        ("user", "Hi"),
    ]
    assert _streamed_text(events) == "mocked LLM response"
