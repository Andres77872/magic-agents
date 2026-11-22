"""A conditional chooses between a canned reply and an LLM, end to end through ``run_agent``.

The provider is faked at the ``MagicLLM`` constructor (``mock_magic_llm``); the
conditional template, routing, bypass and SendMessage nodes are real.
"""

import copy

import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.hooks import RuntimeConfig
from tests.support import RecordingHook, debug_errors

FLOW = {
    "type": "chat",
    "edges": [
        {"id": "e1", "source": "user-input", "target": "cond", "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
        {"id": "e2", "source": "cond", "target": "send-msg", "sourceHandle": "empty", "targetHandle": "handle_send_extra"},
        {"id": "e3", "source": "cond", "target": "llm-node", "sourceHandle": "not_empty", "targetHandle": "handle_user_message"},
        {"id": "e4", "source": "client-node", "target": "llm-node", "sourceHandle": "handle-client-provider", "targetHandle": "handle-client-provider"},
        {"id": "e5", "source": "send-msg", "target": "end-node", "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
        {"id": "e6", "source": "llm-node", "target": "end-node", "sourceHandle": "handle_generated_content", "targetHandle": "handle_flow_input"},
    ],
    "nodes": [
        {"id": "user-input", "type": "user_input"},
        {"id": "cond", "type": "conditional", "data": {"condition": "{{ 'not_empty' if value|default('')|trim else 'empty' }}"}},
        {"id": "send-msg", "type": "send_message", "data": {"message": "", "json_extras": "Input is empty"}},
        {"id": "client-node", "type": "client", "data": {"engine": "openai", "model": "gpt-4o-mini"}},
        {"id": "llm-node", "type": "llm", "data": {"stream": False, "max_tokens": 5}},
        {"id": "end-node", "type": "end"},
    ],
}


def _streamed_text(events):
    text = ""
    for event in events:
        content = event.get("content") if event.get("type") == "content" else None
        if getattr(content, "choices", None) and content.choices[0].delta.content:
            text += content.choices[0].delta.content
    return text


@pytest.mark.parametrize("message", ["", "   "], ids=["empty", "whitespace"])
async def test_blank_input_routes_to_the_canned_reply_without_calling_the_provider(mock_magic_llm, message):
    hook = RecordingHook()
    graph = build(copy.deepcopy(FLOW), message=message, load_chat=None)

    events = [event async for event in run_agent(graph, hooks=RuntimeConfig(graph_hooks=[hook]))]

    assert debug_errors(events) == []
    assert graph.nodes["cond"].selected_handle == "empty"
    assert "Input is empty" in _streamed_text(events)
    # The LLM's client edge still arrives (fan-in runs with what arrived), but
    # without a message NodeLLM returns before any provider call.
    assert mock_magic_llm.llm.async_generate.await_count == 0


async def test_question_routes_to_the_llm_and_bypasses_the_canned_reply(mock_magic_llm):
    hook = RecordingHook()
    graph = build(copy.deepcopy(FLOW), message="What is 2+2?", load_chat=None)

    events = [event async for event in run_agent(graph, hooks=RuntimeConfig(graph_hooks=[hook]))]

    assert debug_errors(events) == []
    assert graph.nodes["cond"].selected_handle == "not_empty"
    assert mock_magic_llm.llm.async_generate.await_count == 1
    assert graph.nodes["llm-node"].generated == "mocked LLM response"
    bypassed = {inv["context"].node_id for inv in hook.get_invocations("on_node_bypass")}
    assert "send-msg" in bypassed
    assert "Input is empty" not in _streamed_text(events)
