"""Authored loop graphs through ``build`` + ``run_agent`` (no provider)."""

import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build


def _mixed_types_graph():
    return {
        "type": "chat",
        "edges": [
            {"id": "mixed-to-loop", "source": "mixed-data", "target": "loop-node",
             "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
            {"id": "loop-to-parser", "source": "loop-node", "target": "type-analyzer",
             "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
            {"id": "parser-to-loop", "source": "type-analyzer", "target": "loop-node",
             "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
            {"id": "loop-to-send", "source": "loop-node", "target": "send-results",
             "sourceHandle": "handle_end", "targetHandle": "handle_send_extra"},
            {"id": "send-to-end", "source": "send-results", "target": "end-node",
             "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
        ],
        "nodes": [
            {"id": "user-input", "type": "user_input"},
            {"id": "mixed-data", "type": "text", "data": {"text": '[42, "hello", true, 3.14, null, {"k": 1}]'}},
            {"id": "loop-node", "type": "loop", "data": {}},
            {"id": "type-analyzer", "type": "parser",
             "data": {"text": "{{ handle_parser_input | tojson }}"}},
            {"id": "send-results", "type": "send_message", "data": {"json_extras": "done"}},
            {"id": "end-node", "type": "end"},
        ],
    }


async def test_loop_items_keep_their_json_types_and_aggregate_in_order():
    graph = build(_mixed_types_graph(), message="", load_chat=None)

    assert graph._validation_errors is None
    assert graph.get_contract_errors() == []
    assert graph.get_contract_warnings() == []

    [event async for event in run_agent(graph)]

    # tojson of each item proves the parsed JSON type reached the iteration body.
    assert graph.nodes["send-results"].inputs["handle_send_extra"] == [
        "42", '"hello"', "true", "3.14", "null", '{"k": 1}',
    ]


@pytest.mark.parametrize("payload", ["[]", "not json", '{"a": 1}'])
async def test_loop_without_a_list_never_runs_its_body(payload):
    graph_def = _mixed_types_graph()
    graph_def["nodes"][1]["data"]["text"] = payload
    graph = build(graph_def, message="", load_chat=None)

    [event async for event in run_agent(graph)]

    assert "handle_parser_input" not in graph.nodes["type-analyzer"].inputs
