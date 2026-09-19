"""
Slice 13 — Simple text flow integration tests (no API keys).

End-to-end flow: user_input -> text -> end.
Proves the full pipeline works without any LLM.
"""
import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.debug.events import DebugEventType


async def collect_run(graph):
    """Collect direct runtime output and typed observer events."""
    results = []
    debug_events = []

    async def capture_debug(event):
        debug_events.append(event)

    async for item in run_agent(graph, debug_callback=capture_debug):
        results.append(item)
    return results, debug_events


def completed_nodes(debug_events) -> set[str]:
    return {
        event.node_id
        for event in debug_events
        if event.event_type == DebugEventType.NODE_END
    }


class TestSimpleTextFlow:
    """Tests for the simplest possible flow end-to-end."""

    @pytest.mark.asyncio
    async def test_simple_text_flow_produces_output(self):
        """Content event is yielded with text node output."""
        agt = {
            "type": "graph",
            "debug": False,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "text_node", "type": "text", "data": {"text": "Hello from text node"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "text_node",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "e2", "source": "text_node", "target": "end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        results = []
        async for item in run_agent(graph):
            results.append(item)

        # Should have at least some results
        assert len(results) > 0

        # The text node output should be propagated (stored in node outputs)
        text_node = graph.nodes.get("text_node")
        assert text_node is not None
        assert "handle_text_output" in text_node.outputs
        output = text_node.outputs["handle_text_output"]
        assert output["content"] == "Hello from text node"

    @pytest.mark.asyncio
    async def test_simple_text_flow_emits_debug_lifecycle(self):
        """debug_callback receives node lifecycle and final graph events."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "text_node", "type": "text", "data": {"text": "Debug test"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "text_node",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "e2", "source": "text_node", "target": "end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        assert not graph._validation_errors
        _, debug_events = await collect_run(graph)

        graph_end = [
            event for event in debug_events
            if event.event_type == DebugEventType.GRAPH_END
        ]
        assert len(graph_end) == 1
        assert graph_end[0].payload["failed_nodes"] == 0
        assert {"input", "text_node", "end"} <= completed_nodes(debug_events)

    @pytest.mark.asyncio
    async def test_text_to_send_message_flow(self):
        """Text node -> send_message -> end produces streamed content."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "text_node", "type": "text", "data": {"text": "Message content"}},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "EXTRA"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "text_node",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "e2", "source": "text_node", "target": "send",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="test")
        assert not graph._validation_errors
        content_output = []
        results, debug_events = await collect_run(graph)
        for item in results:
            if item.get("type") == "content" and hasattr(item.get("content"), "choices"):
                choices = item["content"].choices
                if choices and choices[0].delta.content:
                    content_output.append(choices[0].delta.content)

        content_str = "".join(content_output)
        assert "EXTRA" in content_str
        executed = completed_nodes(debug_events)
        assert "send" in executed
        assert "text_node" in executed

    @pytest.mark.asyncio
    async def test_parser_text_flow_no_api(self):
        """Parser node with static text executes without any API calls."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "parser", "type": "parser", "data": {
                    "text": "Processed: {{ handle_parser_input }}"
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "PARSED"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "parser",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_parser_input"},
                {"id": "e2", "source": "parser", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }

        graph = build(agt, message="hello")
        assert not graph._validation_errors
        content_output = []
        results, debug_events = await collect_run(graph)
        for item in results:
            if item.get("type") == "content" and hasattr(item.get("content"), "choices"):
                choices = item["content"].choices
                if choices and choices[0].delta.content:
                    content_output.append(choices[0].delta.content)

        content_str = "".join(content_output)
        assert "PARSED" in content_str
        executed = completed_nodes(debug_events)
        assert "parser" in executed
        assert "send" in executed
