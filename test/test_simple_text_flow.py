"""
Slice 13 — Simple text flow integration tests (no API keys).

End-to-end flow: user_input -> text -> end.
Proves the full pipeline works without any LLM.
"""
import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build


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
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "text_node", "target": "end",
                 "sourceHandle": "handle_text_output", "targetHandle": "h1"},
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
    async def test_simple_text_flow_produces_debug_summary(self):
        """debug_summary event is yielded when debug=True."""
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
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "text_node", "target": "end",
                 "sourceHandle": "handle_text_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})

        assert debug_summary is not None
        assert "nodes" in debug_summary
        node_ids = {n["node_id"] for n in debug_summary["nodes"]}
        assert "text_node" in node_ids
        assert "input" in node_ids

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
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_input"},
                {"id": "e2", "source": "text_node", "target": "send",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="test")
        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            if item.get("type") == "content" and hasattr(item.get("content"), "choices"):
                choices = item["content"].choices
                if choices and choices[0].delta.content:
                    content_output.append(choices[0].delta.content)

        content_str = "".join(content_output)
        assert "EXTRA" in content_str
        executed = {n["node_id"] for n in debug_summary["nodes"] if n.get("was_executed")}
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
                 "sourceHandle": "handle_message_output", "targetHandle": "h1"},
            ],
        }

        graph = build(agt, message="hello")
        content_output = []
        debug_summary = None
        async for item in run_agent(graph):
            if item.get("type") == "debug_summary":
                debug_summary = item.get("content", {})
            if item.get("type") == "content" and hasattr(item.get("content"), "choices"):
                choices = item["content"].choices
                if choices and choices[0].delta.content:
                    content_output.append(choices[0].delta.content)

        content_str = "".join(content_output)
        assert "PARSED" in content_str
        executed = {n["node_id"] for n in debug_summary["nodes"] if n.get("was_executed")}
        assert "parser" in executed
        assert "send" in executed
