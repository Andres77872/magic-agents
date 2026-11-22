"""
Task 4.2 — NodeCodex integration tests with build() + run_agent().

Covers 6 integration test scenarios per design.md:
1. Full flow with trigger match
2. Full flow with no trigger match (passthrough)
3. Empty codex_entries passthrough
4. Multi-consumer routing (Codex → two SendMessage nodes)
5. Codex → END directly (no intermediate consumer)
6. Graph validation for valid codex graph

Uses send_message nodes as downstream consumers to avoid LLM API key
dependency (no mock_magic_llm needed).

Pattern matches test_simple_text_flow.py:
- @pytest.mark.asyncio for async tests
- build(agt, message=...) + run_agent(graph)
- Check node.outputs after execution for transform verification

NOTE: debug_summary is NOT yielded by run_agent() — it's added by the API
layer (debug_sse.py). Tests verify execution via node outputs instead.
"""

import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build, validate_graph

CODEX_CONTENT_MARKER = "<codex_content>"
TRIGGER_MESSAGE = "hello world"
NO_MATCH_MESSAGE = "goodbye world"
CODE_ENTRIES_SINGLE = [
    {"triggers": ["hello"], "content": "Greeting info"},
]


def _make_codex_graph(
    codex_entries: list = None,
    extra_consumers: int = 0,
    codex_to_end_direct: bool = False,
    debug: bool = True,
) -> dict:
    """Build a graph dict with UserInput → Codex → consumer(s) → END.

    Args:
        codex_entries: List of codex entries for the codex node.
        extra_consumers: Number of additional send_message consumers.
        codex_to_end_direct: If True, connect Codex directly to END (no consumer).
        debug: Enable debug in the graph.

    Returns:
        Graph dict ready for build().
    """
    if codex_entries is None:
        codex_entries = []

    nodes = [
        {"id": "input", "type": "user_input"},
        {"id": "codex", "type": "codex", "data": {"codex_entries": codex_entries}},
    ]
    edges = [
        {
            "id": "e1", "source": "input", "target": "codex",
            "sourceHandle": "handle_user_message", "targetHandle": "handle_codex_input",
        },
    ]

    if codex_to_end_direct:
        # Codex → END directly (no intermediate consumer)
        nodes.append({"id": "end", "type": "end"})
        edges.append({
            "id": "e_end", "source": "codex", "target": "end",
            "sourceHandle": "handle_user_message",
        })
    else:
        # Primary consumer: send_message
        nodes.append({
            "id": "send_main", "type": "send_message",
            "data": {"message": "", "json_extras": ""},
        })
        edges.append({
            "id": "e2", "source": "codex", "target": "send_main",
            "sourceHandle": "handle_user_message",
            "targetHandle": "handle_send_extra",
        })

        # Extra consumers for multi-consumer test
        for i in range(extra_consumers):
            send_id = f"send_extra_{i}"
            nodes.append({
                "id": send_id, "type": "send_message",
                "data": {"message": "", "json_extras": ""},
            })
            edges.append({
                "id": f"e_extra_{i}", "source": "codex", "target": send_id,
                "sourceHandle": "handle_user_message",
                "targetHandle": "handle_send_extra",
            })

        # END node
        nodes.append({"id": "end", "type": "end"})

        # Connect consumers to END
        edges.append({
            "id": "e3", "source": "send_main", "target": "end",
            "sourceHandle": "handle_message_output",
        })
        for i in range(extra_consumers):
            send_id = f"send_extra_{i}"
            edges.append({
                "id": f"e_end_extra_{i}", "source": send_id, "target": "end",
                "sourceHandle": "handle_message_output",
            })

    return {"type": "graph", "debug": debug, "nodes": nodes, "edges": edges}


def _get_codex_output(graph) -> str:
    """Extract the codex node's output content after execution."""
    codex_node = graph.nodes.get("codex")
    assert codex_node is not None, "Codex node not found in graph"
    assert "handle_user_message" in codex_node.outputs, (
        f"Codex node has no output on handle_user_message. "
        f"Available outputs: {list(codex_node.outputs.keys())}"
    )
    output = codex_node.outputs["handle_user_message"]
    assert "content" in output, f"Codex output has no 'content' field: {output}"
    return str(output["content"])


def _get_codex_internal_state(graph) -> dict:
    """Get the codex node's internal state after execution."""
    codex_node = graph.nodes.get("codex")
    assert codex_node is not None, "Codex node not found in graph"
    return codex_node._capture_internal_state()


class TestNodeCodexIntegration:
    """Integration tests for NodeCodex with full graph execution."""

    # ═════════════════════════════════════════════════════════════════════
    # Test 1: Full flow with trigger match
    # ═════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_full_flow_trigger_match(self):
        """UserInput → Codex → SendMessage → END with matching trigger.

        Given codex entry triggers=["hello"], content="Greeting info",
        and user message "hello world", the codex node should prepend
        the content wrapped in <codex_content> tags.
        """
        agt = _make_codex_graph(codex_entries=CODE_ENTRIES_SINGLE)

        graph = build(agt, message=TRIGGER_MESSAGE)
        async for _ in run_agent(graph):
            pass  # Consume all events

        # Transform verification: codex output contains the codex block
        codex_output = _get_codex_output(graph)
        assert CODEX_CONTENT_MARKER in codex_output, (
            f"Expected <codex_content> in output, got: {codex_output!r}"
        )
        assert "Greeting info" in codex_output, (
            f"Expected 'Greeting info' in output, got: {codex_output!r}"
        )
        assert TRIGGER_MESSAGE in codex_output, (
            f"Expected original message in output, got: {codex_output!r}"
        )
        assert codex_output.startswith(
            "<codex_content>Greeting info</codex_content>"
        ), f"Output does not start with expected format: {codex_output!r}"

        # Internal state verification
        state = _get_codex_internal_state(graph)
        assert state["prepended_count"] == 1, (
            f"Expected 1 prepended, got: {state['prepended_count']}"
        )
        assert "hello" in state["matched_triggers"], (
            f"Expected 'hello' in matched triggers: {state['matched_triggers']}"
        )
        assert state["message_length"] == len(TRIGGER_MESSAGE)

    # ═════════════════════════════════════════════════════════════════════
    # Test 2: Full flow with no trigger match (passthrough)
    # ═════════════════════════════════════════════════════════════════════

    # ═════════════════════════════════════════════════════════════════════
    # Test 3: Empty codex_entries passthrough
    # ═════════════════════════════════════════════════════════════════════

    # ═════════════════════════════════════════════════════════════════════
    # Test 4: Multi-consumer routing
    # ═════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_multi_consumer_routing(self):
        """Codex → TWO SendMessage nodes — both receive the same transformed message.

        This verifies that propagate_outputs() correctly routes the codex
        output to all outgoing edges from Codex.
        """
        agt = _make_codex_graph(
            codex_entries=CODE_ENTRIES_SINGLE,
            extra_consumers=1,
        )

        graph = build(agt, message=TRIGGER_MESSAGE)
        async for _ in run_agent(graph):
            pass

        # Codex node output is the transformed message
        codex_output = _get_codex_output(graph)
        assert CODEX_CONTENT_MARKER in codex_output, (
            f"Expected <codex_content> in codex output: {codex_output!r}"
        )

        # Both consumers executed (outputs exist)
        send_main = graph.nodes.get("send_main")
        assert send_main is not None
        assert "handle_message_output" in send_main.outputs, (
            f"send_main has no output: {list(send_main.outputs.keys())}"
        )

        send_extra = graph.nodes.get("send_extra_0")
        assert send_extra is not None
        assert "handle_message_output" in send_extra.outputs, (
            f"send_extra_0 has no output: {list(send_extra.outputs.keys())}"
        )
        # Both consumers received the same transformed message, not just "some" input.
        received = [list(node.inputs.values()) for node in (send_main, send_extra)]
        assert received[0] == received[1] and codex_output in received[0]

        # Codex internal state confirms the match
        state = _get_codex_internal_state(graph)
        assert state["prepended_count"] == 1, (
            f"Expected 1 prepended: {state['prepended_count']}"
        )
        assert "hello" in state["matched_triggers"]

    # ═════════════════════════════════════════════════════════════════════
    # Test 5: Codex → END directly
    # ═════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_codex_to_end_direct(self):
        """Codex → END directly (no intermediate consumer), matching trigger.

        The codex transform should work independently of intermediate
        consumers. The message flows from UserInput → Codex → END,
        and the codex node should still prepend the matched content.
        """
        agt = _make_codex_graph(
            codex_entries=CODE_ENTRIES_SINGLE,
            codex_to_end_direct=True,
        )

        graph = build(agt, message=TRIGGER_MESSAGE)
        async for _ in run_agent(graph):
            pass

        # Codex output is correct even without an intermediate consumer
        codex_output = _get_codex_output(graph)
        assert CODEX_CONTENT_MARKER in codex_output, (
            f"Expected <codex_content> in codex output: {codex_output!r}"
        )
        assert codex_output.startswith(
            "<codex_content>Greeting info</codex_content>"
        ), f"Output format incorrect: {codex_output!r}"

        # Both codex and end executed
        state = _get_codex_internal_state(graph)
        assert state["prepended_count"] == 1

        end_node = graph.nodes.get("end")
        assert end_node is not None
        assert "handle_end_output" in end_node.outputs, (
            f"End node has no output: {list(end_node.outputs.keys())}"
        )

    # ═════════════════════════════════════════════════════════════════════
    # Test 6: Graph validation
    # ═════════════════════════════════════════════════════════════════════

    def test_graph_validation_passes(self):
        """validate_graph() passes for a valid codex graph."""
        agt = _make_codex_graph(codex_entries=CODE_ENTRIES_SINGLE)
        result = validate_graph(agt["nodes"], agt["edges"])

        assert result["valid"] is True, (
            f"Expected graph to pass, got errors: {result.get('errors', [])}"
        )
        assert result["errors"] == [], f"Unexpected errors: {result['errors']}"

    def test_graph_validation_multi_consumer_passes(self):
        """validate_graph() passes for a multi-consumer codex graph."""
        agt = _make_codex_graph(
            codex_entries=CODE_ENTRIES_SINGLE,
            extra_consumers=2,
        )
        result = validate_graph(agt["nodes"], agt["edges"])

        assert result["valid"] is True, (
            f"Expected pass (multi-consumer), got: {result.get('errors')}"
        )
        assert result["errors"] == []
