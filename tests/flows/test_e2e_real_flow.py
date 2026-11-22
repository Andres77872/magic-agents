"""
E2E real-flow tests using build() + run_agent() path.

These tests exercise the ACTUAL runtime execution path, not mocked orchestration.

Test categories:
1. Credential-free execution with real built-in nodes
2. Live execution (requires real keys) - skips honestly if .env.test has placeholders
3. Edge.id tracking through real execution flow

All tests use actual example graph patterns from the repository.
"""
import pytest

from magic_agents import run_agent
from magic_agents.agt_flow import build
from magic_agents.execution.event_dispatcher import GraphEventDispatcher
from magic_agents.node_system import NodeLoop


# ─── Helper functions ───────────────────────────────────────────────────────

def extract_streamed_content(item):
    """Extract streamed content from send_message or LLM output."""
    if not isinstance(item, dict):
        return ""
    if item.get("type") != "content":
        return ""
    content = item.get("content")
    if content is None:
        return ""
    if hasattr(content, "choices") and content.choices:
        delta = content.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            return delta.content
    return ""


async def collect_run_agent_results(graph):
    """Collect all results from run_agent(graph) async generator."""
    results = []
    async for item in run_agent(graph):
        results.append(item)
    return results


# ─── Credential-free E2E Tests ────────────────────────────────────────────────

class TestCredentialFreeE2EExecution:
    """Real build -> run_agent paths with no provider dependency."""
    
    @pytest.mark.asyncio
    async def test_simple_flow_routes_real_node_values(self):
        """UserInput -> Text -> SendMessage -> END executes end to end."""
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "text", "type": "text", "data": {"text": "Hello, world!"}},
                {"id": "send", "type": "send_message", "data": {"message": "Delivered"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "e2", "source": "text", "target": "send",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_send_extra"},
                {"id": "e3", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test message")
        
        results = await collect_run_agent_results(graph)

        assert graph._validation_errors in (None, [])
        assert graph.nodes["text"].inputs == {"handle_flow_input": "test message"}
        assert graph.nodes["send"].inputs == {"handle_send_extra": "Hello, world!"}
        assert graph.nodes["end"].response is not None
        messages = [extract_streamed_content(result) for result in results]
        assert [message for message in messages if message] == ["Delivered"]
        delivered = next(
            result["content"]
            for result in results
            if extract_streamed_content(result) == "Delivered"
        )
        assert delivered.extras == {"text": "Hello, world!"}
    
    @pytest.mark.asyncio
    async def test_loop_flow_executes_each_real_iteration_and_aggregation(self):
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "list_text", "type": "text", "data": {"text": '["a", "b", "c"]'}},
                {"id": "loop", "type": "loop", "data": {}},
                {"id": "transform", "type": "parser", "data": {
                    "text": "item: {{ handle_parser_input }}"
                }},
                {"id": "format", "type": "parser", "data": {
                    "text": "Results: {{ handle_parser_input | join(', ') }}"
                }},
                {"id": "send", "type": "send_message", "data": {"message": "", "json_extras": "DONE"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "list_text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "e2", "source": "list_text", "target": "loop",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_list"},
                {"id": "e3", "source": "loop", "target": "transform",
                 "sourceHandle": "handle_item", "targetHandle": "handle_parser_input"},
                {"id": "e4", "source": "transform", "target": "loop",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_loop"},
                {"id": "e5", "source": "loop", "target": "format",
                 "sourceHandle": "handle_end", "targetHandle": "handle_parser_input"},
                {"id": "e6", "source": "format", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e7", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test")
        assert isinstance(graph.nodes["loop"], NodeLoop)
        
        results = []
        async for item in run_agent(graph):
            results.append(item)

        assert graph._validation_errors in (None, [])
        assert [result["content"]["current"] for result in results if result.get("type") == "loop_progress"] == [0, 1, 2]
        assert graph.nodes["loop"].outputs["handle_end"]["content"] == [
            "item: a",
            "item: b",
            "item: c",
        ]
        assert graph.nodes["format"].inputs == {
            "handle_parser_input": ["item: a", "item: b", "item: c"]
        }
        assert graph.nodes["send"].inputs["handle_send_extra"] == (
            "Results: item: a, item: b, item: c"
        )
        assert [
            extract_streamed_content(result)
            for result in results
            if extract_streamed_content(result)
        ] == ["DONE"]
        assert graph.nodes["end"].response is not None
    
    @pytest.mark.asyncio
    async def test_edge_id_preserved_through_execution(self):
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "text", "type": "text", "data": {"text": "test"}},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "custom_edge_id_alpha", "source": "input", "target": "text",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "custom_edge_id_beta", "source": "text", "target": "end",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test")
        
        edge_ids = [edge.id for edge in graph.edges]
        assert [edge_id for edge_id in edge_ids if edge_id.startswith("custom_")] == [
            "custom_edge_id_alpha",
            "custom_edge_id_beta",
        ]
        dispatcher = GraphEventDispatcher(nodes=graph.nodes, edges=graph.edges)
        assert dispatcher.get_tracker("text").expected_edges == {"custom_edge_id_alpha"}
        assert dispatcher.get_tracker("end").expected_edges == {"custom_edge_id_beta"}

        await collect_run_agent_results(graph)

        assert graph.nodes["text"].inputs == {"handle_flow_input": "test"}
        assert graph.nodes["end"].response is not None
    
    @pytest.mark.asyncio
    async def test_contract_report_is_not_mutated_by_execution(self):
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "end",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test")
        
        assert graph.contract_report is not None
        before = graph.contract_report.to_normalized_dict()

        await collect_run_agent_results(graph)

        assert graph.contract_report.to_normalized_dict() == before
        assert graph.nodes["end"].response is not None


# ─── Live E2E Tests (Require Real Keys, Skip Honestly) ───────────────────────────

# ─── Edge.id Tracking Through Real Flow ─────────────────────────────────────────

class TestEdgeIdTrackingThroughRealFlow:
    """
    Tests proving edge.id tracking works through the ACTUAL execution path.
    """

    @pytest.mark.asyncio
    async def test_fan_in_execution_does_not_collapse(self):
        """
        Fan-in to same handle tracked correctly (not collapsed) through execution.
        
        This is the CRITICAL bug fix proof - multiple edges same handle
        produce multiple tracker entries through ACTUAL run_agent flow.
        """
        # One valid start node fans out to two independent branches, which
        # then converge on distinct input handles of the same parser.
        agt = {
            "type": "graph",
            "debug": True,
            "nodes": [
                {"id": "input", "type": "user_input"},
                {"id": "left", "type": "text", "data": {"text": "left"}},
                {"id": "right", "type": "text", "data": {"text": "right"}},
                {"id": "parser", "type": "parser", "data": {
                    "text": "Combined: {{ handle_parser_input_0 }} and {{ handle_parser_input_1 }}"
                }},
                {"id": "send", "type": "send_message", "data": {
                    "message": "fan-in-complete"
                }},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"id": "trigger_left", "source": "input", "target": "left",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "trigger_right", "source": "input", "target": "right",
                 "sourceHandle": "handle_user_message", "targetHandle": "handle_flow_input"},
                {"id": "fan_in_edge_1", "source": "left", "target": "parser",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_parser_input_0"},
                {"id": "fan_in_edge_2", "source": "right", "target": "parser",
                 "sourceHandle": "handle_text_output", "targetHandle": "handle_parser_input_1"},
                {"id": "e3", "source": "parser", "target": "send",
                 "sourceHandle": "handle_parser_output", "targetHandle": "handle_send_extra"},
                {"id": "e4", "source": "send", "target": "end",
                 "sourceHandle": "handle_message_output", "targetHandle": "handle_flow_input"},
            ],
        }
        
        graph = build(agt, message="test")
        
        # Build dispatcher
        dispatcher = GraphEventDispatcher(nodes=graph.nodes, edges=graph.edges)
        
        # Parser tracker must have BOTH edges (not collapsed)
        tracker = dispatcher.get_tracker('parser')
        assert len(tracker.expected_edges) == 2
        assert 'fan_in_edge_1' in tracker.expected_edges
        assert 'fan_in_edge_2' in tracker.expected_edges
        
        # Execute the actual build -> run_agent path (no LLM mock is needed).
        results = await collect_run_agent_results(graph)

        # Both fan-in deliveries must be preserved and rendered by the parser.
        assert graph.nodes["parser"].inputs == {
            "handle_parser_input_0": "left",
            "handle_parser_input_1": "right",
        }
        assert graph.nodes["send"].inputs["handle_send_extra"] == "Combined: left and right"

        user_messages = [
            event["content"]
            for event in results
            if event.get("type") == "content"
            and getattr(event.get("content"), "choices", None)
            and event["content"].choices[0].delta.content == "fan-in-complete"
        ]
        assert len(user_messages) == 1
        assert user_messages[0].extras == {"text": "Combined: left and right"}


# ─── Contract Mode E2E Tests ─────────────────────────────────────────────────────

class TestContractModeE2E:
    """Contract modes drive both diagnostics and real graph execution."""

    @staticmethod
    def build_graph(mode):
        return build(
            {
                "type": "graph",
                "debug": True,
                "contract_config": {"mode": mode},
                "nodes": [
                    {"id": "input", "type": "user_input"},
                    {
                        "id": "parser",
                        "type": "parser",
                        "data": {"text": "mode={{ legacy_input }}"},
                    },
                    {"id": "end", "type": "end"},
                ],
                "edges": [
                    {
                        "id": "e1",
                        "source": "input",
                        "target": "parser",
                        "sourceHandle": "handle_user_message",
                        # Deliberately opaque so the mode has a diagnostic to handle.
                        "targetHandle": "legacy_input",
                    },
                    {
                        "id": "e2",
                        "source": "parser",
                        "target": "end",
                        "sourceHandle": "handle_parser_output",
                        "targetHandle": "handle_flow_input",
                    },
                ],
            },
            message="test",
        )

    @pytest.mark.asyncio
    async def test_warn_mode_attaches_and_surfaces_diagnostic_then_executes(self, caplog):
        graph = self.build_graph("warn")

        assert graph.contract_report.mode == "warn"
        assert [d["code"] for d in graph.contract_report.diagnostics] == [
            "PORT_TARGET_HANDLE_LEGACY"
        ]
        assert any("legacy_input" in record.message for record in caplog.records)

        await collect_run_agent_results(graph)

        assert graph.nodes["parser"].response == "mode=test"
        assert graph.nodes["end"].response is not None
    
    @pytest.mark.asyncio
    async def test_shadow_mode_attaches_without_logging_then_executes(self, caplog):
        graph = self.build_graph("shadow")

        assert graph.contract_report.mode == "shadow"
        assert [d["code"] for d in graph.contract_report.diagnostics] == [
            "PORT_TARGET_HANDLE_LEGACY"
        ]
        assert not any("legacy_input" in record.message for record in caplog.records)

        await collect_run_agent_results(graph)

        assert graph.nodes["parser"].response == "mode=test"
        assert graph.nodes["end"].response is not None
    
    @pytest.mark.asyncio
    async def test_off_mode_skips_diagnostics_then_executes(self, caplog):
        graph = self.build_graph("off")

        assert graph.contract_report.mode == "off"
        assert graph.contract_report.diagnostics == []
        assert not any("legacy_input" in record.message for record in caplog.records)

        await collect_run_agent_results(graph)

        assert graph.nodes["parser"].response == "mode=test"
        assert graph.nodes["end"].response is not None
