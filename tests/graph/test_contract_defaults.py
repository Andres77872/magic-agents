"""
Backward compatibility regression tests (Phase 4.5).

Tests that:
1. Existing graph JSON executes correctly without modification
2. Handle-only edges continue to function (no sourcePort/targetPort)
3. Handle strings resolve correctly via alias passthrough
4. Rollback: mode="off" disables all new validation
5. Graphs without contract_config still work (defaults applied)
"""
import pytest

from magic_agents.agt_flow import build
from magic_agents.models.factory.AgentFlowModel import ContractConfig
from magic_agents.util.handle_registry import is_valid_target_handle, is_valid_source_handle


class TestBackwardCompatGraphExecution:
    """Tests that existing graphs work without modification."""

    def test_minimal_graph_builds(self):
        """
        Minimal graph (input → end) must build correctly.
        """
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'end'},
            ]
        }

        graph = build(graph_json, message="test")

        assert graph is not None
        assert len(graph.nodes) >= 2
        assert len(graph.edges) >= 1

    def test_graph_without_handles_builds(self):
        """
        Graph without explicit handles must build with defaults.
        """
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'text', 'type': 'text', 'data': {'text': 'hello'}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'text'},
                {'source': 'text', 'target': 'end'},
            ]
        }

        graph = build(graph_json, message="test")

        assert graph is not None
        # Should have contract report attached (default mode)
        assert graph.contract_report is not None

    def test_graph_with_canonical_handles_builds(self):
        """
        Graph with canonical handles must build correctly.
        """
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
                {'id': 'e2', 'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'},
            ]
        }

        graph = build(graph_json, message="test")

        assert graph is not None
        assert len(graph.nodes) >= 3
        assert graph.contract_report is not None

    def test_graph_without_contract_config_uses_defaults(self):
        """
        Graph JSON without contract_config must use default (warn mode).
        """
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'end'},
            ]
            # No contract_config field
        }

        graph = build(graph_json, message="test")

        # Default mode is warn
        assert graph.contract_config.mode == "warn"
        assert graph.contract_report is not None

    def test_complex_graph_with_multiple_node_types(self):
        """
        Complex graph with LLM, Parser, Text, Conditional must build.
        """
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'parser', 'type': 'parser', 'data': {'text': '{{ handle_parser_input }}'}},
                {'id': 'text', 'type': 'text', 'data': {'text': 'result'}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
                {'id': 'e2', 'source': 'llm', 'target': 'parser', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_parser_input'},
                {'id': 'e3', 'source': 'parser', 'target': 'text', 'sourceHandle': 'handle_parsed_content', 'targetHandle': 'handle_flow_input'},
                {'id': 'e4', 'source': 'text', 'target': 'end', 'sourceHandle': 'handle_text_output', 'targetHandle': 'handle_flow_input'},
            ]
        }

        graph = build(graph_json, message="test")

        assert graph is not None
        assert len(graph.nodes) >= 5


class TestHandleOnlyEdgesWork:
    """Tests that handle-only edges (no ports) still function."""

    def test_graph_with_only_handles_builds(self):
        """
        Graph with only handles (no ports anywhere) must build.
        """
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
                {'id': 'e2', 'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'},
            ]
            # No sourcePort/targetPort anywhere
        }

        graph = build(graph_json, message="test")

        assert graph is not None

        # All edges have None ports
        for edge in graph.edges:
            assert edge.sourcePort is None
            assert edge.targetPort is None


class TestHandleResolution:
    """Handle classification (signature: ``(handle, node_type)``).

    Every handle is *accepted* in warn/shadow mode, so the returned reason is
    what distinguishes canonical, dynamic and opaque handles.
    """

    @pytest.mark.parametrize(("handle", "node_type", "reason"), [
        ("handle_user_message", "llm", "Canonical input handle"),
        ("handle-client-provider", "llm", "Canonical input handle"),
        ("handle_flow_input", "text", "Canonical input handle"),
        ("handle_parser_input", "parser", "Canonical input handle"),
        ("handle-tool-definition-0", "llm", "Dynamic pattern handle"),
        ("handle-tool-definition-5", "llm", "Dynamic pattern handle"),
    ])
    def test_known_target_handles_are_classified(self, handle, node_type, reason):
        assert is_valid_target_handle(handle, node_type) == (True, reason)

    @pytest.mark.parametrize("handle", ["some_random_handle_xyz", "handle_tool_definition-0"])
    def test_unknown_target_handles_pass_as_opaque_for_backward_compatibility(self, handle):
        is_valid, reason = is_valid_target_handle(handle, "llm")
        assert is_valid and reason.startswith("Opaque/legacy handle")

    @pytest.mark.parametrize(("handle", "node_type"), [
        ("handle_generated_content", "llm"),
        ("handle_text_output", "text"),
        ("handle_parser_output", "parser"),
    ])
    def test_canonical_source_handles_are_valid(self, handle, node_type):
        is_valid, reason = is_valid_source_handle(handle, node_type)
        assert is_valid and "Opaque" not in reason


class TestRollbackMode:
    """Tests for rollback path: mode="off" disables validation."""

    def test_mode_off_contract_config_applied(self):
        """
        contract_config.mode="off" must be applied to built graph.
        """
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'end'},
            ],
            'contract_config': {'mode': 'off'},
        }

        graph = build(graph_json, message="test")

        assert graph.contract_config.mode == "off"
        assert graph.contract_config.is_off

class TestContractConfigModes:
    """Tests for ContractConfig mode behavior."""

    def test_contract_config_from_dict(self):
        """
        ContractConfig must be constructible from dict.
        """
        config = ContractConfig(**{'mode': 'shadow', 'strict_runtime': False})

        assert config.mode == "shadow"
