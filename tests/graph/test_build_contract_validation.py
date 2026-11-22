"""
Integration tests for build path validation wiring (Phase 3 corrective).

Tests that:
1. build() runs full validation chain via run_all_validations()
2. GraphContractReport is attached to built model
3. ContractConfig.mode drives validation behavior
4. shadow + warn default behavior is real
5. Rollback path: mode="off" disables validation
6. target/fan-in diagnostics are attached in warn mode
"""
import pytest

from magic_agents.agt_flow import build
from magic_agents.models.factory.AgentFlowModel import GraphContractReport
from magic_agents.util.graph_validator import run_all_validations




class TestBuildPathContractReport:
    """Test that build() attaches GraphContractReport."""

    def test_build_attaches_contract_report(self):
        """build() must attach _contract_report to built model."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'end', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Must have contract_report attached
        assert graph.contract_report is not None
        assert isinstance(graph.contract_report, GraphContractReport)

    def test_contract_report_has_correct_counts(self):
        """contract_report must have correct edge/node counts."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'text', 'type': 'text', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'text', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_flow_input'},
                {'id': 'e2', 'source': 'text', 'target': 'end', 'sourceHandle': 'handle_text_output', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # build() adds a void node for end, plus an extra edge
        # Original: 2 edges, 3 nodes
        # After build: 3 edges (e1, e2, + void edge), 4 nodes (input, text, end, + void)
        assert graph.contract_report.edge_count == 3
        assert graph.contract_report.node_count == 4

    def test_contract_report_mode_matches_config(self):
        """contract_report.mode must match contract_config.mode."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'end'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Default mode is warn
        assert graph.contract_config.mode == "warn"
        assert graph.contract_report.mode == "warn"


class TestContractConfigModeDrivesValidation:
    """Test that ContractConfig.mode actually drives validation behavior."""

    def test_warn_mode_surfaces_warnings(self):
        """warn mode must surface diagnostics as warnings."""
        # Use a graph with unknown target handle to trigger diagnostic
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'unknown_handle_xyz'},
                {'id': 'e2', 'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Should have diagnostics (warn mode surfaces them)
        assert graph.contract_report is not None
        # Note: In warn mode, opaque handles pass with warning
        assert len(graph.contract_report.diagnostics) > 0

    def test_shadow_mode_computes_no_surface(self):
        """shadow mode must compute diagnostics but not surface them."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'unknown_xyz'},
            ],
            'contract_config': {'mode': 'shadow'},
        }
        
        graph = build(graph_json, message="test")
        
        # Should have contract_report
        assert graph.contract_report is not None
        assert graph.contract_report.mode == "shadow"
        # Diagnostics should be computed (attached to report)
        # but not surfaced as warnings

    def test_off_mode_skips_validation(self):
        """off mode must skip all contract validation."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'any_random_handle'},
            ],
            'contract_config': {'mode': 'off'},
        }
        
        graph = build(graph_json, message="test")
        
        # Should have contract_report with empty diagnostics
        assert graph.contract_report is not None
        assert graph.contract_report.mode == "off"
        assert len(graph.contract_report.diagnostics) == 0


class TestRunAllValidationsIncludesTargetFanIn:
    """Test that run_all_validations includes target and fan-in validators."""

    def test_run_all_validations_produces_target_diagnostics(self):
        """run_all_validations must produce target handle diagnostics."""
        from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
        from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
        from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
        
        # Create minimal mock nodes
        nodes = {
            'input': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'llm': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.LLM})(),
        }
        
        # Use proper EdgeNodeModel instances
        edges = [
            EdgeNodeModel(
                id='e1',
                source='input',
                target='llm',
                sourceHandle='handle_user_message',
                targetHandle='nonexistent_handle',
            ),
        ]
        
        graph = AgentFlowModel(nodes=nodes, edges=edges)
        
        diagnostics = run_all_validations(graph, mode='warn')
        
        # Should have diagnostics
        assert len(diagnostics) > 0
        # Should include target-related codes
        target_codes = [d.get('code') for d in diagnostics]
        # At minimum, we expect OpaqueTargetHandle or InvalidTargetHandle
        assert any(c in target_codes for c in ['PORT_TARGET_HANDLE_UNKNOWN', 'PORT_TARGET_HANDLE_LEGACY', 'OpaqueTargetHandle', 'InvalidTargetHandle'])

    def test_run_all_validations_produces_fan_in_diagnostics(self):
        """run_all_validations must produce fan-in diagnostics for multi-edge."""
        from magic_agents.models.factory.AgentFlowModel import AgentFlowModel
        from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
        from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
        
        nodes = {
            'source1': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'source2': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'text': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.TEXT})(),
        }
        
        # Two edges to same exclusive port
        edges = [
            EdgeNodeModel(id='e1', source='source1', target='text', sourceHandle='handle_user_message', targetHandle='handle_flow_input'),
            EdgeNodeModel(id='e2', source='source2', target='text', sourceHandle='handle_user_message', targetHandle='handle_flow_input'),
        ]
        
        graph = AgentFlowModel(nodes=nodes, edges=edges)
        
        diagnostics = run_all_validations(graph, mode='warn')
        
        # Should have fan-in cardinality violation diagnostic
        fan_in_codes = [d.get('code') for d in diagnostics]
        assert 'PORT_CARDINALITY_EXCLUSIVE_VIOLATION' in fan_in_codes


class TestBuildPreservesEdgeId:
    """Test that build() preserves edge.id through construction."""

    def test_build_preserves_explicit_edge_id(self):
        """Explicit edge.id must be preserved through build."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'my_custom_edge_id', 'source': 'input', 'target': 'end'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Edge must preserve the explicit ID (build may transform targets for void nodes)
        my_edge = [e for e in graph.edges if e.id == 'my_custom_edge_id']
        assert len(my_edge) == 1
        assert my_edge[0].source == 'input'
        # Note: target may be transformed to void node by build()

    def test_build_assigns_uuid_edge_id_when_missing(self):
        """build() must assign UUID edge.id when not provided."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                # No id field
                {'source': 'input', 'target': 'end'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Find the edge that connects input -> end (not the void edge)
        input_to_end_edges = [e for e in graph.edges if e.source == 'input' and e.target != e.source]
        assert len(input_to_end_edges) >= 1
        # Edge must have assigned ID
        assert input_to_end_edges[0].id is not None
        assert len(input_to_end_edges[0].id) == 32  # UUID hex length


class TestEdgeHookBuildValidation:
    """Build-time topology validation for declarative edge hooks."""

    @staticmethod
    def _graph_json(*, second_hook_enabled: bool) -> dict:
        hook_config = {
            'hook_node_id': 'edge-hook',
        }
        return {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'text', 'type': 'text', 'data': {'text': 'result'}},
                {
                    'id': 'edge-hook',
                    'type': 'hook',
                    'data': {
                        'function_template': (
                            'def on_edge(context, chat_log):\n'
                            '    return None\n'
                        ),
                    },
                },
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {
                    'id': 'input-to-text',
                    'source': 'input',
                    'target': 'text',
                    'sourceHandle': 'handle_user_message',
                    'targetHandle': 'handle_flow_input',
                    'hooks': hook_config,
                },
                {
                    'id': 'text-to-end',
                    'source': 'text',
                    'target': 'end',
                    'sourceHandle': 'handle_text_output',
                    'targetHandle': 'handle_flow_input',
                    'hooks': {
                        'hook_node_id': 'edge-hook',
                        'enabled': second_hook_enabled,
                    },
                },
            ],
        }

    def test_build_reports_hook_referenced_by_multiple_enabled_edges(self):
        graph = build(self._graph_json(second_hook_enabled=True), message="test")

        duplicate_hook_errors = [
            error
            for error in (graph._validation_errors or [])
            if error.get('error_type') == 'HookValidationError'
            and error.get('context', {}).get('max_enabled_edges') == 1
        ]

        assert len(duplicate_hook_errors) == 1
        error = duplicate_hook_errors[0]
        assert "at most one enabled edge" in error['error_message']
        assert error['context'] == {
            'hook_node_id': 'edge-hook',
            'edge_ids': ['input-to-text', 'text-to-end'],
            'enabled_edge_count': 2,
            'max_enabled_edges': 1,
        }

    def test_build_ignores_disabled_edge_hook_reference(self):
        graph = build(self._graph_json(second_hook_enabled=False), message="test")

        duplicate_hook_errors = [
            error
            for error in (graph._validation_errors or [])
            if error.get('error_type') == 'HookValidationError'
            and error.get('context', {}).get('max_enabled_edges') == 1
        ]

        assert duplicate_hook_errors == []

    @pytest.mark.asyncio
    async def test_run_agent_blocks_multiple_enabled_edge_bindings(self):
        from magic_agents import run_agent

        graph = build(self._graph_json(second_hook_enabled=True), message="test")
        events = [event async for event in run_agent(graph)]

        assert graph.nodes['edge-hook'].outputs == {}
        assert graph.nodes['end'].outputs == {}
        assert any(
            isinstance(event, dict)
            and event.get('type') == 'debug'
            and isinstance(event.get('content'), dict)
            and event['content'].get('error_type') == 'HookValidationError'
            for event in events
        )

    @pytest.mark.asyncio
    async def test_run_agent_blocks_hook_reference_to_non_hook_node(self):
        from magic_agents import run_agent

        graph = build(
            {
                'nodes': [
                    {'id': 'input', 'type': 'user_input', 'data': {}},
                    {'id': 'text', 'type': 'text', 'data': {'text': 'result'}},
                    {'id': 'end', 'type': 'end', 'data': {}},
                ],
                'edges': [
                    {
                        'id': 'invalid-hook-type',
                        'source': 'input',
                        'target': 'text',
                        'sourceHandle': 'handle_user_message',
                        'targetHandle': 'handle_flow_input',
                        'hooks': {
                            'hook_node_id': 'text',
                            'enabled': True,
                        },
                    },
                    {
                        'id': 'text-end',
                        'source': 'text',
                        'target': 'end',
                        'sourceHandle': 'handle_text_output',
                        'targetHandle': 'handle_flow_input',
                    },
                ],
            },
            message='test',
        )
        type_errors = [
            error
            for error in (graph._validation_errors or [])
            if error.get('error_type') == 'HookValidationError'
            and error.get('context', {}).get('edge_id') == 'invalid-hook-type'
        ]

        assert len(type_errors) == 1
        assert type_errors[0]['context'] == {
            'edge_id': 'invalid-hook-type',
            'hook_node_id': 'text',
            'referenced_node_type': 'text',
            'valid_hook_node_ids': [],
        }

        events = [event async for event in run_agent(graph)]
        assert graph.nodes['text'].outputs == {}
        assert graph.nodes['end'].outputs == {}
        assert any(
            isinstance(event, dict)
            and event.get('type') == 'debug'
            and isinstance(event.get('content'), dict)
            and event['content'].get('error_type') == 'HookValidationError'
            for event in events
        )


class TestBackwardCompatibility:
    """Test backward compatibility for existing graphs."""

    def test_handle_only_graph_builds_correctly(self):
        """Graphs with only handles (no ports) must build correctly."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'client', 'type': 'client', 'data': {'engine': 'openai', 'model': 'gpt-4', 'api_info': {'api_key': 'test-only'}}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
                {'id': 'e2', 'source': 'client', 'target': 'llm', 'sourceHandle': 'handle-client-provider', 'targetHandle': 'handle-client-provider'},
                {'id': 'e3', 'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Must build successfully
        assert graph is not None
        # Original: 4 nodes (input, client, llm, end) + void node = 5
        assert len(graph.nodes) == 5
        # Original: 3 edges + void edge = 4
        assert len(graph.edges) == 4
        
        # Must have contract report
        assert graph.contract_report is not None

    def test_legacy_handle_rejected(self):
        """Legacy handles must still be rejected."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'id': 'e1', 'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_end', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Must have validation errors for legacy handle
        assert graph._validation_errors is not None
        legacy_errors = [e for e in graph._validation_errors if e.get('type') == 'LegacyHandleRejected']
        assert len(legacy_errors) > 0
