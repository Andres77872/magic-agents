"""
Tests for fan-in compatibility validation (Phase 2).

Tests that:
1. Safe fan-in to declared multi-input port passes
2. Ambiguous fan-in produces advisory warning (PORT_FAN_IN_CARDINALITY_AMBIGUOUS)
3. Exclusive port with multiple edges flagged (PORT_CARDINALITY_EXCLUSIVE_VIOLATION)
4. Mixed-type fan-in allowed only with explicit multi-compatibility declaration
5. Runtime fan-in materialization semantics verified
"""

from magic_agents.util.handle_registry import (
    CardinalityInfo,
    get_port_cardinality,
    is_multi_compatible_port,
    ModelAgentFlowTypesModel,
)
from magic_agents.util.graph_validator import (
    validate_edge_fan_in_compatibility,
)
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel
from magic_agents.models.factory.AgentFlowModel import GraphContractReport


class TestCardinalityInfo:
    """Test CardinalityInfo dataclass."""

    def test_default_cardinality_ambiguous(self):
        """Default CardinalityInfo should be ambiguous."""
        info = CardinalityInfo()
        assert info.cardinality == "ambiguous"
        assert info.exclusive is True
        assert info.multi_compatible is False

    def test_multi_compatible_port(self):
        """Multi-compatible port has cardinality='many'."""
        info = CardinalityInfo(
            cardinality="many",
            exclusive=False,
            multi_compatible=True,
            merge_policy="collect"
        )
        assert info.cardinality == "many"
        assert info.multi_compatible is True

    def test_exclusive_port(self):
        """Exclusive port has cardinality='one' and exclusive=True."""
        info = CardinalityInfo(cardinality="one", exclusive=True)
        assert info.cardinality == "one"
        assert info.exclusive is True


class TestPortCardinalityRegistry:
    """Test PORT_CARDINALITY registry."""

    def test_llm_tool_definition_multi_compatible(self):
        """LLM tool-definition handle must be multi-compatible."""
        cardinality = get_port_cardinality(
            ModelAgentFlowTypesModel.LLM,
            'handle-tool-definition'
        )
        assert cardinality.multi_compatible is True
        assert cardinality.cardinality == "many"

    def test_llm_user_message_exclusive(self):
        """LLM user_message handle must be exclusive."""
        cardinality = get_port_cardinality(
            ModelAgentFlowTypesModel.LLM,
            'handle_user_message'
        )
        assert cardinality.exclusive is True
        assert cardinality.cardinality == "one"

    def test_parser_input_multi_compatible(self):
        """Parser input handle must be multi-compatible (templates)."""
        cardinality = get_port_cardinality(
            ModelAgentFlowTypesModel.PARSER,
            'handle_parser_input'
        )
        assert cardinality.multi_compatible is True

    def test_text_input_exclusive(self):
        """Text input handle must be exclusive."""
        cardinality = get_port_cardinality(
            ModelAgentFlowTypesModel.TEXT,
            'handle_flow_input'
        )
        assert cardinality.exclusive is True
        assert cardinality.cardinality == "one"

    def test_unknown_handle_returns_ambiguous(self):
        """Unknown handle should return ambiguous cardinality."""
        cardinality = get_port_cardinality(
            ModelAgentFlowTypesModel.LLM,
            'unknown_handle_xyz'
        )
        assert cardinality.cardinality == "ambiguous"

    def test_dynamic_handle_matches_base(self):
        """Dynamic handle should match base pattern cardinality."""
        cardinality = get_port_cardinality(
            ModelAgentFlowTypesModel.LLM,
            'handle-tool-definition-0'
        )
        # Should match 'handle-tool-definition' base pattern
        assert cardinality.multi_compatible is True


class TestIsMultiCompatiblePort:
    """Test is_multi_compatible_port function."""

    def test_multi_compatible_returns_true(self):
        """Declared multi-compatible port returns True."""
        assert is_multi_compatible_port(ModelAgentFlowTypesModel.LLM, 'handle-tool-definition') is True

    def test_exclusive_returns_false(self):
        """Exclusive port returns False."""
        assert is_multi_compatible_port(ModelAgentFlowTypesModel.LLM, 'handle_user_message') is False

    def test_ambiguous_returns_false(self):
        """Ambiguous port returns False."""
        assert is_multi_compatible_port(ModelAgentFlowTypesModel.LLM, 'unknown_handle') is False


class TestValidateEdgeFanInCompatibility:
    """Test validate_edge_fan_in_compatibility function."""

    def test_single_edge_no_warning(self):
        """Single edge to a handle should produce no warning."""
        nodes = {
            'source': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'text': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.TEXT})(),
        }
        edges = [
            EdgeNodeModel(
                id='e1',
                source='source',
                target='text',
                sourceHandle='handle_user_message',
                targetHandle='handle_flow_input'
            )
        ]
        
        errors = validate_edge_fan_in_compatibility(nodes, edges, mode="warn")
        
        # No fan-in warnings for single edge
        fan_in_errors = [e for e in errors if 'FanIn' in e.get('type', '') or 'Cardinality' in e.get('type', '')]
        assert len(fan_in_errors) == 0

    def test_multi_compatible_port_multiple_edges_passes(self):
        """Two edges into the SAME multi-compatible handle are a fan-in and pass."""
        nodes = {
            'tool1': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.PYTHON_EXEC})(),
            'tool2': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.MCP})(),
            'llm': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.LLM})(),
        }
        edges = [
            EdgeNodeModel(
                id='e1',
                source='tool1',
                target='llm',
                sourceHandle='handle-tool-definition',
                targetHandle='handle-tool-definition-0'
            ),
            EdgeNodeModel(
                id='e2',
                source='tool2',
                target='llm',
                sourceHandle='handle-tool-definition',
                targetHandle='handle-tool-definition-0'
            ),
        ]
        
        errors = validate_edge_fan_in_compatibility(nodes, edges, mode="warn")
        
        assert errors == []

    def test_exclusive_port_multiple_edges_warning(self):
        """Exclusive port with multiple edges should produce warning."""
        nodes = {
            'source1': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'source2': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'text': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.TEXT})(),
        }
        edges = [
            EdgeNodeModel(
                id='e1',
                source='source1',
                target='text',
                sourceHandle='handle_user_message',
                targetHandle='handle_flow_input'
            ),
            EdgeNodeModel(
                id='e2',
                source='source2',
                target='text',
                sourceHandle='handle_user_message',
                targetHandle='handle_flow_input'
            ),
        ]
        
        errors = validate_edge_fan_in_compatibility(nodes, edges, mode="warn")
        
        # Should have exclusive violation warning
        violation_errors = [e for e in errors if e.get('code') == 'PORT_CARDINALITY_EXCLUSIVE_VIOLATION']
        assert len(violation_errors) >= 1
        assert violation_errors[0]['severity'] == 'warning'

    def test_ambiguous_port_multiple_edges_advisory(self):
        """Ambiguous port with multiple edges should produce advisory warning."""
        nodes = {
            'source1': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'source2': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'target': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.LLM})(),
        }
        # Using an undeclared handle to test ambiguous case
        edges = [
            EdgeNodeModel(
                id='e1',
                source='source1',
                target='target',
                sourceHandle='handle_user_message',
                targetHandle='undeclared_handle_xyz'
            ),
            EdgeNodeModel(
                id='e2',
                source='source2',
                target='target',
                sourceHandle='handle_user_message',
                targetHandle='undeclared_handle_xyz'
            ),
        ]
        
        errors = validate_edge_fan_in_compatibility(nodes, edges, mode="warn")
        
        # Should have ambiguous cardinality advisory
        ambiguous_errors = [e for e in errors if e.get('code') == 'PORT_FAN_IN_CARDINALITY_AMBIGUOUS']
        assert len(ambiguous_errors) >= 1

    def test_error_includes_edge_ids(self):
        """Fan-in error must include all edge_ids involved."""
        nodes = {
            'source1': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'source2': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'text': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.TEXT})(),
        }
        edges = [
            EdgeNodeModel(id='edge_alpha', source='source1', target='text', targetHandle='handle_flow_input'),
            EdgeNodeModel(id='edge_beta', source='source2', target='text', targetHandle='handle_flow_input'),
        ]
        
        errors = validate_edge_fan_in_compatibility(nodes, edges, mode="warn")

        assert len(errors) == 1
        assert set(errors[0]['edge_ids']) == {'edge_alpha', 'edge_beta'}


class TestGraphContractReport:
    """Test GraphContractReport model."""

    def test_empty_report(self):
        """Empty report has no diagnostics."""
        report = GraphContractReport()
        assert len(report.diagnostics) == 0
        assert report.has_errors() is False

    def test_add_diagnostic(self):
        """Can add diagnostic to report."""
        report = GraphContractReport()
        report.add_diagnostic({
            "type": "TestError",
            "severity": "error",
            "code": "TEST_ERROR",
        })
        assert len(report.diagnostics) == 1
        assert report.has_errors() is True

    def test_get_errors_warnings_separated(self):
        """Can separate errors and warnings."""
        report = GraphContractReport()
        report.add_diagnostic({"severity": "error", "code": "E1"})
        report.add_diagnostic({"severity": "warning", "code": "W1"})
        report.add_diagnostic({"severity": "warning", "code": "W2"})
        
        assert len(report.get_errors()) == 1
        assert len(report.get_warnings()) == 2
        assert len(report.get_info()) == 0

    def test_to_normalized_dict(self):
        """to_normalized_dict produces CI-stable output."""
        report = GraphContractReport(
            mode="warn",
            edge_count=5,
            node_count=3,
        )
        report.add_diagnostic({"severity": "warning", "code": "CODE_B", "edge_id": "e2"})
        report.add_diagnostic({"severity": "warning", "code": "CODE_A", "edge_id": "e1"})
        
        normalized = report.to_normalized_dict()
        
        # Should not contain timestamp
        assert 'timestamp' not in normalized
        # Diagnostics should be sorted by code
        assert normalized['diagnostics'][0]['code'] == 'CODE_A'
        assert normalized['diagnostics'][1]['code'] == 'CODE_B'


class TestAgentFlowModelContractConfig:
    """Test AgentFlowModel with ContractConfig."""

    def test_model_has_contract_config(self):
        """AgentFlowModel should have contract_config field."""
        from magic_agents.agt_flow import build
        
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'end', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        assert hasattr(graph, 'contract_config')
        assert graph.contract_config.mode == "warn"

    def test_model_can_get_contract_errors(self):
        """AgentFlowModel can get contract errors."""
        from magic_agents.agt_flow import build
        
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'end', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Should be able to get errors/warnings
        errors = graph.get_contract_errors()
        warnings = graph.get_contract_warnings()
        
        assert isinstance(errors, list)
        assert isinstance(warnings, list)


