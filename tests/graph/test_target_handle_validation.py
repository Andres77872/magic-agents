"""
Tests for target handle validation (Phase 1).

Tests that:
1. Valid canonical targetHandle passes validation
2. Invalid targetHandle produces diagnostic with code PORT_TARGET_HANDLE_UNKNOWN
3. Dynamic pattern handles (tool-definition-{N}) recognized as valid
4. Legacy/undeclared handles pass in shadow/warn mode with warning
5. validate_edge_target_handles() integrated in run_all_validations()
"""

from magic_agents.agt_flow import build
from magic_agents.util.handle_registry import (
    is_valid_target_handle,
    get_canonical_input_handles,
)
from magic_agents.util.graph_validator import (
    validate_edge_target_handles,
    run_all_validations,
)
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from magic_agents.models.factory.EdgeNodeModel import EdgeNodeModel




class TestCanonicalInputHandlesRegistry:
    """Test canonical input handles registry."""

    def test_llm_input_handles(self):
        """LLM must have input handles defined."""
        llm_inputs = get_canonical_input_handles(ModelAgentFlowTypesModel.LLM)
        assert 'handle_user_message' in llm_inputs
        assert 'handle-client-provider' in llm_inputs  # Actual client input handle
        assert 'handle-tool-definition' in llm_inputs

    def test_text_input_handles(self):
        """Text node must have input handle defined."""
        text_inputs = get_canonical_input_handles(ModelAgentFlowTypesModel.TEXT)
        assert 'handle_flow_input' in text_inputs

    def test_parser_input_handles(self):
        """Parser must have input handle defined."""
        parser_inputs = get_canonical_input_handles(ModelAgentFlowTypesModel.PARSER)
        assert 'handle_parser_input' in parser_inputs

    def test_user_input_has_no_inputs(self):
        """User input node has no inputs (source node)."""
        user_inputs = get_canonical_input_handles(ModelAgentFlowTypesModel.USER_INPUT)
        assert len(user_inputs) == 0

    def test_void_accepts_anything(self):
        """VOID node accepts any handle (sink)."""
        void_inputs = get_canonical_input_handles(ModelAgentFlowTypesModel.VOID)
        assert len(void_inputs) == 0  # No restriction


class TestIsValidTargetHandle:
    """Test is_valid_target_handle validation function."""

    def test_canonical_llm_input_passes(self):
        """Canonical LLM input handle should pass."""
        is_valid, reason = is_valid_target_handle(
            'handle_user_message',
            ModelAgentFlowTypesModel.LLM
        )
        assert is_valid is True
        assert 'Canonical' in reason

    def test_canonical_text_input_passes(self):
        """Canonical text input handle should pass."""
        is_valid, reason = is_valid_target_handle(
            'handle_flow_input',
            ModelAgentFlowTypesModel.TEXT
        )
        assert is_valid is True
        assert 'Canonical' in reason

    def test_dynamic_tool_handle_passes(self):
        """Dynamic tool handle pattern should pass."""
        is_valid, reason = is_valid_target_handle(
            'handle-tool-definition-0',
            ModelAgentFlowTypesModel.LLM
        )
        assert is_valid is True
        assert 'Dynamic' in reason

    def test_dynamic_parser_input_passes(self):
        """Dynamic parser input pattern should pass."""
        is_valid, reason = is_valid_target_handle(
            'handle_parser_input_0',
            ModelAgentFlowTypesModel.PARSER
        )
        assert is_valid is True
        assert 'Dynamic' in reason

    def test_void_accepts_any_handle(self):
        """VOID node should accept any target handle."""
        is_valid, reason = is_valid_target_handle(
            'any_random_handle',
            ModelAgentFlowTypesModel.VOID
        )
        assert is_valid is True
        assert 'Void' in reason

    def test_unknown_handle_warn_mode(self):
        """Unknown handle should pass with warning in warn mode."""
        is_valid, reason = is_valid_target_handle(
            'unknown_handle_xyz',
            ModelAgentFlowTypesModel.LLM,
            mode="warn"
        )
        assert is_valid is True
        assert 'Opaque' in reason or 'legacy' in reason

    def test_unknown_handle_strict_mode_deferred(self):
        """Unknown handle rejected in strict mode (deferred behavior)."""
        is_valid, reason = is_valid_target_handle(
            'unknown_handle_xyz',
            ModelAgentFlowTypesModel.LLM,
            mode="strict"
        )
        # In strict mode, unknown handles should be rejected
        assert is_valid is False


class TestValidateEdgeTargetHandles:
    """Test validate_edge_target_handles function."""

    def test_valid_target_handle_no_error(self):
        """Valid targetHandle should produce no error."""
        # Create a minimal graph
        nodes = {
            'input': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'llm': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.LLM})(),
        }
        edges = [
            EdgeNodeModel(
                id='e1',
                source='input',
                target='llm',
                sourceHandle='handle_user_message',
                targetHandle='handle_user_message'
            )
        ]
        
        errors = validate_edge_target_handles(nodes, edges, mode="warn")
        
        # No errors for valid targetHandle
        target_errors = [e for e in errors if e.get('type') == 'InvalidTargetHandle']
        assert len(target_errors) == 0

    def test_unknown_target_handle_produces_warning(self):
        """Unknown targetHandle should produce warning in warn mode."""
        nodes = {
            'input': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'llm': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.LLM})(),
        }
        edges = [
            EdgeNodeModel(
                id='e1',
                source='input',
                target='llm',
                sourceHandle='handle_user_message',
                targetHandle='unknown_input_handle'  # Unknown
            )
        ]
        
        errors = validate_edge_target_handles(nodes, edges, mode="warn")
        
        # Should produce opaque/legacy warning
        opaque_errors = [e for e in errors if e.get('type') == 'OpaqueTargetHandle']
        assert len(opaque_errors) >= 1

    def test_error_includes_edge_id(self):
        """Target validation error must include edge_id."""
        nodes = {
            'input': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'llm': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.LLM})(),
        }
        edges = [
            EdgeNodeModel(
                id='my_edge_id',
                source='input',
                target='llm',
                sourceHandle='handle_user_message',
                targetHandle='unknown_input_handle'
            )
        ]
        
        errors = validate_edge_target_handles(nodes, edges, mode="warn")

        assert len(errors) == 1
        assert errors[0]['edge_id'] == 'my_edge_id'

    def test_error_includes_code(self):
        """Target validation error must include diagnostic code."""
        nodes = {
            'input': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.USER_INPUT})(),
            'llm': type('MockNode', (), {'node_type': ModelAgentFlowTypesModel.LLM})(),
        }
        edges = [
            EdgeNodeModel(
                id='e1',
                source='input',
                target='llm',
                sourceHandle='handle_user_message',
                targetHandle='unknown_input_handle'
            )
        ]
        
        errors = validate_edge_target_handles(nodes, edges, mode="warn")

        assert len(errors) == 1
        assert errors[0]['code'] in (
            'PORT_TARGET_HANDLE_UNKNOWN',
            'PORT_TARGET_HANDLE_LEGACY',
            'PORT_TARGET_HANDLE_OPAQUE',
        )


class TestRunAllValidationsIncludesTarget:
    """Test run_all_validations includes targetHandle validation."""

    def test_run_all_validations_includes_target(self):
        """run_all_validations should call validate_edge_target_handles."""
        # Build a simple graph - note: handle-client-provider is the actual input handle for LLM
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'client', 'type': 'client', 'data': {'engine': 'openai', 'model': 'test', 'api_info': {'api_key': 'test-only'}}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
                {'source': 'client', 'target': 'llm', 'sourceHandle': 'handle-client-provider', 'targetHandle': 'handle-client-provider'},  # Correct handle
                {'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Run all validations
        errors = run_all_validations(graph, mode="warn")
        
        # No errors for valid graph (check for InvalidTargetHandle, not OpaqueTargetHandle)
        target_errors = [e for e in errors if e.get('type') == 'InvalidTargetHandle']
        assert len(target_errors) == 0


class TestBuildIntegration:
    """Test build() integration with target validation."""

    def test_build_accepts_valid_target_handles(self):
        """build() should accept graphs with valid targetHandles."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'text', 'type': 'text', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'text', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_flow_input'},
                {'source': 'text', 'target': 'end', 'sourceHandle': 'handle_text_output', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Graph should have no validation errors
        handle_errors = [
            e for e in (getattr(graph, '_validation_errors', None) or [])
            if e.get('type') in ('InvalidTargetHandle', 'InvalidSourceHandle')
        ]
        assert len(handle_errors) == 0

    def test_build_with_unknown_target_handle_shadow_mode(self):
        """build() in shadow/warn mode should accept unknown targetHandles."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'custom_unknown_handle'},
                {'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")

        # Accepted in warn mode, but surfaced as exactly one contract warning.
        assert [(d["code"], d["severity"]) for d in graph.contract_report.diagnostics] == [
            ("PORT_TARGET_HANDLE_UNKNOWN", "warning"),
        ]


class TestDynamicTargetHandles:
    """Test dynamic target handle patterns."""

    def test_tool_definition_handle_valid(self):
        """handle-tool-definition-{N} should be valid for LLM."""
        for i in range(5):
            is_valid, reason = is_valid_target_handle(
                f'handle-tool-definition-{i}',
                ModelAgentFlowTypesModel.LLM
            )
            assert is_valid is True

    def test_parser_input_handle_valid(self):
        """handle_parser_input_{N} should be valid for Parser."""
        for i in range(5):
            is_valid, reason = is_valid_target_handle(
                f'handle_parser_input_{i}',
                ModelAgentFlowTypesModel.PARSER
            )
            assert is_valid is True

    def test_build_with_tool_handles(self):
        """build() should accept dynamic tool handles."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'client', 'type': 'client', 'data': {'engine': 'openai', 'model': 'test', 'api_info': {'api_key': 'test-only'}}},
                {'id': 'python', 'type': 'python_exec', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
                {'source': 'client', 'target': 'llm', 'sourceHandle': 'handle-client-provider', 'targetHandle': 'handle-client-provider'},
                {'source': 'python', 'target': 'llm', 'sourceHandle': 'handle-tool-definition', 'targetHandle': 'handle-tool-definition-0'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Dynamic tool slots are canonical: no contract diagnostic at all.
        assert graph.contract_report.diagnostics == []
