"""Tests for edge handle validation (clean-break policy).

Tests that:
1. Legacy handles (handle_generated_end) are rejected
2. Canonical handles are accepted
3. Error output aligns with existing invalid-graph validation behavior
"""

from magic_agents.agt_flow import build
from magic_agents.util.handle_registry import (
    is_legacy_rejected_handle,
    is_valid_source_handle,
    get_node_output_handles_from_instance,
    LEGACY_REJECTED_HANDLES,
    CANONICAL_OUTPUT_HANDLES,
)
from magic_agents.models.factory.Nodes import ModelAgentFlowTypesModel
from magic_agents.node_system import NodeLLM, NodeParser, NodeSendMessage, NodeConstant
from magic_agents.models.factory.Nodes import LlmNodeModel, ParserNodeModel, SendMessageNodeModel, ConstantNodeModel




class TestHandleRegistryLegacyRejection:
    """Test legacy handle rejection in the registry."""

    def test_handle_generated_end_is_legacy_rejected(self):
        """handle_generated_end must be in LEGACY_REJECTED_HANDLES."""
        assert 'handle_generated_end' in LEGACY_REJECTED_HANDLES
        assert is_legacy_rejected_handle('handle_generated_end')

    def test_other_legacy_handles_not_in_canonical(self):
        """Legacy handles should not appear in canonical handles for any node type."""
        for node_type, handles in CANONICAL_OUTPUT_HANDLES.items():
            assert 'handle_generated_end' not in handles
            for handle in handles:
                assert handle != 'handle_generated_end'

    def test_is_legacy_rejected_handle_function(self):
        """is_legacy_rejected_handle should return True for legacy handles."""
        assert is_legacy_rejected_handle('handle_generated_end') is True
        assert is_legacy_rejected_handle('handle_generated_content') is False
        assert is_legacy_rejected_handle('handle_parser_output') is False


class TestHandleRegistryCanonicalHandles:
    """Test canonical handle definitions."""

    def test_llm_canonical_handles(self):
        """NodeLLM should have canonical handles defined."""
        llm_handles = CANONICAL_OUTPUT_HANDLES[ModelAgentFlowTypesModel.LLM]
        assert 'handle_generated_content' in llm_handles
        assert 'handle_streaming_content' in llm_handles
        assert 'handle-tool-calls' in llm_handles

    def test_parser_canonical_handle(self):
        """NodeParser should have canonical handle defined."""
        parser_handles = CANONICAL_OUTPUT_HANDLES[ModelAgentFlowTypesModel.PARSER]
        assert 'handle_parser_output' in parser_handles

    def test_constant_canonical_handle(self):
        """NodeConstant should have canonical handle defined."""
        constant_handles = CANONICAL_OUTPUT_HANDLES[ModelAgentFlowTypesModel.CONSTANT]
        assert 'handle_constant_output' in constant_handles

    def test_send_message_canonical_handle(self):
        """NodeSendMessage should have canonical handle defined."""
        send_handles = CANONICAL_OUTPUT_HANDLES[ModelAgentFlowTypesModel.SEND_MESSAGE]
        assert 'handle_message_output' in send_handles

    def test_end_node_has_output_handle(self):
        """NodeEND should have canonical output handle."""
        end_handles = CANONICAL_OUTPUT_HANDLES[ModelAgentFlowTypesModel.END]
        assert 'handle_end_output' in end_handles

    def test_void_node_has_no_outputs(self):
        """Void node should have no output handles."""
        void_handles = CANONICAL_OUTPUT_HANDLES[ModelAgentFlowTypesModel.VOID]
        assert len(void_handles) == 0


class TestIsValidSourceHandle:
    """Test is_valid_source_handle validation function."""

    def test_legacy_handle_always_fails(self):
        """Legacy handle must fail regardless of node type."""
        is_valid, reason = is_valid_source_handle('handle_generated_end', ModelAgentFlowTypesModel.LLM)
        assert is_valid is False
        assert 'Legacy handle' in reason
        assert 'handle_generated_end' in reason

    def test_legacy_handle_fails_for_parser(self):
        """Legacy handle must fail for parser node."""
        is_valid, reason = is_valid_source_handle('handle_generated_end', ModelAgentFlowTypesModel.PARSER)
        assert is_valid is False
        assert 'Legacy handle' in reason

    def test_canonical_llm_handle_passes(self):
        """Canonical LLM handle should pass."""
        is_valid, reason = is_valid_source_handle('handle_generated_content', ModelAgentFlowTypesModel.LLM)
        assert is_valid is True

    def test_canonical_parser_handle_passes(self):
        """Canonical parser handle should pass."""
        is_valid, reason = is_valid_source_handle('handle_parser_output', ModelAgentFlowTypesModel.PARSER)
        assert is_valid is True

    def test_canonical_send_message_handle_passes(self):
        """Canonical send_message handle should pass."""
        is_valid, reason = is_valid_source_handle('handle_message_output', ModelAgentFlowTypesModel.SEND_MESSAGE)
        assert is_valid is True

    def test_dynamic_tool_handle_passes(self):
        """Dynamic tool handle pattern should pass."""
        is_valid, reason = is_valid_source_handle('handle-tool-definition-0', ModelAgentFlowTypesModel.LLM)
        assert is_valid is True
        assert 'Dynamic' in reason

    def test_dynamic_parser_input_handle_passes(self):
        """Dynamic parser input handle pattern should pass."""
        # Parser input handles are used as targetHandle, but sourceHandle validation
        # should accept them as dynamic patterns
        is_valid, reason = is_valid_source_handle('handle_parser_input_0', ModelAgentFlowTypesModel.PARSER)
        assert is_valid is True
        assert 'Dynamic' in reason

    def test_conditional_dynamic_passes(self):
        """Conditional node without instance handles should pass (dynamic)."""
        is_valid, reason = is_valid_source_handle('yes', ModelAgentFlowTypesModel.CONDITIONAL)
        assert is_valid is True
        assert 'Conditional' in reason

    def test_void_node_rejects_any_handle(self):
        """Void node should reject any source handle."""
        is_valid, reason = is_valid_source_handle('any_handle', ModelAgentFlowTypesModel.VOID)
        assert is_valid is False
        assert 'Void' in reason

    def test_instance_handle_passes(self):
        """Custom instance handle should pass when provided."""
        # NodeLLM with custom output handle
        model = LlmNodeModel()
        node = NodeLLM(data=model, node_id='test', handles={'output_generated': 'custom_output'})
        instance_handles = get_node_output_handles_from_instance(node)
        
        is_valid, reason = is_valid_source_handle(
            'custom_output',
            ModelAgentFlowTypesModel.LLM,
            instance_handles
        )
        assert is_valid is True
        assert 'instance' in reason.lower()


class TestGetNodeOutputHandlesFromInstance:
    """Test extracting output handles from node instances."""

    def test_llm_instance_handles(self):
        """NodeLLM instance should have multiple output handles."""
        model = LlmNodeModel()
        node = NodeLLM(data=model, node_id='test_llm', debug=False)
        handles = get_node_output_handles_from_instance(node)
        
        assert 'handle_generated_content' in handles
        assert 'handle_streaming_content' in handles
        assert 'handle-tool-calls' in handles

    def test_parser_instance_handle(self):
        """NodeParser instance should have output handle."""
        model = ParserNodeModel(text="test")
        node = NodeParser(data=model, node_id='test_parser', debug=False)
        handles = get_node_output_handles_from_instance(node)
        
        assert 'handle_parser_output' in handles

    def test_send_message_instance_handle(self):
        """NodeSendMessage instance should have output handle."""
        model = SendMessageNodeModel(message="test")
        node = NodeSendMessage(data=model, node_id='test_send', debug=False)
        handles = get_node_output_handles_from_instance(node)
        
        assert 'handle_message_output' in handles

    def test_constant_instance_handle(self):
        """NodeConstant instance should have output handle."""
        model = ConstantNodeModel(value_type='int', value='7')
        node = NodeConstant(data=model, node_id='test_constant', debug=False)
        handles = get_node_output_handles_from_instance(node)

        assert 'handle_constant_output' in handles

    def test_custom_handle_in_instance(self):
        """Custom handle from data.handles should be in instance handles."""
        model = LlmNodeModel()
        node = NodeLLM(
            data=model,
            node_id='test_llm',
            handles={'output_generated': 'my_custom_handle'}
        )
        handles = get_node_output_handles_from_instance(node)
        
        assert 'my_custom_handle' in handles


class TestBuildRejectsLegacyHandles:
    """Test that build() rejects graphs with legacy handles."""

    def test_build_rejects_llm_to_end_with_legacy_handle(self):
        """Graph with legacy handle_generated_end from LLM should fail validation."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'client', 'type': 'client', 'data': {'engine': 'openai', 'model': 'test', 'api_info': {'api_key': 'test-only'}}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
                {'source': 'client', 'target': 'llm', 'sourceHandle': 'handle-client-provider', 'targetHandle': 'handle-client-provider'},
                # Legacy handle - should be rejected
                {'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_end', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Graph should have validation errors
        assert hasattr(graph, '_validation_errors')
        assert graph._validation_errors is not None
        assert len(graph._validation_errors) > 0
        
        # Find the legacy handle error
        legacy_errors = [
            e for e in graph._validation_errors
            if e.get('type') == 'LegacyHandleRejected'
        ]
        assert len(legacy_errors) > 0
        
        # Check error content
        legacy_error = legacy_errors[0]
        assert 'handle_generated_end' in legacy_error.get('sourceHandle', '')
        # Check for legacy handle mention (case-insensitive)
        error_msg = legacy_error.get('error_message', '')
        assert 'legacy handle' in error_msg.lower()

    def test_build_rejects_parser_to_end_with_legacy_handle(self):
        """Graph with legacy handle_generated_end from parser should fail validation."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'parser', 'type': 'parser', 'data': {'text': 'test'}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'parser', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_parser_input_0'},
                # Legacy handle - should be rejected
                {'source': 'parser', 'target': 'end', 'sourceHandle': 'handle_generated_end', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Graph should have validation errors
        assert hasattr(graph, '_validation_errors')
        assert graph._validation_errors is not None
        
        # Find the legacy handle error
        legacy_errors = [
            e for e in graph._validation_errors
            if e.get('type') == 'LegacyHandleRejected'
        ]
        assert len(legacy_errors) > 0

    def test_build_rejects_send_message_with_legacy_handle(self):
        """Graph with legacy handle_generated_end from send_message should fail."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'send', 'type': 'send_message', 'data': {'message': 'test'}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'send', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_send_extra'},
                # Legacy handle - should be rejected
                {'source': 'send', 'target': 'end', 'sourceHandle': 'handle_generated_end', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Graph should have validation errors
        assert hasattr(graph, '_validation_errors')
        assert graph._validation_errors is not None
        
        # Find the legacy handle error
        legacy_errors = [
            e for e in graph._validation_errors
            if e.get('type') == 'LegacyHandleRejected'
        ]
        assert len(legacy_errors) > 0


class TestBuildAcceptsCanonicalHandles:
    """Test that build() accepts graphs with canonical handles."""

    def test_build_accepts_llm_to_end_with_canonical_handle(self):
        """Graph with canonical handle_generated_content from LLM should pass."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'client', 'type': 'client', 'data': {'engine': 'openai', 'model': 'test', 'api_info': {'api_key': 'test-only'}}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'llm', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_user_message'},
                {'source': 'client', 'target': 'llm', 'sourceHandle': 'handle-client-provider', 'targetHandle': 'handle-client-provider'},
                # Canonical handle - should be accepted
                {'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_content', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Graph should NOT have handle validation errors
        handle_errors = [
            e for e in (getattr(graph, '_validation_errors', None) or [])
            if e.get('type') in ('LegacyHandleRejected', 'InvalidSourceHandle')
        ]
        assert len(handle_errors) == 0

    def test_build_accepts_parser_with_canonical_handle(self):
        """Graph with canonical handle_parser_output should pass."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'parser', 'type': 'parser', 'data': {'text': 'test'}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'parser', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_parser_input_0'},
                # Canonical handle - should be accepted
                {'source': 'parser', 'target': 'end', 'sourceHandle': 'handle_parser_output', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Graph should NOT have handle validation errors
        handle_errors = [
            e for e in (getattr(graph, '_validation_errors', None) or [])
            if e.get('type') in ('LegacyHandleRejected', 'InvalidSourceHandle')
        ]
        assert len(handle_errors) == 0

    def test_build_accepts_send_message_with_canonical_handle(self):
        """Graph with canonical handle_message_output should pass."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'send', 'type': 'send_message', 'data': {'message': 'test'}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'input', 'target': 'send', 'sourceHandle': 'handle_user_message', 'targetHandle': 'handle_send_extra'},
                # Canonical handle - should be accepted
                {'source': 'send', 'target': 'end', 'sourceHandle': 'handle_message_output', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Graph should NOT have handle validation errors
        handle_errors = [
            e for e in (getattr(graph, '_validation_errors', None) or [])
            if e.get('type') in ('LegacyHandleRejected', 'InvalidSourceHandle')
        ]
        assert len(handle_errors) == 0

    def test_build_accepts_dynamic_tool_handles(self):
        """Graph with dynamic tool handles should pass."""
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
                # Dynamic tool handle - should be accepted
                {'source': 'python', 'target': 'llm', 'sourceHandle': 'handle-tool-definition', 'targetHandle': 'handle-tool-definition-0'},
            ]
        }
        
        graph = build(graph_json, message="test")
        
        # Graph should NOT have handle validation errors for the tool handle
        handle_errors = [
            e for e in (getattr(graph, '_validation_errors', None) or [])
            if e.get('type') in ('LegacyHandleRejected', 'InvalidSourceHandle')
        ]
        assert len(handle_errors) == 0


class TestErrorMessagesAlignWithContract:
    """Test error messages align with JSON_CONTRACT.md documentation."""

    def test_legacy_error_suggests_documentation(self):
        """Legacy handle error should suggest JSON_CONTRACT.md."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_end', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        legacy_errors = [
            e for e in (getattr(graph, '_validation_errors', None) or [])
            if e.get('type') == 'LegacyHandleRejected'
        ]
        
        assert len(legacy_errors) == 1
        suggestion = legacy_errors[0].get('suggestion', '')
        assert 'JSON_CONTRACT.md' in suggestion or 'canonical handles' in suggestion.lower()

    def test_error_includes_node_type(self):
        """Error should include node type for debugging."""
        graph_json = {
            'nodes': [
                {'id': 'input', 'type': 'user_input', 'data': {}},
                {'id': 'llm', 'type': 'llm', 'data': {}},
                {'id': 'end', 'type': 'end', 'data': {}},
            ],
            'edges': [
                {'source': 'llm', 'target': 'end', 'sourceHandle': 'handle_generated_end', 'targetHandle': 'handle_flow_input'},
            ]
        }
        
        graph = build(graph_json, message="test")
        legacy_errors = [
            e for e in (getattr(graph, '_validation_errors', None) or [])
            if e.get('type') == 'LegacyHandleRejected'
        ]
        
        assert len(legacy_errors) == 1
        assert legacy_errors[0]['node_type'] == ModelAgentFlowTypesModel.LLM
